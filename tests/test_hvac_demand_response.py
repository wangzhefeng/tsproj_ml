"""需求响应常态估计：范围保真、过去原观测选型和真实发布边界。"""
import unittest
import json
from pathlib import Path
import os
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from config.aidc_hvac_load_5min.scripts.v3.outlier_remove_data.adjust_demand_response import correct_points, prepare
from config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data import sha256_file

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'config/aidc_hvac_load_5min/scripts/v3/outlier_remove_data/adjust_demand_response.py'


def fixture(root):
    index = pd.date_range('2026-01-01', periods=40 * 288, freq='5min', name='time')
    start, rebound, end = index[35 * 288], index[35 * 288 + 12], index[35 * 288 + 24]
    paths = {}
    for devices in ('hvac_all_devices', 'hvac_remove_devices'):
        totals = {}
        for route, value in (('route_A', 10.), ('route_B', 20.)):
            frame = pd.DataFrame({'shared': value}, index=index)
            frame.loc[start:rebound - pd.Timedelta(minutes=5), 'shared'] = 0.
            frame.loc[rebound:end - pd.Timedelta(minutes=5), 'shared'] = 2 * value
            if devices == 'hvac_all_devices':
                frame['pump'] = 3.
            frame['total_load'] = frame.sum(axis=1)
            totals[route] = frame.total_load
            for category in ('raw_data', 'imputed_data/data_v3'):
                paths[f'{category}/{devices}/{route}/A1_data.csv'] = frame
        target = pd.DataFrame({'hvac_total_load_AB': totals['route_A'] + totals['route_B'],
                               'hvac_total_load_A': totals['route_A'], 'hvac_total_load_B': totals['route_B'],
                               'it_subset_load': 5.}, index=index)
        paths[f'forecast_data/data_v3/{devices}/A1_all/data.csv'] = target
        mask = pd.DataFrame(index=index)
        for col in target:
            mask[col + '_observed'] = True
            mask[col + '_eligibility_known_at'] = index
        paths[f'analysis/data_v3/forecast_windows/{devices}/A1_all/mask.csv'] = mask
    for path, frame in paths.items():
        dest = root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(dest)
    return {'kind': 'hvac_demand_response_v1', 'purpose': 'synthetic test',
            'events': [{'event_id': 'dr_test', 'start': str(start), 'rebound_start': str(rebound), 'end_exclusive': str(end)}],
            'input_sha256': {p: sha256_file(root / p) for p in paths}}


class DemandResponseTest(unittest.TestCase):
    def test_event_reconstruction_preserves_outside_and_ignores_future(self):
        index = pd.date_range('2026-01-01', periods=40 * 288, freq='5min', name='time')
        normal = 100. + np.arange(len(index)) % 288
        raw = pd.DataFrame({'chiller': normal, 'pump': 3.}, index=index)
        start, end = index[35 * 288 + 100], index[35 * 288 + 124]
        event = {'event_id': 'dr_test', 'start': str(start),
                 'rebound_start': str(start + pd.Timedelta(hours=1)), 'end_exclusive': str(end)}
        scope = (index >= start) & (index < end)
        raw.loc[scope, 'chiller'] = 0.
        raw.loc[index[35 * 288 + 112]:index[35 * 288 + 123], 'chiller'] = 999.
        raw.loc[index[35 * 288 + 105], 'chiller'] = np.nan
        parent = raw.fillna(777.)
        # 旧的窗外估计也必须保持，不能顺带全量重填。
        raw.loc[index[34 * 288 + 20], 'pump'] = np.nan
        parent.loc[index[34 * 288 + 20], 'pump'] = 88.
        output, methods, cells, blocks = correct_points(raw, parent, [event])
        np.testing.assert_array_equal(output.loc[scope, 'chiller'], normal[scope])
        np.testing.assert_array_equal(output.loc[scope, 'pump'], 3.)
        pd.testing.assert_frame_equal(output.loc[~scope], parent.loc[~scope])
        self.assertEqual(methods.query("point == 'chiller'").iloc[0]['method'], 'past_day_repeat')
        self.assertEqual(len(cells), 48)
        self.assertEqual(int((~cells.raw_observed).sum()), 1)
        self.assertTrue((pd.to_datetime(blocks.validation_end) < start).all())
        changed = raw.copy()
        changed.loc[end:] += 123456.
        second = correct_points(changed, parent, [event])
        pd.testing.assert_frame_equal(output, second[0])
        pd.testing.assert_frame_equal(methods, second[1])

    def test_previous_events_cannot_enter_reference_or_validation_truth(self):
        index = pd.date_range('2026-01-01', periods=40 * 288, freq='5min', name='time')
        raw = pd.DataFrame({'p': 10.}, index=index)
        events = []
        for day in (35, 36):
            start, end = index[day * 288], index[day * 288 + 24]
            events.append({'event_id': f'dr_{day}', 'start': str(start),
                           'rebound_start': str(start + pd.Timedelta(hours=1)), 'end_exclusive': str(end)})
            raw.loc[start:end - pd.Timedelta(minutes=5), 'p'] = 999.
        adjusted, methods, _, blocks = correct_points(raw, raw, events)
        np.testing.assert_array_equal(adjusted.p, 10.)
        for event in events:
            overlaps = ((pd.to_datetime(blocks.validation_start) < pd.Timestamp(event['end_exclusive']))
                        & (pd.to_datetime(blocks.validation_end) >= pd.Timestamp(event['start'])))
            self.assertFalse(overlaps.any())
        self.assertEqual(methods.iloc[1]['method'], 'locf')
        self.assertTrue(pd.isna(methods.iloc[1]['past_day_repeat_median_mae']))

    def test_invalid_or_insufficient_history_rejected(self):
        index = pd.date_range('2026-01-01', periods=600, freq='5min', name='time')
        raw = pd.DataFrame({'p': 10.}, index=index)
        event = {'event_id': 'e', 'start': str(index[100]), 'rebound_start': str(index[112]), 'end_exclusive': str(index[124])}
        with self.assertRaisesRegex(ValueError, '校准不足'):
            correct_points(raw, raw, [event])
        for events in ([], [event, event], [{**event, 'end_exclusive': str(index[90])}], [{**event, 'extra': True}]):
            with self.subTest(events=events), self.assertRaises(ValueError):
                correct_points(raw, raw, events)
        broken = raw.copy()
        broken.iloc[1] = np.inf
        with self.assertRaisesRegex(ValueError, '无限值'):
            correct_points(broken, raw, [event])

    def test_cli_publishes_two_named_outputs_without_changing_sources(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = fixture(root)
            path = root / 'recipe.json'
            path.write_text(json.dumps(recipe), encoding='utf-8')
            env = dict(os.environ)
            env.pop('PYTHONPATH', None)
            result = subprocess.run([sys.executable, str(SCRIPT), '--root', str(root), '--recipe', str(path)],
                                    cwd=root, env=env, text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            event = recipe['events'][0]
            self.assertFalse((root / 'analysis/data_v3/demand_response/point_tables').exists())
            for devices, expected in (('hvac_all_devices', 36.), ('hvac_remove_devices', 30.)):
                target = pd.read_csv(root / f'forecast_data/data_v3/{devices}/A1_all/data_dr_adjusted.csv')
                np.testing.assert_array_equal(target.hvac_total_load_AB, expected)
                np.testing.assert_array_equal(target.it_subset_load, 5.)
                for route in ('route_A', 'route_B'):
                    points = pd.read_csv(root / f'outlier_remove_data/data_v3/demand_response/{devices}/{route}/A1_data.csv')
                    np.testing.assert_array_equal(points.total_load, target['hvac_total_load_' + route[-1]])
                mask = pd.read_csv(root / f'analysis/data_v3/demand_response/{devices}/mask.csv')
                active = mask.dr_counterfactual
                self.assertEqual(int(active.sum()), 24)
                self.assertFalse(mask.loc[active, 'hvac_total_load_AB_observed'].any())
                self.assertTrue(mask.loc[~active, 'hvac_total_load_AB_observed'].all())
                self.assertTrue(mask.it_subset_load_observed.all())
                known = pd.to_datetime(mask.loc[active, 'hvac_total_load_AB_eligibility_known_at'])
                self.assertTrue((known == pd.Timestamp(event['end_exclusive']) + pd.Timedelta(minutes=25)).all())
                self.assertEqual({p.name for p in (root / f'forecast_data/data_v3/{devices}/A1_all').glob('*.csv')},
                                 {'data.csv', 'data_dr_adjusted.csv'})
            report = json.loads((root / 'analysis/data_v3/demand_response/manifest.json').read_text())
            self.assertFalse(report['strict_online_eligible'])
            for name, digest in {**recipe['input_sha256'], **report['artifact_sha256']}.items():
                self.assertEqual(sha256_file(root / name), digest)
            png = root / 'analysis/data_v3/demand_response/plots/dr_test.png'
            self.assertEqual(png.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            with self.assertRaises(FileExistsError):
                prepare(root, recipe)

    def test_source_drift_or_boundary_mismatch_does_not_publish(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = fixture(root)
            bad = json.loads(json.dumps(recipe))
            first = next(iter(bad['input_sha256']))
            bad['input_sha256'][first] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'SHA'):
                prepare(root, bad)
            bad = json.loads(json.dumps(recipe))
            bad['events'][0]['end_exclusive'] = str(pd.Timestamp(bad['events'][0]['end_exclusive']) + pd.Timedelta(minutes=5))
            with self.assertRaisesRegex(ValueError, '结束边界'):
                prepare(root, bad)
            self.assertFalse((root / 'analysis/data_v3/demand_response').exists())
            self.assertFalse(list(root.glob('forecast_data/**/data_dr_adjusted.csv')))


if __name__ == '__main__':
    unittest.main()
