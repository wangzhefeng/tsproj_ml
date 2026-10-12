"""独立回读真实需求响应派生，重算选型评分和每个点位估计，不调用生成器。"""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset/aidc_hvac_load_5min'
AUDIT = DATA / 'analysis/data_v3/demand_response'
POINTS = DATA / 'outlier_remove_data/data_v3/demand_response'


def read(path):
    return pd.read_csv(path, index_col='time', parse_dates=True, float_precision='round_trip')


def reference(values, origin, length, method):
    if method == 'locf':
        return np.full(length, values[origin - 1])
    if method == 'past_mean_1h':
        return np.full(length, np.mean(values[origin - np.arange(1, 13)]))
    if method == 'past_day_repeat':
        return values[origin - 288 + np.arange(length)]
    raise AssertionError(method)


class DemandResponseAssetsTest(unittest.TestCase):
    def test_actual_inputs_outputs_masks_and_all_point_estimates(self):
        manifest = json.loads((AUDIT / 'manifest.json').read_text())
        recipe_path = ROOT / 'config/aidc_hvac_load_5min/scripts/v3/outlier_remove_data/demand_response.json'
        recipe = json.loads(recipe_path.read_text())
        self.assertEqual(manifest['recipe'], recipe)
        self.assertFalse(manifest['strict_online_eligible'])
        for name, digest in {**recipe['input_sha256'], **manifest['artifact_sha256']}.items():
            self.assertEqual(sha256_file(DATA / name), digest, name)
        if 'path_migration' in manifest:
            migration = json.loads((DATA / manifest['path_migration']).read_text())
            original_path = DATA / migration['original_manifest']
            self.assertEqual(sha256_file(original_path), migration['original_manifest_sha256'])
            original = json.loads(original_path.read_text())
            self.assertEqual(manifest['builder_sha256'], original['builder_sha256'])
            expected = {migration['path_mapping'].get(p, p): digest
                        for p, digest in original['artifact_sha256'].items()}
            self.assertEqual(manifest['artifact_sha256'], expected)
            self.assertFalse((AUDIT / 'point_tables').exists())
        else:
            self.assertEqual(manifest['builder_sha256'], sha256_file(recipe_path.with_name('adjust_demand_response.py')))
        methods = pd.read_csv(AUDIT / 'point_methods.csv', float_precision='round_trip')
        blocks = pd.read_csv(AUDIT / 'validation_blocks.csv', float_precision='round_trip')
        changes = pd.read_csv(AUDIT / 'point_changes.csv', float_precision='round_trip')
        self.assertFalse(changes.duplicated(['route', 'point', 'time']).any())
        event_map = {e['event_id']: e for e in recipe['events']}
        expected_estimates = 0
        for route in ('route_A', 'route_B'):
            key = f'hvac_all_devices/{route}/A1_data.csv'
            raw = read(DATA / 'raw_data' / key).drop(columns='total_load')
            parent = read(DATA / 'imputed_data/data_v3' / key).drop(columns='total_load')
            output = read(POINTS / key).drop(columns='total_load')
            masked = raw.copy()
            scope = np.zeros(len(parent), dtype=bool)
            for event in recipe['events']:
                start, end = pd.Timestamp(event['start']), pd.Timestamp(event['end_exclusive'])
                masked.loc[(raw.index >= start) & (raw.index < end)] = np.nan
                scope |= (parent.index >= start) & (parent.index < end)
            pd.testing.assert_frame_equal(output.loc[~scope], parent.loc[~scope], check_exact=True)
            expected_estimates += int(scope.sum()) * len(raw.columns)
            rows = methods[methods.route == route]
            self.assertEqual(set(zip(rows.event_id, rows.point)), {(e, p) for e in event_map for p in raw})
            self.assertEqual(len(rows), len(event_map) * len(raw.columns))
            for row in rows.to_dict('records'):
                event = event_map[row['event_id']]
                start, end = pd.Timestamp(event['start']), pd.Timestamp(event['end_exclusive'])
                index = output.loc[(output.index >= start) & (output.index < end)].index
                point = row['point']
                values = masked[point].to_numpy()
                origin = raw.index.get_loc(start)
                expected = reference(values, origin, len(index), row['method'])
                np.testing.assert_array_equal(output.loc[index, point], expected)
                evidence = changes[(changes.route == route) & (changes.point == point) & (changes.event_id == row['event_id'])]
                self.assertEqual(list(pd.to_datetime(evidence.time)), list(index))
                np.testing.assert_array_equal(evidence.parent_value, parent.loc[index, point])
                np.testing.assert_array_equal(evidence.adjusted_value, expected)
                np.testing.assert_array_equal(evidence.raw_value, raw.loc[index, point])
                np.testing.assert_array_equal(evidence.raw_observed, raw.loc[index, point].notna())
                trials = blocks[(blocks.route == route) & (blocks.point == point) & (blocks.event_id == row['event_id'])]
                scores = []
                for method in ('locf', 'past_mean_1h', 'past_day_repeat'):
                    subset = trials[trials.method == method]
                    if subset.empty:
                        self.assertTrue(pd.isna(row[method + '_median_mae']))
                        scores.append(np.inf)
                        continue
                    self.assertEqual(len(subset), row['validation_count'])
                    independent = []
                    for trial in subset.to_dict('records'):
                        left, right = pd.Timestamp(trial['validation_start']), pd.Timestamp(trial['validation_end'])
                        self.assertLess(right, start)
                        self.assertGreaterEqual(left, start - pd.Timedelta(days=30))
                        pos = raw.index.get_loc(left)
                        truth = values[pos:pos + len(index)]
                        self.assertTrue(np.isfinite(truth).all())
                        predicted = reference(values, pos, len(index), method)
                        error = float(np.abs(predicted - truth).mean())
                        self.assertAlmostEqual(error, trial['mae'], places=10)
                        independent.append(error)
                    score = float(np.median(independent))
                    self.assertAlmostEqual(score, row[method + '_median_mae'], places=10)
                    scores.append(score)
                self.assertEqual(row['method'], ('locf', 'past_mean_1h', 'past_day_repeat')[int(np.argmin(scores))])
        self.assertEqual(len(changes), expected_estimates)
        for devices in ('hvac_all_devices', 'hvac_remove_devices'):
            directory = DATA / f'forecast_data/data_v3/{devices}/A1_all'
            old, new = read(directory / 'data.csv'), read(directory / 'data_dr_adjusted.csv')
            scope = np.zeros(len(old), dtype=bool)
            for event in recipe['events']:
                scope |= (old.index >= event['start']) & (old.index < event['end_exclusive'])
            pd.testing.assert_index_equal(old.index, new.index)
            pd.testing.assert_frame_equal(new.loc[~scope], old.loc[~scope], check_exact=True)
            pd.testing.assert_series_equal(old.it_subset_load, new.it_subset_load, check_exact=True)
            np.testing.assert_array_equal(new.hvac_total_load_AB, new.hvac_total_load_A + new.hvac_total_load_B)
            self.assertTrue(np.isfinite(new.to_numpy()).all())
            mask = read(AUDIT / devices / 'mask.csv')
            original_mask = read(DATA / f'analysis/data_v3/forecast_windows/{devices}/A1_all/mask.csv')
            pd.testing.assert_frame_equal(mask.loc[~scope, original_mask.columns], original_mask.loc[~scope])
            np.testing.assert_array_equal(mask.dr_counterfactual, scope)
            self.assertFalse(mask.loc[scope, 'hvac_total_load_AB_observed'].any())
            for route, field in (('route_A', 'hvac_total_load_A'), ('route_B', 'hvac_total_load_B')):
                points = read(POINTS / f'{devices}/{route}/A1_data.csv').drop(columns='total_load')
                np.testing.assert_allclose(new[field], points.sum(axis=1), rtol=1e-14, atol=1e-10)
                whole = read(POINTS / f'hvac_all_devices/{route}/A1_data.csv')
                pd.testing.assert_frame_equal(points, whole[points.columns], check_exact=True)


if __name__ == '__main__':
    unittest.main()
