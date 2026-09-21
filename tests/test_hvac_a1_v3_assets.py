"""独立核验 A1 v3 真实点位、逐缺口数值、来源掩码与天气绑定。"""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset/aidc_hvac_load_5min'
GRID = pd.date_range('2026-04-08', '2026-09-16 23:55', freq='5min', name='time')


def read(path):
    return pd.read_csv(path, parse_dates=['time'], index_col='time', float_precision='round_trip')


def verify_assets():
    recipe = json.loads((ROOT / 'config/aidc_hvac_load_5min/scripts/v3/preparation.json').read_text())
    manifest = json.loads((DATA / 'analysis/data_v3/manifest.json').read_text())
    assert manifest['recipe'] == recipe
    assert manifest['outlier_processing'] is False
    assert len(recipe['retained_it_points']) == 204
    assert len(recipe['excluded_absent_it_points']) == 39
    assert len(recipe['excluded_late_it_points']) == 43
    tables, masks, hashes, counts = {}, {}, {}, {}
    for name, expected in recipe['input_sha256'].items():
        assert sha256_file(DATA / name) == expected, name
        key = name.removeprefix('raw_data/')
        raw = read(DATA / name)
        path = DATA / 'imputed_data/data_v3' / key
        filled = read(path)
        hashes[str(path.relative_to(DATA))] = sha256_file(path)
        pd.testing.assert_index_equal(filled.index, GRID)
        points = [c for c in filled if c != 'total_load']
        assert points == (recipe['retained_it_points'] if key.startswith('IT_load')
                          else [c for c in raw if c != 'total_load'])
        truth = raw.loc[GRID, points]
        observed = truth.notna().to_numpy()
        assert np.isfinite(filled.to_numpy()).all(), key
        np.testing.assert_array_equal(filled[points].to_numpy()[observed], truth.to_numpy()[observed])
        np.testing.assert_allclose(filled.total_load, filled[points].sum(axis=1), rtol=1e-14, atol=1e-10)
        mask_path = DATA / 'analysis/data_v3/imputation' / key.replace('.csv', '.mask.csv')
        mask = read(mask_path)
        pd.testing.assert_index_equal(mask.index, GRID)
        np.testing.assert_array_equal(mask.total_observed, observed.all(axis=1))
        np.testing.assert_array_equal(mask.total_imputed, ~observed.all(axis=1))
        masks[key], tables[key] = mask, filled
        counts[key] = {'retained_points': len(points), 'filled_cells': int((~observed).sum())}
        if not key.startswith('hvac_remove_devices'):
            audit_path = DATA / 'analysis/data_v3/imputation' / key.replace('.csv', '.gaps.csv')
            audit = pd.read_csv(audit_path, float_precision='round_trip')
            hashes[str(audit_path.relative_to(DATA))] = sha256_file(audit_path)
            assert (audit.validation_count >= 3).all()
            threshold = np.where(audit.validation_bucket_points <= 72, 1., .8)
            assert (audit.validation_coverage_min >= threshold).all()
            gap_start, gap_end = pd.to_datetime(audit.gap_start), pd.to_datetime(audit.gap_end)
            assert (pd.to_datetime(audit.validation_end) < gap_start).all()
            assert (pd.to_datetime(audit.raw_dependency_end) < gap_start).all()
            assert (pd.to_datetime(audit.eligibility_known_at) == gap_end + pd.Timedelta(minutes=5)).all()
            scores = audit[['locf_median_mae', 'past_mean_1h_median_mae', 'past_day_repeat_median_mae']].to_numpy()
            methods = ('locf', 'past_mean_1h', 'past_day_repeat')
            winner = np.nanargmin(scores, axis=1)
            np.testing.assert_array_equal(audit.method, np.asarray(methods)[winner])
            assert int(audit.gap_points.sum()) == int((~observed).sum())
            for point, rows in audit.groupby('point', sort=False):
                values = raw[point].to_numpy()
                actual = filled[point].to_numpy()
                covered = np.zeros(len(GRID), dtype=bool)
                origin_positions = raw.index.get_indexer(pd.to_datetime(rows.gap_start))
                output_positions = GRID.get_indexer(pd.to_datetime(rows.gap_start))
                for origin, output, length, method in zip(origin_positions, output_positions, rows.gap_points, rows.method):
                    assert origin > 0 and output >= 0
                    assert not covered[output:output + length].any()
                    assert np.isnan(values[origin:origin + length]).all()
                    if method == 'locf':
                        predicted = np.repeat(values[origin - 1], length)
                    elif method == 'past_mean_1h':
                        predicted = np.repeat(values[origin - np.arange(1, 13)].mean(), length)
                    else:
                        predicted = values[origin - 288 + np.arange(length) % 288]
                    np.testing.assert_array_equal(actual[output:output + length], predicted)
                    covered[output:output + length] = True
                np.testing.assert_array_equal(covered, truth[point].isna())
    for route in ('route_A', 'route_B'):
        whole = tables[f'hvac_all_devices/{route}/A1_data.csv']
        subset = tables[f'hvac_remove_devices/{route}/A1_data.csv']
        points = subset.columns.drop('total_load')
        pd.testing.assert_frame_equal(whole[points], subset[points])
    expected_targets = {f'forecast_data/data_v3/{d}/A1_all/data.csv'
                        for d in ('hvac_all_devices', 'hvac_remove_devices')}
    assert {str(p.relative_to(DATA)) for p in (DATA / 'forecast_data/data_v3').rglob('*.csv')} == expected_targets
    assert {r['path'] for r in manifest['targets']} == expected_targets
    weather_manifest = pd.read_csv(DATA / 'weather_data/data_v3/manifest.csv')
    assert set(weather_manifest.target_file) == {'dataset/aidc_hvac_load_5min/' + p for p in expected_targets}
    for item in manifest['targets']:
        relative = item['path']
        frame = read(DATA / relative)
        assert sha256_file(DATA / relative) == item['sha256']
        pd.testing.assert_index_equal(frame.index, GRID)
        devices = Path(relative).parts[2]
        for route in ('A', 'B'):
            np.testing.assert_array_equal(frame['hvac_total_load_' + route],
                                          tables[f'{devices}/route_{route}/A1_data.csv'].total_load)
        np.testing.assert_array_equal(frame.hvac_total_load_AB, frame.hvac_total_load_A + frame.hvac_total_load_B)
        np.testing.assert_array_equal(frame.it_subset_load, tables['IT_load/A1_data.csv'].total_load)
        mask = read(DATA / f'analysis/data_v3/forecast_windows/{devices}/A1_all/mask.csv')
        pd.testing.assert_index_equal(mask.index, GRID)
        a, b = (masks[f'{devices}/route_{r}/A1_data.csv'] for r in ('A', 'B'))
        np.testing.assert_array_equal(mask.hvac_total_load_AB_observed, a.total_observed & b.total_observed)
        np.testing.assert_array_equal(mask.it_subset_load_observed, masks['IT_load/A1_data.csv'].total_observed)
        hashes[relative] = item['sha256']
    for row in weather_manifest.to_dict('records'):
        path = ROOT / row['file']
        meta = json.loads(path.with_suffix('.meta.json').read_text())
        assert meta['target_sha256'] == sha256_file(ROOT / row['target_file'])
        assert row['sha256'] == sha256_file(path)
        frame = pd.read_csv(path)
        assert pd.DatetimeIndex(pd.to_datetime(frame.ts)).equals(GRID.rename('ts'))
        names = ['rt_tt2', 'cal_rh', 'rt_ssr', 'rt_ws10', 'rt_ps', 'rt_rain',
                 'pred_tt2', 'pred_rh', 'pred_ssrd', 'pred_ws10', 'pred_ps', 'pred_rain']
        assert np.isfinite(frame[names].to_numpy()).all()
        hashes[str(path.relative_to(DATA))] = row['sha256']
    return {'status': 'passed', 'rows_per_target': len(GRID), 'tables': counts,
            'target_count': len(expected_targets), 'weather_count': len(weather_manifest),
            'artifact_sha256': hashes, 'models_fitted': False}


class A1V3AssetsTest(unittest.TestCase):
    def test_raw_preservation_gap_reconstruction_masks_and_weather(self):
        report = verify_assets()
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['target_count'], 2)
        self.assertEqual(report['weather_count'], 2)


if __name__ == '__main__':
    unittest.main()
