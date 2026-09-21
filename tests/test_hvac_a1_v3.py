"""A1 v3 固定子集与过去观测填补；合成数据不作为业务效果证据。"""
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from config.aidc_hvac_load_5min.scripts.v3.imputed_data.gap_filling import fill_series, predictions
from config.aidc_hvac_load_5min.scripts.v3.imputed_data.prepare_data import prepare, select_it
from config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data import sha256_file


class A1GapFillingTest(unittest.TestCase):
    def test_day_gap_scores_observed_truth_without_imputing_validation_labels(self):
        index = pd.date_range('2026-01-01', periods=70 * 288, freq='5min')
        source = pd.Series(10., index=index, name='sparse')
        source.iloc[144::288] = np.nan
        start = 50 * 288 + 20
        source.iloc[start:start + 285] = np.nan
        filled, audit = fill_series(source, output_start=index[30 * 288])
        row = next(r for r in audit if r['gap_points'] == 285)
        self.assertGreaterEqual(row['validation_coverage_min'], .8)
        self.assertLess(row['validation_coverage_min'], 1.)
        np.testing.assert_array_equal(filled.iloc[start:start + 285], 10.)

    def test_edge_insufficient_and_invalid_input_fail_closed(self):
        index = pd.date_range('2026-01-01', periods=600, freq='5min')
        for position in (0, 2, 599):
            source = pd.Series(1., index=index)
            source.iloc[position] = np.nan
            with self.subTest(position=position), self.assertRaises(ValueError):
                fill_series(source, output_start=index[0])
        source = pd.Series(1., index=index)
        source.iloc[20] = np.inf
        with self.assertRaises(ValueError):
            fill_series(source, output_start=index[0])

    def test_subset_is_fixed_not_dynamic_or_zero_filled(self):
        index = pd.date_range('2026-01-01', periods=600, freq='5min')
        raw = pd.DataFrame({'kept': 2., 'absent': np.nan, 'late': np.nan}, index=index)
        raw.loc[index[400]:, 'late'] = 10.
        recipe = {'start': str(index[300]), 'excluded_absent_it_points': ['absent'],
                  'excluded_late_it_points': ['late'], 'retained_it_points': ['kept']}
        pd.testing.assert_frame_equal(select_it(raw, recipe), raw[['kept']])
        raw.loc[index[200], 'late'] = 5.
        with self.assertRaises(ValueError):
            select_it(raw, recipe)

    def test_two_scenarios_strict_sum_and_no_overwrite(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            index = pd.date_range('2026-01-01', periods=40 * 288, freq='5min', name='time')
            paths = {}
            for devices in ('hvac_all_devices', 'hvac_remove_devices'):
                for route, value in (('route_A', 10.), ('route_B', 20.)):
                    frame = pd.DataFrame({'shared': value}, index=index)
                    if devices == 'hvac_all_devices':
                        frame['pump'] = 3.
                    frame.iloc[35 * 288, 0] = np.nan
                    frame['total_load'] = frame.sum(axis=1, min_count=1)
                    paths[f'raw_data/{devices}/{route}/A1_data.csv'] = frame
            it = pd.DataFrame({'kept': 5., 'absent': np.nan, 'late': np.nan}, index=index)
            it.loc[index[-10]:, 'late'] = 2.
            it['total_load'] = it.sum(axis=1, min_count=1)
            paths['raw_data/IT_load/A1_data.csv'] = it
            for name, frame in paths.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path)
            recipe = {'start': str(index[30 * 288]), 'end': str(index[-1]), 'context_days': 30,
                      'excluded_absent_it_points': ['absent'], 'excluded_late_it_points': ['late'],
                      'retained_it_points': ['kept'], 'data_version': 'data_v3',
                      'input_sha256': {p: sha256_file(root / p) for p in paths}}
            report = prepare(root, recipe)
            self.assertEqual(len(report['targets']), 2)
            for devices, expected in (('hvac_all_devices', 36.), ('hvac_remove_devices', 30.)):
                path = root / f'forecast_data/data_v3/{devices}/A1_all/data.csv'
                actual = pd.read_csv(path)
                self.assertEqual(len(actual), 10 * 288)
                np.testing.assert_array_equal(actual.hvac_total_load_AB, expected)
                np.testing.assert_array_equal(actual.it_subset_load, 5.)
                np.testing.assert_array_equal(actual.hvac_total_load_AB,
                                              actual.hvac_total_load_A + actual.hvac_total_load_B)
            self.assertFalse((root / 'outlier_remove_data').exists())
            with self.assertRaises(FileExistsError):
                prepare(root, recipe)

    def test_long_gap_uses_only_past_day_and_preserves_observations(self):
        index = pd.date_range('2026-01-01', periods=70 * 288, freq='5min')
        source = pd.Series(100. + np.arange(len(index)) % 288, index=index, name='point')
        start, stop = 50 * 288, 53 * 288
        source.iloc[start:stop] = np.nan
        original = source.copy()
        filled, audit = fill_series(source, output_start=index[30 * 288])
        np.testing.assert_array_equal(filled[source.notna()], source.dropna())
        np.testing.assert_array_equal(filled.iloc[start:stop], 100. + np.arange(stop - start) % 288)
        self.assertEqual(audit[0]['method'], 'past_day_repeat')
        self.assertLess(pd.Timestamp(audit[0]['raw_dependency_end']), index[start])
        changed = source.copy()
        changed.iloc[stop:] += 999999.
        altered, second = fill_series(changed, output_start=index[30 * 288])
        np.testing.assert_array_equal(altered.iloc[start:stop], filled.iloc[start:stop])
        self.assertEqual(audit[0], second[0])
        pd.testing.assert_series_equal(source, original)
        pred = predictions(source.to_numpy(), np.array([start]), stop - start)
        self.assertTrue(np.isfinite(pred).all())


if __name__ == '__main__':
    unittest.main()
