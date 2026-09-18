"""AIDC HVAC 离线准备合同；新测试默认 integration。"""
from pathlib import Path
import importlib
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / 'config/aidc_hvac_load_5min/scripts'
sys.path.insert(0, str(SCRIPTS))
import impute_hvac_data as imputer


class MigrationTest(unittest.TestCase):
    def test_collision_fails_before_any_move(self):
        migration = importlib.import_module('migrate_hvac_data')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in migration.FAMILIES:
                folder = root / family
                folder.mkdir(parents=True)
                for name in migration.FILES:
                    (folder / name).write_text('raw', encoding='utf-8')
            (root / 'IT_load/analysis').mkdir()
            (root / 'IT_load/analysis/missing_summary.csv').write_text('source', encoding='utf-8')
            (root / 'analysis').mkdir()
            (root / 'analysis/IT_load_missing_summary.csv').write_text('destination', encoding='utf-8')
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with self.assertRaises(FileExistsError):
                migration.migrate(root)
            self.assertEqual(before, {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_archive_guard_refuses_any_existing_target(self):
        migration = importlib.import_module('migrate_hvac_data')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'raw_data/A1_data.csv'
            migration.require_new_files([target])
            target.parent.mkdir()
            target.write_bytes(b'original')
            with self.assertRaises(FileExistsError):
                migration.require_new_files([root / 'other.csv', target])
            self.assertEqual(target.read_bytes(), b'original')

    def test_migration_preserves_bytes_and_resolves_analysis_collisions(self):
        migration = importlib.import_module('migrate_hvac_data')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            families = ['hvac_all_devices/route_A', 'hvac_all_devices/route_B',
                        'hvac_remove_devices/route_A', 'hvac_remove_devices/route_B', 'IT_load']
            for family in families:
                folder = root / family
                folder.mkdir(parents=True)
                for name in ['A1_data.csv', 'A2_data.csv', 'A3_data.csv', 'data.csv']:
                    (folder / name).write_bytes(b'time,total_load\n2026-01-01,1\n')
            (root / 'analysis').mkdir()
            (root / 'analysis/missing_summary.csv').write_bytes(b'hvac')
            (root / 'IT_load/analysis').mkdir()
            (root / 'IT_load/analysis/missing_summary.csv').write_bytes(b'it')
            result = migration.migrate(root)
            self.assertEqual(len(result['raw_files']), 20)
            self.assertEqual((root / 'analysis/missing_summary.csv').read_bytes(), b'hvac')
            self.assertEqual((root / 'analysis/IT_load_missing_summary.csv').read_bytes(), b'it')
            self.assertFalse((root / 'IT_load').exists())
            self.assertEqual(migration.migrate(root), result)
            for row in result['raw_files']:
                self.assertEqual(migration.sha256_file(root / row['destination']), row['sha256'])


class DatasetPreparationTest(unittest.TestCase):
    def test_entrypoints_work_from_another_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            for script in ['migrate_hvac_data.py', 'impute_hvac_data.py', 'select_hvac_windows.py']:
                result = subprocess.run([sys.executable, str(SCRIPTS / script), '--help'],
                                        cwd=tmp, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('--root', result.stdout)

    def test_authorized_exclusions_are_exact_and_unique(self):
        recipe = json.loads(imputer.RECIPE.read_text(encoding='utf-8'))
        excluded = recipe['excluded_it_points']
        self.assertEqual(set(excluded), {'A1', 'A3'})
        self.assertEqual(len(excluded['A1']), 39)
        self.assertEqual(len(excluded['A3']), 43)
        for points in excluded.values():
            self.assertEqual(len(points), len(set(points)))

    def test_window_tie_prefers_latest_and_ignores_partial_days(self):
        selector = importlib.import_module('select_hvac_windows')
        index = pd.date_range('2026-07-14 00:05', '2026-07-20 23:50', freq='5min')
        valid = pd.Series(True, index=index)
        valid.loc['2026-07-17'] = False
        start, stop, days = selector.select_window(valid, '2026-07-14')
        self.assertEqual((str(start), str(stop), days), ('2026-07-18 00:00:00', '2026-07-20 00:00:00', 2))
        with self.assertRaises(ValueError):
            selector.select_window(valid & False, '2026-07-14')

    def test_complete_dataset_exports_20_strict_tables_without_touching_raw(self):
        migration = importlib.import_module('migrate_hvac_data')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = pd.date_range('2026-07-01', periods=288 * 50, freq='5min', name='time')
            before = {}
            for family in migration.FAMILIES:
                folder = root / 'raw_data' / family
                folder.mkdir(parents=True)
                buildings = {}
                for number, building in enumerate(migration.BUILDINGS, 1):
                    frame = pd.DataFrame({'p': float(number)}, index=index)
                    frame.iloc[288 * 20:288 * 20 + 2, 0] = np.nan
                    frame.iloc[288 * 26:288 * 26 + 73, 0] = np.nan
                    frame.iloc[288 * 43 - 1:288 * 43 + 1, 0] = np.nan
                    buildings[building] = frame.copy().add_prefix(building + '_')
                    frame['total_load'] = frame.sum(axis=1, min_count=1)
                    frame.to_csv(folder / (building + '_data.csv'))
                combined = pd.concat(buildings.values(), axis=1)
                combined['total_load'] = combined.sum(axis=1, min_count=1)
                combined.to_csv(folder / 'data.csv')
                for path in folder.glob('*.csv'):
                    before[path] = migration.sha256_file(path)
            inventory = imputer.impute_dataset(root, excluded_it_points={})
            self.assertEqual(len(inventory['files']), 20)
            self.assertEqual(len(list((root / 'imputed_data').rglob('*.csv'))), 20)
            self.assertEqual(before, {p: migration.sha256_file(p) for p in before})
            for row in inventory['files']:
                df = imputer.read_table(root / row['output'])
                points = df.drop(columns='total_load')
                pd.testing.assert_series_equal(df.total_load, points.sum(axis=1, min_count=len(points.columns)), check_names=False)
            with self.assertRaises(FileExistsError):
                imputer.impute_dataset(root, excluded_it_points={})
            selector = importlib.import_module('select_hvac_windows')
            windows = selector.export_windows(root, recent_start='2026-07-14')
            self.assertEqual(len(windows), 32)
            self.assertEqual(len(list((root / 'forecast_data').rglob('*.csv'))), 32)
            for row in windows:
                frame = pd.read_csv(root / row['output'], parse_dates=['time'])
                self.assertFalse(frame.isna().any().any())
                self.assertEqual(len(frame), row['days'] * 288)
                self.assertEqual(row['folds_14_1'], max(0, row['days'] - 14))
                self.assertGreater(row['folds_14_1'], 0)
                self.assertLess(row['eligibility_safe_folds_14_1'], row['folds_14_1'])
                self.assertEqual(frame.time.iloc[0], pd.Timestamp(row['start']))
            combined = pd.read_csv(root / 'forecast_data/hvac_all_devices/route_A/data_with_it.csv')
            self.assertEqual(list(combined.columns), ['time', 'hvac_total_load', 'it_total_load',
                'A1_hvac_total_load', 'A2_hvac_total_load', 'A3_hvac_total_load',
                'A1_it_total_load', 'A2_it_total_load', 'A3_it_total_load'])
            np.testing.assert_array_equal(combined.hvac_total_load,
                combined[['A1_hvac_total_load', 'A2_hvac_total_load', 'A3_hvac_total_load']].sum(axis=1))
            self.assertEqual(before, {p: migration.sha256_file(p) for p in before})
            with self.assertRaises(FileExistsError):
                selector.export_windows(root, recent_start='2026-07-14')


class CausalImputationTest(unittest.TestCase):
    def test_periodic_signal_selects_previous_day_using_historical_truth_only(self):
        index = pd.date_range('2026-01-01', periods=288 * 20, freq='5min', name='time')
        truth = pd.Series(10 + np.tile(np.arange(288), 20), index=index, dtype=float)
        source = truth.copy()
        start = 288 * 15 + 100
        source.iloc[start:start + 12] = np.nan
        result, audit = imputer.impute_series(source)
        self.assertEqual(audit[0]['method'], 'previous_day')
        self.assertEqual(audit[0]['previous_day_median_mae'], 0.0)
        pd.testing.assert_series_equal(result, truth)
        changed = source.copy()
        changed.iloc[start + 12:] = np.nan
        # 保留恢复点，之后的未来缺失模式及数值变化均不能改变当前修复。
        changed.iloc[start + 12] = 90000.0
        again, rows = imputer.impute_series(changed)
        pd.testing.assert_series_equal(result.iloc[:start + 12], again.iloc[:start + 12])
        self.assertEqual(audit[0], rows[0])

    def test_edges_insufficient_history_and_bad_inputs_are_not_silently_filled(self):
        index = pd.date_range('2026-01-01', periods=300, freq='5min')
        source = pd.Series(5.0, index=index)
        source.iloc[:2] = np.nan
        source.iloc[5:7] = np.nan
        source.iloc[-2:] = np.nan
        result, audit = imputer.impute_series(source)
        pd.testing.assert_series_equal(result, source)
        self.assertEqual([r['status'] for r in audit], ['unfilled_edge', 'unfilled_insufficient_validation', 'unfilled_edge'])
        for bad in [source.iloc[::-1], source.iloc[[0, 1, 3]], source.replace(5.0, np.inf), source.replace(5.0, -1.0)]:
            with self.assertRaises(ValueError):
                imputer.impute_series(bad)

    def test_table_excludes_only_authorized_points_and_requires_complete_total(self):
        index = pd.date_range('2026-01-01', periods=288 * 12, freq='5min', name='time')
        source = pd.DataFrame({'p1': 2.0, 'p2': 3.0, 'absent': np.nan}, index=index)
        source.iloc[288 * 7:288 * 7 + 72, 0] = np.nan
        source.iloc[288 * 8:288 * 8 + 73, 0] = np.nan
        source['total_load'] = source.sum(axis=1, min_count=1)
        before = source.copy(deep=True)
        result, audit, mask = imputer.impute_table(source, excluded=['absent'])
        pd.testing.assert_frame_equal(source, before)
        self.assertEqual(list(result.columns), ['p1', 'p2', 'total_load'])
        self.assertTrue((result.iloc[288 * 7:288 * 7 + 72].total_load == 5.0).all())
        self.assertTrue(result.iloc[288 * 8:288 * 8 + 73].total_load.isna().all())
        self.assertFalse(mask.iloc[288 * 7].total_observed)
        self.assertTrue(mask.iloc[288 * 7].total_imputed)
        self.assertGreater(pd.Timestamp(mask.iloc[288 * 7].eligibility_known_at), index[288 * 7])
        self.assertTrue(any(row['point'] == 'p1' and row['status'] == 'filled' for row in audit))
        with self.assertRaises(ValueError):
            imputer.impute_table(source, excluded=[])
        with self.assertRaises(ValueError):
            imputer.impute_table(source, excluded=['p2', 'absent'])

    def test_short_gap_selected_from_past_masks_preserves_observations(self):
        index = pd.date_range('2026-01-01', periods=288 * 12, freq='5min', name='time')
        source = pd.Series(17.25, index=index, name='device')
        start = 288 * 8 + 10
        source.iloc[start:start + 72] = np.nan
        source.iloc[start + 288:start + 288 + 73] = np.nan
        filled, audit = imputer.impute_series(source)
        np.testing.assert_array_equal(filled[source.notna()], source.dropna())
        self.assertTrue((filled.iloc[start:start + 72] == 17.25).all())
        self.assertTrue(filled.iloc[start + 288:start + 288 + 73].isna().all())
        selected = next(row for row in audit if row['status'] == 'filled')
        self.assertEqual(selected['method'], 'locf')
        self.assertGreaterEqual(selected['validation_count'], 3)
        self.assertLess(pd.Timestamp(selected['validation_end']), index[start])
        changed = source.copy()
        changed.iloc[start + 72:] = 9000.0
        again, second = imputer.impute_series(changed)
        np.testing.assert_array_equal(filled.iloc[:start + 72], again.iloc[:start + 72])
        self.assertEqual(selected, next(row for row in second if row['status'] == 'filled'))


if __name__ == '__main__':
    unittest.main()
