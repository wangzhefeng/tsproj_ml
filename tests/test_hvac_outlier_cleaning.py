"""暖通孤立异常清洗；只测离线数据，不训练模型。"""
from pathlib import Path
import importlib
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / 'config/aidc_hvac_load_5min/scripts'
sys.path.insert(0, str(SCRIPTS))


class IsolatedOutlierTest(unittest.TestCase):
    def test_dataset_build_preserves_sources_and_shared_device_versions(self):
        cleaner = importlib.import_module('config.aidc_hvac_load_5min.scripts.v1.outlier_remove_data.clean_hvac_outliers')
        imputer = importlib.import_module('config.aidc_hvac_load_5min.scripts.imputed_data.impute_hvac_data')
        selector = importlib.import_module('config.aidc_hvac_load_5min.scripts.forecast_data.select_hvac_windows')
        migration = importlib.import_module('config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = pd.date_range('2026-08-01', periods=5 * 288, freq='5min', name='time')
            for family in migration.FAMILIES:
                parts = []
                for building in migration.BUILDINGS:
                    frame = pd.DataFrame({'chiller': 100.0, 'pump': 20.0}, index=index)
                    if family.endswith('route_A') and building == 'A2':
                        frame.loc[index[1000], 'chiller'] = 200.0
                    if family.startswith('hvac_remove_devices'):
                        frame = frame.drop(columns='pump')
                    parts.append(frame.add_prefix(building + '_'))
                    frame['total_load'] = frame.sum(axis=1)
                    imputer.write_csv(frame, root / 'raw_data' / family / f'{building}_data.csv')
                combined = pd.concat(parts, axis=1)
                combined['total_load'] = combined.sum(axis=1)
                imputer.write_csv(combined, root / 'raw_data' / family / 'data.csv')
            imputer.impute_dataset(root, excluded_it_points={})
            selector.export_windows(root, recent_start=index[0])
            before = {p: migration.sha256_file(p) for p in root.rglob('*.csv')}
            destination = cleaner.build_cleaned_dataset(root)
            self.assertEqual({p: migration.sha256_file(p) for p in before}, before)
            for version in migration.VERSIONS:
                frame = imputer.read_table(destination / f'imputed_data/{version}/route_A/A2_data.csv')
                self.assertEqual(frame.loc[index[1000], 'chiller'], 100.0)
            self.assertEqual((root / 'imputed_data/IT_load/data.csv').read_bytes(),
                             (destination / 'imputed_data/IT_load/data.csv').read_bytes())
            with self.assertRaises(FileExistsError):
                cleaner.build_cleaned_dataset(root)
            selector.export_windows(root, recent_start=index[0], replace=True,
                                    preparation_root='outlier_remove_data/isolated_v1')
            schema = importlib.import_module('config.aidc_hvac_load_5min.scripts.forecast_data.forecast_schema')
            self.assertEqual(schema.resolve_preparation_root(root), destination)
            visual = importlib.import_module('config.aidc_hvac_load_5min.scripts.analysis.analyze_forecast_data')
            observed = visual.observed_mask(root, Path('hvac_all_devices/route_A/A2_data.csv'),
                                            'hvac_total_load_A', index, {})
            self.assertFalse(observed.loc[index[1000]])
            self.assertTrue(observed.loc[index[999]])
            selector.export_windows(root, recent_start=index[0], replace=True)
            a = pd.read_csv(root / 'forecast_data/hvac_all_devices/route_A/A2_data.csv')
            b = pd.read_csv(root / 'forecast_data/hvac_all_devices/route_B/A2_data.csv')
            pd.testing.assert_frame_equal(a, b)
            self.assertEqual(a.loc[1000, 'hvac_total_load_A'], 120.0)
            with self.assertRaises(ValueError):
                schema.resolve_preparation_root(root, '../outside')
            mask_path = destination / 'analysis/imputation/masks/hvac_all_devices/route_A/A2_data.csv'
            mask_path.write_text(mask_path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, '掩码哈希'):
                selector.export_windows(root, recent_start=index[0], replace=True)

    def test_invalid_values_and_insufficient_replacement_history_raise(self):
        cleaner = importlib.import_module('config.aidc_hvac_load_5min.scripts.v1.outlier_remove_data.clean_hvac_outliers')
        index = pd.date_range('2026-08-01', periods=50, freq='5min', name='time')
        points = pd.DataFrame({'chiller': 100.0, 'pump': 20.0}, index=index)
        points.loc[index[10], 'chiller'] = 200.0
        event = cleaner.detect_isolated(points, index[0], index[-1])
        source = points.assign(total_load=points.sum(axis=1))
        with self.assertRaisesRegex(ValueError, '不足|足够'):
            cleaner.clean_table(source, event)
        for value in [-1.0, np.inf]:
            bad = points.copy()
            bad.loc[index[10], 'chiller'] = value
            with self.assertRaises(ValueError):
                cleaner.detect_isolated(bad, index[0], index[-1])
        with self.assertRaises(ValueError):
            cleaner.detect_isolated(points.iloc[::2], index[0], index[-1])

    def test_cleaning_reimputes_point_preserves_observations_and_availability(self):
        cleaner = importlib.import_module('config.aidc_hvac_load_5min.scripts.v1.outlier_remove_data.clean_hvac_outliers')
        index = pd.date_range('2026-08-01', periods=5 * 288, freq='5min', name='time')
        source = pd.DataFrame({'chiller': 100.0, 'pump': 20.0}, index=index)
        source.loc[index[1000], 'chiller'] = 200.0
        source.loc[index[1004], 'chiller'] = np.nan
        source['total_load'] = source.sum(axis=1, min_count=1)
        event = cleaner.detect_isolated(source.drop(columns='total_load'), index[0], index[-1])
        result, gaps, mask, changes = cleaner.clean_table(source, event)
        self.assertEqual(result.loc[index[1000], 'chiller'], 100.0)
        self.assertEqual(result.loc[index[1004], 'chiller'], 100.0)
        self.assertFalse(mask.loc[index[1000], 'total_observed'])
        self.assertEqual(mask.loc[index[1000], 'eligibility_known_at'], index[1003])
        self.assertEqual(changes.iloc[0].new_value, 100.0)
        self.assertEqual(changes.iloc[0].method, 'locf')
        pd.testing.assert_series_equal(result.pump, source.pump)
        keep = source.chiller.notna() & (source.index != index[1000])
        pd.testing.assert_series_equal(result.loc[keep, 'chiller'], source.loc[keep, 'chiller'])
        self.assertTrue(result.total_load.eq(result.chiller + result.pump).all())

    def test_single_point_attribution_and_operating_events_preserved(self):
        cleaner = importlib.import_module('config.aidc_hvac_load_5min.scripts.v1.outlier_remove_data.clean_hvac_outliers')
        index = pd.date_range('2026-08-01', periods=100, freq='5min', name='time')
        points = pd.DataFrame({'chiller': 100.0, 'pump': 20.0}, index=index)
        points.loc[index[10], 'chiller'] = 200.0
        points.loc[index[20], 'chiller'] = 40.0
        points.loc[index[30:32], 'chiller'] = 200.0
        points.loc[index[40:50], 'chiller'] = 200.0
        points.loc[index[60], ['chiller', 'pump']] = [200.0, 40.0]
        points.loc[index[70], 'chiller'] = 101.0
        points.loc[index[80], 'chiller'] = 200.0
        points.loc[index[81], 'chiller'] = np.nan
        before = points.copy()
        rows = cleaner.detect_isolated(points, index[0], index[-1])
        accepted = rows.loc[rows.status.eq('accepted')]
        self.assertEqual(list(pd.to_datetime(accepted.time)), [index[10], index[20]])
        self.assertEqual(accepted.point.tolist(), ['chiller', 'chiller'])
        self.assertEqual(list(pd.to_datetime(accepted.detection_known_at)), [index[13], index[23]])
        pd.testing.assert_frame_equal(points, before)


if __name__ == '__main__':
    unittest.main()
