"""HVAC 双版本路径与清洗前置合同。"""
from pathlib import Path
import sys
import tempfile
import unittest
import json
import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / 'config/aidc_hvac_load_5min/scripts'
sys.path.insert(0, str(SCRIPTS))


class VersionedPathsTest(unittest.TestCase):
    def test_v2_full_preparation_and_export_do_not_consume_v1(self):
        from config.aidc_hvac_load_5min.scripts.v2.imputed_data.prepare_data import prepare
        from config.aidc_hvac_load_5min.scripts.imputed_data.impute_hvac_data import write_csv
        from config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data import FAMILIES, BUILDINGS
        from config.aidc_hvac_load_5min.scripts.forecast_data.select_hvac_windows import export_windows
        from config.aidc_hvac_load_5min.scripts.analysis.analyze_forecast_data import observed_mask
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = pd.date_range('2026-08-01', periods=5 * 288, freq='5min', name='time')
            for family in FAMILIES:
                parts = []
                for building in BUILDINGS:
                    frame = pd.DataFrame({'device': 100.}, index=index)
                    if family != 'IT_load':
                        frame.iloc[1000, 0] = 300.
                    parts.append(frame.add_prefix(building + '_'))
                    frame['total_load'] = frame.sum(axis=1)
                    write_csv(frame, root / 'raw_data' / family / f'{building}_data.csv')
                combined = pd.concat(parts, axis=1)
                combined['total_load'] = combined.sum(axis=1)
                write_csv(combined, root / 'raw_data' / family / 'data.csv')
            marker = root / 'imputed_data/data_v1/marker.txt'
            marker.parent.mkdir(parents=True)
            marker.write_text('not a usable input')
            recipe = root / 'recipe.json'
            recipe.write_text(json.dumps({'baseline_points': 12, 'minimum_baseline_points': 8,
                'relative_threshold': .1, 'absolute_threshold_kw': 1., 'mad_multiplier': 6., 'events': [],
                'isolated_windows': {b: {'start': str(index[0]), 'end': str(index[-1])} for b in BUILDINGS}}))
            inventory = prepare(root, recipe, excluded_it_points={})
            self.assertEqual(len(inventory['files']), 20)
            self.assertEqual(marker.read_text(), 'not a usable input')
            rows = export_windows(root, recent_start=index[0], data_version='data_v2')
            self.assertEqual(len(rows), 32)
            self.assertTrue(all(r['output'].startswith('forecast_data/data_v2/') for r in rows))
            observed = observed_mask(root, Path('hvac_all_devices/route_A/A1_data.csv'),
                                     'hvac_total_load_A', index, {}, data_version='data_v2')
            self.assertFalse(observed.iloc[1000])
            mask = pd.read_csv(root / 'analysis/data_v2/imputation/masks/hvac_all_devices/route_A/A1_data.csv')
            self.assertEqual(pd.Timestamp(mask.iloc[1000].eligibility_known_at), index[1003])
            with self.assertRaises(FileExistsError):
                prepare(root, recipe, excluded_it_points={})
            # 同一raw仍可走原顺序，v1/v2独立发布、分别绑定掩码。
            from config.aidc_hvac_load_5min.scripts.imputed_data.impute_hvac_data import impute_dataset
            from config.aidc_hvac_load_5min.scripts.v1.outlier_remove_data.clean_hvac_outliers import build_cleaned_dataset
            from config.aidc_hvac_load_5min.scripts.v1.outlier_remove_data.clean_hvac_redboxes import build
            impute_dataset(root, excluded_it_points={}, data_version='data_v1')
            export_windows(root, recent_start=index[0], data_version='data_v1')
            parent = build_cleaned_dataset(root, data_version='data_v1')
            self.assertEqual(parent, root / 'outlier_remove_data/data_v1/isolated_v1')
            export_windows(root, recent_start=index[0], data_version='data_v1', replace=True,
                           preparation_root='outlier_remove_data/data_v1/isolated_v1')
            old_recipe = json.loads(recipe.read_text())
            old_recipe.update(parent='outlier_remove_data/data_v1/isolated_v1', version='redbox_past_v2')
            recipe.write_text(json.dumps(old_recipe))
            destination = build(root, recipe, data_version='data_v1')
            self.assertEqual(destination, root / 'outlier_remove_data/data_v1/redbox_past_v2')
            export_windows(root, recent_start=index[0], data_version='data_v1', replace=True,
                           preparation_root='outlier_remove_data/data_v1/redbox_past_v2')
            self.assertEqual(marker.read_text(), 'not a usable input')

    def test_version_paths_are_isolated_and_reject_unknown_versions(self):
        from config.aidc_hvac_load_5min.scripts.preparation_paths import artifact_path
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('analysis', 'forecast_data', 'imputed_data', 'outlier_remove_data', 'weather_data'):
                self.assertEqual(artifact_path(root, name, 'data_v1'), root / name / 'data_v1')
                self.assertEqual(artifact_path(root, name, 'data_v2'), root / name / 'data_v2')
            with self.assertRaises(ValueError):
                artifact_path(root, 'analysis', '../escape')
            with self.assertRaises(ValueError):
                artifact_path(root, 'raw_data', 'data_v2')

    def test_raw_mask_combines_rules_before_imputation(self):
        from config.aidc_hvac_load_5min.scripts.v2.outlier_remove_data.clean_hvac_data import mask_family
        from config.aidc_hvac_load_5min.scripts.imputed_data.impute_hvac_data import impute_table
        index = pd.date_range('2026-08-01', periods=6 * 288, freq='5min', name='time')
        raw = pd.DataFrame({'device': 100., 'pump': 20.}, index=index)
        raw.loc[index[900], 'device'] = 300.
        raw.loc[index[1000:1010], 'device'] = 500.
        raw.loc[index[1010:1012], 'device'] = np.nan
        recipe = {'baseline_points': 12, 'minimum_baseline_points': 8,
                  'relative_threshold': .1, 'absolute_threshold_kw': 1., 'mad_multiplier': 6.}
        events = [{'start': str(index[1000]), 'end': str(index[1009])}]
        before = raw.copy()
        masked, points, _, _ = mask_family(raw, raw[['device']], str(index[0]), str(index[-1]), events, recipe)
        pd.testing.assert_frame_equal(raw, before)
        self.assertTrue(masked.loc[index[1000:1012], 'device'].isna().all())
        self.assertTrue(pd.isna(masked.loc[index[900], 'device']))
        self.assertEqual(set(points.rule), {'isolated', 'redbox'})
        filled, _, _ = impute_table(masked.assign(total_load=masked.sum(axis=1, min_count=1)))
        self.assertTrue(filled.loc[index[1000:1012], 'device'].eq(100.).all())
        self.assertEqual(filled.loc[index[900], 'device'], 100.)

    def test_long_combined_gap_is_not_restored_to_bad_values(self):
        from config.aidc_hvac_load_5min.scripts.v2.outlier_remove_data.clean_hvac_data import mask_family
        from config.aidc_hvac_load_5min.scripts.imputed_data.impute_hvac_data import impute_table
        index = pd.date_range('2026-08-01', periods=6 * 288, freq='5min', name='time')
        raw = pd.DataFrame({'device': 100.}, index=index)
        raw.iloc[1000:1072] = 500.
        raw.iloc[1072] = np.nan
        recipe = {'baseline_points': 12, 'minimum_baseline_points': 8,
                  'relative_threshold': .1, 'absolute_threshold_kw': 1., 'mad_multiplier': 6.}
        events = [{'start': str(index[1000]), 'end': str(index[1071])}]
        masked, _, _, _ = mask_family(raw, raw, str(index[0]), str(index[-1]), events, recipe)
        filled, gaps, _ = impute_table(masked.assign(total_load=masked.sum(axis=1, min_count=1)))
        self.assertTrue(filled.iloc[1000:1073].isna().all().all())
        self.assertEqual(gaps[0]['status'], 'unfilled_long')


if __name__ == '__main__':
    unittest.main()
