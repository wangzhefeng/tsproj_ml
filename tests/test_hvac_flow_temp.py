"""水温流量准备的公开构建边界：公式、源保真、缺失证据及发布保护。"""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from config.aidc_hvac_load_5min.scripts.v3.forecast_data.prepare_flow_temp import (
    build, feature_spec, read_table, selected_tables, sha256, verify,
)


class FlowTempTest(unittest.TestCase):
    def test_grid_and_duplicate_headers_rejected(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'bad.csv'
            grid = pd.date_range('2026-04-08', periods=2, freq='5min', name='time')
            path.write_text('time,a,a\n2026-04-08 00:00:00,1,2\n2026-04-08 00:05:00,3,4\n')
            with self.assertRaisesRegex(ValueError, '重复列'):
                read_table(path, grid)
            path.write_text('time,a\n2026-04-08 00:00:00,1\n2026-04-08 00:00:00,3\n')
            with self.assertRaisesRegex(ValueError, '时间网格'):
                read_table(path, grid)

    def test_selected_empty_sensor_is_not_fabricated(self):
        points, _ = feature_spec()
        grid = pd.date_range('2026-04-08', periods=2, freq='5min', name='time')
        inputs = {f'raw_data/hvac_flow_temp/{table}': pd.DataFrame(
            {p['point']: 1. for p in points if p['table'] == table}, index=grid)
            for table in ('data_temp.csv', 'data_flow.csv')}
        inputs['raw_data/hvac_flow_temp/data_temp.csv']['31_15_0_16_1_14_0'] = np.nan
        with self.assertRaisesRegex(ValueError, '全空'):
            selected_tables(inputs)

    def test_build_roundtrip_formulas_and_preserved_versions(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            grid = pd.date_range('2026-04-08', periods=10 * 288, freq='5min', name='time')
            points, _ = feature_spec()
            values = {'chw_flow': 100.25, 'cw_flow': 200.5, 'cond_in_temp': 25.,
                      'cond_out_temp': 30., 'evap_out_temp': 15., 'tower_out_temp': 24.}
            tables = {'data_temp.csv': {}, 'data_flow.csv': {}}
            for item in points:
                value = next((v for key, v in values.items() if item['feature'].endswith('_' + key)), 20.25)
                tables[item['table']][item['point']] = pd.Series(value, index=grid)
            paths = []
            for name, columns in tables.items():
                frame = pd.DataFrame(columns, index=grid)
                frame.iloc[6 * 288:6 * 288 + 3, 0] = np.nan
                frame.iloc[8 * 288:8 * 288 + 285, 1] = np.nan
                # 未选列的全空值不应被填造或进入基础特征。
                frame['excluded_sensor'] = np.nan
                path = root / 'raw_data/hvac_flow_temp' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path)
                paths.append(path)
            for devices in ('hvac_all_devices', 'hvac_remove_devices'):
                directory = root / 'forecast_data/data_v3' / devices / 'A1_all'
                directory.mkdir(parents=True)
                for name, value in [('data.csv', 31.125), ('data_dr_adjusted.csv', 29.625)]:
                    path = directory / name
                    pd.DataFrame({'hvac_total_load_AB': value, 'it_subset_load': 8.75}, index=grid).to_csv(path)
                    paths.append(path)
            recipe = {'start': str(grid[0]), 'end': str(grid[-1]),
                      'input_sha256': {str(p.relative_to(root)): sha256(p) for p in paths}}
            drifted = json.loads(json.dumps(recipe))
            drifted['input_sha256'][str(paths[0].relative_to(root))] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'SHA漂移'):
                build(root, drifted)
            self.assertFalse((root / 'imputed_data').exists())
            report = build(root, recipe)
            self.assertEqual(report['feature_count'], 76)
            self.assertEqual(report['selected_points'], 58)
            self.assertEqual(verify(root, recipe)['status'], 'passed')
            features = pd.read_csv(root / 'forecast_data/data_v3/hvac_flow_temp/features.csv')
            np.testing.assert_array_equal(features.hvac_ft_n1_cond_delta_temp, 5.)
            np.testing.assert_array_equal(features.hvac_ft_n1_water_lift_proxy, 10.)
            np.testing.assert_array_equal(features.hvac_ft_n1_reject_proxy, 1002.5)
            self.assertFalse(features.isna().to_numpy().any())
            for path in paths:
                self.assertEqual(sha256(path), recipe['input_sha256'][str(path.relative_to(root))])
            original = pd.read_csv(root / 'forecast_data/data_v3/hvac_all_devices/A1_all/data_with_flow_temp.csv')
            adjusted = pd.read_csv(root / 'forecast_data/data_v3/hvac_all_devices/A1_all/data_dr_adjusted_with_flow_temp.csv')
            np.testing.assert_array_equal(original.hvac_total_load_AB, 31.125)
            np.testing.assert_array_equal(adjusted.hvac_total_load_AB, 29.625)
            pd.testing.assert_frame_equal(original[features.columns], adjusted[features.columns])
            with self.assertRaises(FileExistsError):
                build(root, recipe)
            # 从非仓库工作目录通过真实CLI复核，不能依赖隐含PYTHONPATH。
            recipe_path = root / 'recipe.json'
            recipe_path.write_text(json.dumps(recipe), encoding='utf-8')
            script = Path(__file__).resolve().parents[1] / 'config/aidc_hvac_load_5min/scripts/v3/forecast_data/prepare_flow_temp.py'
            result = subprocess.run([sys.executable, str(script), '--root', str(root),
                                     '--recipe', str(recipe_path), '--verify-only'],
                                    cwd=root, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout)['status'], 'passed')
            # 真实落盘损坏必须被独立校验拒绝。
            path = root / 'forecast_data/data_v3/hvac_flow_temp/features.csv'
            damaged = features.copy()
            damaged.loc[0, 'hvac_ft_n1_reject_proxy'] = 0.
            damaged.to_csv(path, index=False)
            with self.assertRaises((ValueError, AssertionError)):
                verify(root, recipe)


if __name__ == '__main__':
    unittest.main()
