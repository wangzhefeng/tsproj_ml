"""独立回读真实切换点位扩展：原值、共享点、总分及旧资产保真。"""
from pathlib import Path
import hashlib
import json
import unittest
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1] / 'dataset/aidc_hvac_load_5min'


def read(path):
    return pd.read_csv(path, index_col='time', parse_dates=True, float_precision='round_trip')


class SwitchBuildingAssetsTest(unittest.TestCase):
    def test_real_sources_units_aggregates_and_protected_assets(self):
        report = json.loads((ROOT / 'analysis/hvac_switch/building_extension/manifest.json').read_text())
        self.assertEqual(report['new_business_tables'], 80)
        self.assertFalse(report['forecast_data_touched'])
        for mapping in (report['source_sha256'], report['protected_sha256'], report['output_sha256']):
            for path, digest in mapping.items():
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest, path)
        meta = pd.read_csv(ROOT / 'analysis/equipment_regroup/mapping.csv')
        for device in ('hvac_all_devices', 'hvac_remove_devices'):
            for route in ('route_A', 'route_B'):
                for building in ('A1', 'A2', 'A3'):
                    key = f'{device}/{route}/{building}_data.csv'
                    raw = read(ROOT / 'raw_data' / key).drop(columns='total_load')
                    clean = raw.mask((raw < 0) | np.isinf(raw))
                    cleaned = read(ROOT / 'outlier_remove_data/hvac_switch' / key)
                    pd.testing.assert_frame_equal(cleaned[clean.columns], clean)
                    filled = read(ROOT / 'imputed_data/hvac_switch' / key)
                    if building != 'A1':
                        pd.testing.assert_index_equal(filled.index, raw.index)
                    truth = clean.loc[filled.index]
                    observed = truth.notna().to_numpy()
                    np.testing.assert_array_equal(filled[truth.columns].to_numpy()[observed], truth.to_numpy()[observed])
                    for category, table in [('outlier_remove_data', cleaned), ('imputed_data', filled)]:
                        points = table.drop(columns='total_load')
                        np.testing.assert_allclose(table.total_load, points.sum(axis=1, min_count=len(points.columns)), equal_nan=True)
                        selected = meta[meta.building.eq(building) & meta.group_route.eq(route[-1])]
                        if device == 'hvac_remove_devices':
                            selected = selected[selected['备注'].ne('冷冻水二次泵')]
                        seen = []
                        for label, group in selected.groupby('东西南北'):
                            unit = read(ROOT / category / 'hvac_switch' / device / route / f'{building}_{label}_data.csv')
                            self.assertNotIn('total_load', unit)
                            self.assertEqual(set(unit), set(group.spot_id))
                            pd.testing.assert_frame_equal(unit, points[unit.columns])
                            seen.extend(unit.columns)
                        self.assertEqual(len(seen), len(set(seen)))
                        self.assertEqual(set(seen), set(points))
                        if device == 'hvac_remove_devices':
                            whole = read(ROOT / category / 'hvac_switch/hvac_all_devices' / route / f'{building}_data.csv')
                            pd.testing.assert_frame_equal(points, whole[points.columns])
                grid = read(ROOT / 'raw_data' / device / route / 'data.csv').index
                for category in ('outlier_remove_data', 'imputed_data'):
                    parent = ROOT / category / 'hvac_switch' / device / route
                    total = read(parent / 'data.csv')
                    pd.testing.assert_index_equal(total.index, grid)
                    expected = pd.concat([read(parent / f'{b}_data.csv').drop(columns='total_load').add_prefix(b + '_').reindex(grid)
                                          for b in ('A1', 'A2', 'A3')], axis=1)
                    pd.testing.assert_frame_equal(total[expected.columns], expected)
                    np.testing.assert_allclose(total.total_load, expected.sum(axis=1, min_count=len(expected.columns)), equal_nan=True)

if __name__ == '__main__':
    unittest.main()
