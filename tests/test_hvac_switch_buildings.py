"""切换点位跨楼扩展：真实事件保真、不可填缺口、汇总和覆盖保护。"""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
import numpy as np
import pandas as pd
from config.aidc_hvac_load_5min.scripts.imputed_data.prepare_switch_buildings import build, unit_specs, DEVICES, ROUTES, CATEGORIES


def fixture(root):
    grid = pd.date_range('2026-07-01', periods=4 * 288, freq='5min', name='time')
    rows, parts = [], {}
    for building in ('A1', 'A2', 'A3'):
        for number in range(1, 7):
            label = ('北' if number <= 3 else '南') + str((number - 1) % 3 + 1)
            route = 'route_A' if number % 2 else 'route_B'
            points = [f'{building}_u{number}_p{i}' for i in range(6)]
            raw = pd.DataFrame(50., index=grid, columns=points)
            if building != 'A1':
                raw.iloc[3 * 288:3 * 288 + 12, 0] = .1  # real demand-response/switching dip, must survive
                raw.iloc[2 * 288, 1] = -1.
                raw.iloc[2 * 288 + 2, 2] = np.inf
                raw.iloc[-2:, 0] = np.nan
            for i, point in enumerate(points):
                rows.append({'building': building, '东西南北': label, 'group_route': route[-1], 'spot_id': point,
                             '备注': '冷水机组' if i == 0 else ('冷冻水二次泵' if i == 5 else '泵')})
            for device, columns in zip(DEVICES, (points, points[:5])):
                path = root / 'raw_data' / device / route / f'{building}_{label}_data.csv'
                path.parent.mkdir(parents=True, exist_ok=True)
                raw[columns].to_csv(path)
                parts.setdefault((device, route, building), []).append(raw[columns])
                if building == 'A1':
                    for category in CATEGORIES:
                        dest = root / category / 'hvac_switch' / device / route / path.name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        raw[columns].to_csv(dest)
    for device in DEVICES:
        for route in ROUTES:
            totals = []
            for building in ('A1', 'A2', 'A3'):
                points = pd.concat(parts[device, route, building], axis=1)
                points.assign(total_load=points.sum(axis=1, min_count=len(points.columns))).to_csv(
                    root / 'raw_data' / device / route / f'{building}_data.csv')
                totals.append(points.add_prefix(building + '_'))
            points = pd.concat(totals, axis=1)
            points.assign(total_load=points.sum(axis=1, min_count=len(points.columns))).to_csv(root / 'raw_data' / device / route / 'data.csv')
    path = root / 'analysis/equipment_regroup/mapping.csv'
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    sentinel = root / 'forecast_data/untouched.csv'
    sentinel.parent.mkdir()
    sentinel.write_bytes(b'unchanged')
    return grid


class SwitchBuildingTest(unittest.TestCase):
    def test_complete_build_preserves_real_events_and_existing_assets(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            grid = fixture(root)
            before = {p: p.read_bytes() for p in root.rglob('*.csv')}
            report = build(root)
            self.assertEqual(report['new_business_tables'], 80)
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)
            for category in CATEGORIES:
                for device in DEVICES:
                    for route in ROUTES:
                        parent = root / category / 'hvac_switch' / device / route
                        total = pd.read_csv(parent / 'data.csv', index_col='time')
                        np.testing.assert_allclose(total.total_load, total.drop(columns='total_load').sum(axis=1, min_count=len(total.columns) - 1), equal_nan=True)
                        for building in ('A1', 'A2', 'A3'):
                            self.assertTrue((parent / f'{building}_data.csv').is_file())
            path = root / 'imputed_data/hvac_switch/hvac_all_devices/route_A/A2_北1_data.csv'
            filled = pd.read_csv(path, index_col='time')
            np.testing.assert_array_equal(filled.iloc[3 * 288:3 * 288 + 12, 0], .1)
            self.assertTrue(filled.iloc[-2:, 0].isna().all())
            self.assertEqual(filled.iloc[2 * 288, 1], 50.)
            cleaned = pd.read_csv(root / 'outlier_remove_data/hvac_switch/hvac_all_devices/route_A/A2_北1_data.csv', index_col='time')
            self.assertTrue(pd.isna(cleaned.iloc[2 * 288, 1]))
            self.assertTrue(pd.isna(cleaned.iloc[2 * 288 + 2, 2]))
            with self.assertRaises(FileExistsError):
                build(root)

    def test_conflict_precedes_publication(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            conflict = root / 'imputed_data/hvac_switch/hvac_all_devices/route_A/A2_data.csv'
            conflict.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                build(root)
            self.assertEqual(conflict.read_bytes(), b'keep')
            self.assertFalse((root / 'analysis/hvac_switch/building_extension').exists())

    def test_mapping_and_shared_source_drift_fail_closed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            meta = pd.read_csv(root / 'analysis/equipment_regroup/mapping.csv')
            with self.assertRaises(ValueError):
                unit_specs(meta.iloc[:-1])
            path = root / 'raw_data/hvac_remove_devices/route_A/A2_北1_data.csv'
            table = pd.read_csv(path)
            table.iloc[0, 1] = 99.
            table.to_csv(path, index=False)
            with self.assertRaises(AssertionError):
                build(root)
            self.assertFalse((root / 'imputed_data/hvac_switch/hvac_all_devices/route_A/A2_data.csv').exists())

if __name__ == '__main__':
    unittest.main()
