"""单版本分楼准备：不可填缺口保留、严格汇总、预测资产保真。"""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import json
import numpy as np
import pandas as pd
from config.aidc_hvac_load_5min.scripts.imputed_data.gap_filling import fill_series
from config.aidc_hvac_load_5min.scripts.imputed_data.prepare_buildings import prepare, aggregate, FAMILIES
from config.aidc_hvac_load_5min.scripts.analysis.analyze_building_events import candidates
from config.aidc_hvac_load_5min.scripts.outlier_remove_data.prepare_building_events import prepare as correct_events
from config.aidc_hvac_load_5min.scripts.raw_data.migrate_hvac_data import sha256_file

class BuildingExtensionTest(unittest.TestCase):
    def test_unfillable_policy_is_explicit_and_preserves_observations(self):
        grid = pd.date_range('2026-01-01', periods=30 * 288, freq='5min', name='time')
        source = pd.Series(10., index=grid, name='p')
        source.iloc[100:7500] = np.nan
        source.iloc[-3:] = np.nan
        with self.assertRaises(ValueError):
            fill_series(source, output_start=grid[0])
        result, audit = fill_series(source, output_start=grid[0], on_unfillable='keep')
        pd.testing.assert_series_equal(result, source)
        self.assertEqual([r['status'] for r in audit], ['unresolved', 'unresolved'])
        with self.assertRaises(ValueError):
            fill_series(source, output_start=grid[0], on_unfillable='zero')

    def test_candidates_are_unapproved_and_independently_bounded(self):
        grid = pd.date_range('2026-08-01', periods=100, freq='5min')
        frame = pd.DataFrame({'all': 200., 'reduced': 160.}, index=grid)
        frame.iloc[24:36] = [100., 80.]
        rows = candidates(frame)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['start'], str(grid[24]))
        self.assertEqual(rows[0]['end_exclusive'], str(grid[36]))
        self.assertFalse(rows[0]['approved'])

    def test_builds_all_families_without_touching_forecast(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            grid = pd.date_range('2026-01-01', periods=4 * 288, freq='5min', name='time')
            source_bytes = {}
            for family in FAMILIES:
                raw_parts = []
                for building in ('A1', 'A2', 'A3'):
                    col = building + '_' + family.replace('/', '_').replace('hvac_remove_devices', 'hvac_all_devices')
                    points = pd.DataFrame({col: 10.}, index=grid)
                    if building != 'A1':
                        points.iloc[-2:] = np.nan
                    raw_parts.append(points.add_prefix(building + '_'))
                    table = points.assign(total_load=points.sum(axis=1, min_count=1))
                    path = root / 'raw_data' / family / f'{building}_data.csv'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    table.to_csv(path)
                    source_bytes[path] = path.read_bytes()
                    if building == 'A1':
                        for category in ('imputed_data', 'outlier_remove_data'):
                            target = root / category / family / 'A1_data.csv'
                            target.parent.mkdir(parents=True, exist_ok=True)
                            table.to_csv(target)
                total = pd.concat(raw_parts, axis=1)
                total['total_load'] = total.sum(axis=1, min_count=len(total.columns))
                total.to_csv(root / 'raw_data' / family / 'data.csv')
            sentinel = root / 'forecast_data/sentinel.csv'
            sentinel.parent.mkdir()
            sentinel.write_bytes(b'never-recompute')
            report = prepare(root)
            self.assertFalse(report['forecast_data_touched'])
            self.assertEqual(sentinel.read_bytes(), b'never-recompute')
            for path, content in source_bytes.items():
                self.assertEqual(path.read_bytes(), content)
            for category in ('imputed_data', 'outlier_remove_data'):
                for family in FAMILIES:
                    total = pd.read_csv(root / category / family / 'data.csv')
                    self.assertTrue(total.total_load.iloc[-2:].isna().all())
                    np.testing.assert_array_equal(total.total_load.iloc[:-2], 30.)
            with self.assertRaises(FileExistsError):
                prepare(root)
            event = {'event_id': 'reviewed', 'start': str(grid[3 * 288]),
                     'rebound_start': str(grid[3 * 288 + 3]), 'end_exclusive': str(grid[3 * 288 + 6])}
            recipe = {'approved': True, 'events': {'A2': [event], 'A3': [event]},
                      'input_sha256': {f'{category}/{family}/{b}_data.csv': sha256_file(root / category / family / f'{b}_data.csv')
                                       for category in ('raw_data', 'imputed_data') for family in FAMILIES[:-1]
                                       for b in ('A2', 'A3')}}
            result = correct_events(root, recipe)
            self.assertEqual(result['unresolved_point_events'], 0)
            self.assertEqual(result['estimated_point_events'], 4)
            self.assertEqual(sentinel.read_bytes(), b'never-recompute')
            with self.assertRaises(FileExistsError):
                correct_events(root, recipe)

if __name__ == '__main__':
    unittest.main()
