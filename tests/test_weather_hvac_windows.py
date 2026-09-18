"""HVAC 天气逐目标窗口构建合同，不训练模型。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from data_loading import InformationSetRequest, SourceRegistry
from forecasting_core.specs.config import parse_data_spec
from scripts import build_scenario_weather as builder
from config.aidc_hvac_load_5min.scripts import prepare_weather as scenario


MAPPING = {'rt_tt2': 'pred_tt2', 'cal_rh': 'pred_rh', 'rt_ssr': 'pred_ssrd',
           'rt_ws10': 'pred_ws10', 'rt_ps': 'pred_ps', 'rt_rain': 'pred_rain'}


class WeatherHvacWindowsTest(unittest.TestCase):
    def test_authorized_tail_interpolation_is_local_and_audited(self):
        times = pd.date_range('2026-09-16 15:00', '2026-09-16 23:00', freq='1h')
        hourly = pd.DataFrame({col: np.arange(9, dtype=float) + 280.
                               for col in ('rt_tt2', 'rt_dt', 'rt_ws10', 'rt_rain')}, index=times)
        hourly.loc[times[1:8], ['rt_tt2', 'rt_dt']] = np.nan
        hourly.loc[times[1:6], ['rt_ws10', 'rt_rain']] = np.nan
        hourly['cal_rh'] = builder.calc_rh(hourly.rt_tt2, hourly.rt_dt)
        original = hourly.copy(deep=True)
        filled, audit = builder.repair_source_intervals(hourly)
        pd.testing.assert_frame_equal(hourly, original)
        self.assertTrue(np.isfinite(filled.to_numpy()).all())
        np.testing.assert_allclose(filled.rt_tt2, np.arange(9) + 280.)
        self.assertEqual(len(audit), 31)
        self.assertTrue(all(row['method'] in ('linear_interpolation', 'derived_relative_humidity') for row in audit))
        # 未授权日期的缺失不随此特例填补。
        outside = original.copy()
        outside.index = outside.index - pd.Timedelta(days=1)
        unchanged, audit = builder.repair_source_intervals(outside)
        pd.testing.assert_frame_equal(outside, unchanged)
        self.assertEqual(audit, [])

    def test_independent_target_windows_and_source_fragments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / 'dataset/aidc_hvac_load_5min/forecast_data/family/route_A'
            inputs.mkdir(parents=True)
            windows = {
                'A1_data': pd.date_range('2026-08-31 23:30', periods=12, freq='5min'),
                'A1_data_with_it': pd.date_range('2026-09-01 01:00', periods=12, freq='5min'),
            }
            originals = {}
            for name, times in windows.items():
                path = inputs / f'{name}.csv'
                pd.DataFrame({'time': times, 'hvac_total_load_A': 1., 'hvac_total_load_B': 2.}).to_csv(path, index=False)
                originals[path] = path.read_bytes()
            hourly = pd.DataFrame(
                {col: np.arange(4, dtype=float) + i for i, col in enumerate(list(MAPPING) + list(MAPPING.values()))},
                index=pd.date_range('2026-08-31 23:00', periods=4, freq='1h'),
            )
            with patch.object(scenario, 'ROOT', root), patch.object(builder, 'ROOT', root), patch.object(builder, 'SOURCE_REPAIR_REPORT', root / 'absent.json'):
                outputs = scenario.build_hvac_weather(hourly, {'sources': [], 'offline_interpolation': [], 'processed_asset': {}})
            self.assertEqual(len(outputs), 2)
            for row in outputs:
                target = root / row['target_file']
                weather = root / row['file']
                frame = pd.read_csv(weather)
                times = windows[target.stem]
                pd.testing.assert_index_equal(pd.DatetimeIndex(pd.to_datetime(frame['ts'])), times, check_names=False)
                np.testing.assert_allclose(frame[list(MAPPING)].to_numpy(), hourly.loc[times.floor('1h'), list(MAPPING)].to_numpy())
                source = yaml.safe_load((root / row['source_yaml']).read_text())['data']['sources'][0]
                self.assertEqual(source['history_path'], row['file'])
                self.assertEqual(source['inference_columns'], MAPPING)
                self.assertEqual(source['availability'], 'forecast_origin')
                self.assertNotIn('future_path', source)
                meta = json.loads(weather.with_suffix('.meta.json').read_text())
                self.assertEqual(meta['target_sha256'], hashlib.sha256(originals[target]).hexdigest())
                self.assertEqual(target.read_bytes(), originals[target])

    def test_invalid_grid_and_missing_weather_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / 'dataset/aidc_hvac_load_5min/forecast_data/family/route_A'
            inputs.mkdir(parents=True)
            path = inputs / 'data.csv'
            hourly = pd.DataFrame({col: [1.] for col in list(MAPPING) + list(MAPPING.values())},
                                  index=pd.DatetimeIndex(['2026-08-01']))
            for times in (['2026-08-01 00:00', '2026-08-01 00:10'],
                          ['2026-08-01 00:00', '2026-08-01 00:00'],
                          ['2026-08-01 00:01', '2026-08-01 00:06']):
                pd.DataFrame({'time': times}).to_csv(path, index=False)
                with patch.object(scenario, 'ROOT', root), self.assertRaisesRegex(ValueError, '5min 网格'):
                    scenario.build_hvac_weather(hourly, {'sources': [], 'offline_interpolation': [], 'processed_asset': {}})
            pd.DataFrame({'time': ['2026-08-01 00:00', '2026-08-01 00:05']}).to_csv(path, index=False)
            for column in ('rt_tt2', 'pred_tt2'):
                broken = hourly.copy()
                broken[column] = np.nan
                with patch.object(scenario, 'ROOT', root), self.assertRaisesRegex(ValueError, '缺失或非有限'):
                    scenario.build_hvac_weather(broken, {'sources': [], 'offline_interpolation': [], 'processed_asset': {}})
            self.assertFalse((root / 'dataset/aidc_hvac_load_5min/weather_data').exists())

    def test_all_real_assets_align_and_materialize_both_phases(self):
        root = Path(__file__).resolve().parents[1]
        scenario = root / 'dataset/aidc_hvac_load_5min'
        manifest = pd.read_csv(scenario / 'weather_data/manifest.csv')
        targets = {str(p.relative_to(root)) for p in (scenario / 'forecast_data').glob('*/*/*.csv')}
        self.assertEqual(len(targets), 32)
        self.assertEqual(set(manifest.target_file), targets)
        self.assertEqual(len(manifest), len(targets))
        for row in manifest.to_dict('records'):
            with self.subTest(target=row['target_file']):
                target = pd.read_csv(root / row['target_file'])
                frame = pd.read_csv(root / row['file'])
                times = pd.DatetimeIndex(pd.to_datetime(target['time']))
                pd.testing.assert_index_equal(pd.DatetimeIndex(pd.to_datetime(frame['ts'])), times, check_names=False)
                self.assertEqual(len(frame), row['rows'])
                self.assertTrue(np.isfinite(frame[list(MAPPING) + list(MAPPING.values())].to_numpy()).all())
                self.assertEqual(hashlib.sha256((root / row['file']).read_bytes()).hexdigest(), row['sha256'])
                metadata = json.loads((root / row['file']).with_suffix('.meta.json').read_text())
                self.assertEqual(metadata['target_sha256'], hashlib.sha256((root / row['target_file']).read_bytes()).hexdigest())
                source = yaml.safe_load((root / row['source_yaml']).read_text())['data']['sources'][0]
                self.assertEqual(source['inference_columns'], MAPPING)
                route = Path(row['target_file']).parent.name.removeprefix('route_')
                target_column = 'hvac_total_load_' + route
                self.assertIn(target_column, target.columns)
                target_source = {'name': 'target', 'source_type': 'file', 'time_col': 'time',
                                 'history_path': row['target_file'], 'availability': 'source_time',
                                 'columns': [{'name': target_column, 'role': 'target', 'categorical': False}]}
                registry = SourceRegistry(parse_data_spec({'sources': [target_source, source]}, 'hvac-weather'), root)
                positions = [1, len(times) // 2, len(times) - 1]
                for training in (True, False):
                    request = InformationSetRequest(times[0], times[positions], (),
                                                    target_access='supervised_labels' if training else 'history_only')
                    info = registry.materialize(request)
                    physical = list(MAPPING) if training else list(MAPPING.values())
                    np.testing.assert_allclose(info.known_future['weather'][list(MAPPING)].to_numpy(),
                                               frame.iloc[positions][physical].to_numpy())


if __name__ == '__main__':
    unittest.main()
