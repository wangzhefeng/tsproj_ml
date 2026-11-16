"""HVAC 天气绑定合同：配置直连共享 processed 5min 资产，不训练模型。"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_loading import InformationSetRequest, SourceRegistry
from forecasting_core.specs.config import parse_data_spec
from scripts import build_scenario_weather as builder
from config.aidc_hvac_load_5min.scripts.weather_data import bind_weather as binder
from config.aidc_hvac_load_5min.scripts.weather_data.weather_source import weather_source


MAPPING = {'rt_tt2': 'pred_tt2', 'cal_rh': 'pred_rh', 'rt_ssr': 'pred_ssrd',
           'rt_ws10': 'pred_ws10', 'rt_ps': 'pred_ps', 'rt_rain': 'pred_rain'}
PROCESSED = Path('dataset/shared/weather/processed/weather_5min.csv')


def publish_processed_asset(root, frame):
    """按共享资产合同写入 5min processed CSV 与 meta，供 read_processed 校验。"""
    dest = root / PROCESSED
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest)
    meta = {'file': dest.name, 'sha256_file': hashlib.sha256(dest.read_bytes()).hexdigest(),
            'rows': len(frame), 'role': 'shared_processed', 'freq': '5min'}
    dest.with_suffix('.meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return dest


def synthetic_asset(times):
    columns = [*MAPPING, *MAPPING.values()]
    return pd.DataFrame({column: np.arange(len(times), dtype=float) + i
                         for i, column in enumerate(columns)}, index=pd.DatetimeIndex(times, name='ts'))


class WeatherHvacBindingTest(unittest.TestCase):
    def run_binder(self, root, data_version, *, asset_times, targets):
        for path, times in targets.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({'time': times, 'hvac_total_load_AB': 1.}).to_csv(path, index=False)
        publish_processed_asset(root, synthetic_asset(asset_times))
        with patch.object(binder, 'ROOT', root), patch.object(builder, 'ROOT', root):
            return binder.bind_weather(root / 'dataset/aidc_hvac_load_5min', PROCESSED,
                                       data_version=data_version)

    def test_v3_binding_without_route_partition_ignores_named_derivatives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            times = pd.date_range('2026-04-08', periods=288, freq='5min')
            targets = {}
            for devices in ('hvac_all_devices', 'hvac_remove_devices'):
                path = root / 'dataset/aidc_hvac_load_5min/forecast_data/data_v3' / devices / 'A1_all/data.csv'
                path.parent.mkdir(parents=True)
                targets[path] = times
                # 同目录反事实派生不是原天气绑定的自动消费目标。
                pd.DataFrame({'time': times, 'hvac_total_load_AB': 4.}).to_csv(
                    path.with_name('data_dr_adjusted.csv'), index=False)
            rows = self.run_binder(root, 'data_v3', asset_times=times, targets=targets)
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertIn('/data_v3/', row['target_file'])
                self.assertNotIn('/route_', row['target_file'])
                self.assertTrue(row['target_file'].endswith('/data.csv'))
                self.assertEqual((row['start'], row['end'], row['rows']),
                                 (str(times[0]), str(times[-1]), len(times)))
                self.assertEqual(row['target_sha256'],
                                 hashlib.sha256((root / row['target_file']).read_bytes()).hexdigest())
                self.assertEqual(row['weather_file'], str(PROCESSED))
                self.assertEqual(row['weather_sha256'],
                                 hashlib.sha256((root / PROCESSED).read_bytes()).hexdigest())
            self.assertFalse((root / 'dataset/aidc_hvac_load_5min/weather_data').exists())

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

    def test_independent_target_windows_and_manifest_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / 'dataset/aidc_hvac_load_5min/forecast_data/data_v1/family/route_A'
            windows = {
                'A1_data': pd.date_range('2026-08-31 23:30', periods=12, freq='5min'),
                'A1_data_with_it': pd.date_range('2026-09-01 01:00', periods=12, freq='5min'),
            }
            asset_times = pd.date_range('2026-08-31 23:00', periods=36, freq='5min')
            rows = self.run_binder(root, 'data_v1', asset_times=asset_times,
                                   targets={inputs / f'{name}.csv': times
                                            for name, times in windows.items()})
            self.assertEqual({Path(row['target_file']).stem for row in rows}, set(windows))
            for row in rows:
                self.assertEqual(row['rows'], len(windows[Path(row['target_file']).stem]))
                self.assertEqual(row['start'], str(windows[Path(row['target_file']).stem][0]))
            manifest_dir = root / 'dataset/aidc_hvac_load_5min/analysis/weather_binding/data_v1'
            with patch.object(binder, 'ROOT', root), patch.object(builder, 'ROOT', root):
                with patch.object(sys, 'argv', ['bind_weather.py', '--data-version', 'data_v1']):
                    binder.main()
                    with self.assertRaises(FileExistsError):
                        binder.main()
                with patch.object(sys, 'argv', ['bind_weather.py', '--data-version', 'data_v1', '--replace']):
                    binder.main()
            manifest = pd.read_csv(manifest_dir / 'manifest.csv')
            self.assertEqual(set(manifest.target_file), {row['target_file'] for row in rows})

    def test_invalid_grid_uncovered_window_and_missing_weather_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / 'dataset/aidc_hvac_load_5min/forecast_data/data_v1/family/route_A'
            inputs.mkdir(parents=True)
            path = inputs / 'data.csv'
            asset_times = pd.date_range('2026-08-01', periods=288, freq='5min')
            for times in (['2026-08-01 00:00', '2026-08-01 00:10'],
                          ['2026-08-01 00:00', '2026-08-01 00:00'],
                          ['2026-08-01 00:01', '2026-08-01 00:06']):
                pd.DataFrame({'time': times}).to_csv(path, index=False)
                with patch.object(binder, 'ROOT', root), self.assertRaisesRegex(ValueError, '5min 网格'):
                    self.run_binder(root, 'data_v1', asset_times=asset_times, targets={})
            # 目标窗口超出共享资产覆盖。
            pd.DataFrame({'time': ['2026-08-02 00:00', '2026-08-02 00:05']}).to_csv(path, index=False)
            short = pd.date_range('2026-08-01', periods=12, freq='5min')
            with patch.object(binder, 'ROOT', root), self.assertRaisesRegex(ValueError, '未完整覆盖目标窗口'):
                self.run_binder(root, 'data_v1', asset_times=short, targets={})
            # 必需天气列缺失：共享资产校验直接拒绝，不产生绑定。
            for column in ('rt_tt2', 'pred_tt2'):
                frame = synthetic_asset(asset_times)
                frame.loc[asset_times[0], column] = np.nan
                publish_processed_asset(root, frame)
                with patch.object(binder, 'ROOT', root), patch.object(builder, 'ROOT', root), \
                     self.assertRaisesRegex(ValueError, '缺失或非有限'):
                    binder.bind_weather(root / 'dataset/aidc_hvac_load_5min', PROCESSED, data_version='data_v1')
            self.assertFalse((root / 'dataset/aidc_hvac_load_5min/weather_data').exists())

    def test_real_assets_bind_and_materialize_both_phases(self):
        root = Path(__file__).resolve().parents[1]
        scenario = root / 'dataset/aidc_hvac_load_5min'
        asset = pd.read_csv(root / PROCESSED, float_precision='round_trip',
                            index_col='ts', parse_dates=True)
        source = weather_source()
        self.assertEqual(source['history_path'], str(PROCESSED))
        self.assertEqual(source['inference_columns'], MAPPING)
        for data_version, expected in (('data_v1', 32), ('data_v2', 32), ('data_v3', 2)):
            manifest = pd.read_csv(scenario / f'analysis/weather_binding/{data_version}/manifest.csv')
            pattern = '*/A1_all/data.csv' if data_version == 'data_v3' else '*/*/*.csv'
            targets = {str(p.relative_to(root)) for p in
                       (scenario / 'forecast_data' / data_version).glob(pattern)}
            self.assertEqual(len(targets), expected)
            self.assertEqual(set(manifest.target_file), targets)
            for row in manifest.to_dict('records'):
                with self.subTest(target=row['target_file']):
                    target = pd.read_csv(root / row['target_file'])
                    times = pd.DatetimeIndex(pd.to_datetime(target['time']))
                    self.assertEqual((row['start'], row['end'], row['rows']),
                                     (str(times[0]), str(times[-1]), len(times)))
                    self.assertEqual(row['target_sha256'],
                                     hashlib.sha256((root / row['target_file']).read_bytes()).hexdigest())
                    slice_frame = asset.loc[times]
                    self.assertEqual(len(slice_frame), len(times))
                    self.assertTrue(np.isfinite(
                        slice_frame[list(MAPPING) + list(MAPPING.values())].to_numpy()).all())
                    if data_version != 'data_v1':
                        continue
                    route = Path(row['target_file']).parent.name.removeprefix('route_')
                    target_column = 'hvac_total_load_' + route
                    self.assertIn(target_column, target.columns)
                    target_source = {'name': 'target', 'source_type': 'file', 'time_col': 'time',
                                     'history_path': row['target_file'], 'availability': 'source_time',
                                     'columns': [{'name': target_column, 'role': 'target', 'categorical': False}]}
                    registry = SourceRegistry(parse_data_spec(
                        {'sources': [target_source, source]}, 'hvac-weather'), root)
                    positions = [1, len(times) // 2, len(times) - 1]
                    for training in (True, False):
                        request = InformationSetRequest(times[0], times[positions], (),
                                                        target_access='supervised_labels' if training else 'history_only')
                        info = registry.materialize(request)
                        physical = list(MAPPING) if training else list(MAPPING.values())
                        np.testing.assert_allclose(info.known_future['weather'][list(MAPPING)].to_numpy(),
                                                   slice_frame.iloc[positions][physical].to_numpy())


if __name__ == '__main__':
    unittest.main()
