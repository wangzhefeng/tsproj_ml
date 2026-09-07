"""天气取数包络来自真实目标时间轴，不读取旧天气值、不冒充逐请求验收。"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from test_weather_compiler import config_fixture


class WeatherRequestPlanningTest(unittest.TestCase):
    def test_fixed_origins_match_real_runtime_without_fitting(self):
        from data_loading import SourceRegistry
        from model_pipeline.runner import CanonicalBaseModelRunner
        from scripts.plan_weather_requests import plan_single_model
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = config_fixture(root)
            times = pd.date_range('2026-01-01', periods=48, freq='h')
            pd.DataFrame({'time': times, 'load': range(48)}).to_csv(root / 'target.csv', index=False)
            config = replace(config, data=replace(config.data, sources=(config.data.sources[0],)))
            result = plan_single_model(config, root, origin=times[-1])
            runner = CanonicalBaseModelRunner(config, SourceRegistry(config.data, root), times[-1])
            self.assertEqual(result['groups'][0]['supervised_origins'], [t.isoformat() for t in runner.supervised_origins])

    def test_fixed_window_is_bounded_by_actual_supervised_timeline(self):
        from scripts.plan_weather_requests import plan_single_model
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = config_fixture(root)
            times = pd.date_range('2026-01-01', periods=48, freq='h')
            pd.DataFrame({'time': times, 'load': range(48)}).to_csv(root / 'target.csv', index=False)
            # 损坏天气资产不妨碍目标几何规划；规划不读取其值，也不证明它合格。
            (root / 'raw.csv').write_text('invalid weather fixture')
            result = plan_single_model(config, root, origin=times[-1])
            group = result['groups'][0]
            self.assertEqual(group['supervised_origins'], [t.isoformat() for t in times[42:46]])
            self.assertEqual(group['label_start'], times[43].isoformat())
            self.assertEqual(group['label_end'], (times[-1] + pd.Timedelta(hours=2)).isoformat())
            self.assertFalse(result['runtime_requests_verified'])
            self.assertFalse(result['weather_values_verified'])

    def test_irregular_target_timeline_is_rejected(self):
        from scripts.plan_weather_requests import plan_single_model
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = config_fixture(root)
            times = pd.date_range('2026-01-01', periods=48, freq='h').delete(44)
            pd.DataFrame({'time': times, 'load': range(47)}).to_csv(root / 'target.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'regular'):
                plan_single_model(config, root, origin=times[-1])

    def test_gap_outside_requested_training_window_does_not_expand_scope(self):
        from scripts.plan_weather_requests import plan_single_model
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = config_fixture(root)
            times = pd.date_range('2026-01-01', periods=48, freq='h').delete(20)
            pd.DataFrame({'time': times, 'load': range(47)}).to_csv(root / 'target.csv', index=False)
            result = plan_single_model(config, root, origin=times[-1])
            self.assertEqual(result['label_start'], '2026-01-02T19:00:00')

    def test_calendar_includes_variable_horizon_runner_envelopes(self):
        from scripts.plan_weather_requests import plan_single_model
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = config_fixture(root)
            times = pd.date_range('2025-10-01', '2026-07-31', freq='D')
            pd.DataFrame({'time': times, 'load': range(len(times))}).to_csv(root / 'target.csv', index=False)
            config = replace(config, data=replace(config.data, sources=(config.data.sources[0],)), problem=replace(config.problem, freq='1D', horizon=31), validation={'forecast_origin': str(times[-1]), 'horizon_mode': 'calendar_month', 'train_window_days': 120, 'fold_count': 6, 'stride_months': 1})
            result = plan_single_model(config, root, origin=times[-1])
            self.assertEqual({g['horizon'] for g in result['groups']}, {28, 30, 31})
            self.assertEqual(result['label_end'], '2026-08-31T00:00:00')
