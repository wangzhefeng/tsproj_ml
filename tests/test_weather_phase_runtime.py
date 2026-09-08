"""历史天气文件/列分流的真实编译、缓存与生命周期验证。"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from data_loading import SourceRegistry
from forecasting_core.specs import ColumnSpec, DataSourceSpec
from feature_engineering.cache import compute_raw_design_fingerprint
from model_pipeline.runner import run_canonical_config
from model_pipeline.supervised_design import SupervisedDesignBuilder
from tests import test_canonical_runtime_smoke as smoke


class WeatherPhaseRuntimeTest(unittest.TestCase):
    def test_history_training_test_future_and_cache_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            times = pd.date_range('2026-01-01', periods=52, freq='1h')
            target = root / 'target.csv'
            pd.DataFrame({'time': times, 'load': 100. + np.arange(52)}).to_csv(target, index=False)
            history = root / 'weather_history.csv'
            future = root / 'weather_future.csv'
            pd.DataFrame({'time': times, 'rt_tt2': 280., 'pred_tt2': 270.}).to_csv(history, index=False)
            base = smoke.CanonicalRuntimeSmokeTest().build_config(target, mode='point')
            weather = DataSourceSpec(name='weather', source_type='file', time_col='time',
                columns=(ColumnSpec('rt_tt2', 'known_future'), ColumnSpec('pred_tt2', 'ignored')),
                history_path=str(history), future_path=str(future), availability='forecast_origin',
                inference_columns={'rt_tt2': 'pred_tt2'})
            config = replace(base, data=replace(base.data, sources=(*base.data.sources, weather)))
            origin = times[47]
            builder = SupervisedDesignBuilder(config, SourceRegistry(config.data, root))
            train, _ = builder.training_row(origin)
            predicted, _ = builder.forecast_designs(origin)
            positions = [i for i, name in enumerate(builder.feature_schema) if 'rt_tt2' in name]
            self.assertTrue(positions)
            np.testing.assert_array_equal(train[0][..., positions], 280.)
            np.testing.assert_array_equal(predicted[0][..., positions], 270.)
            before = compute_raw_design_fingerprint(config, base_dir=root, origin=origin, generators={})
            # 真正未来文件可只含预报列；与历史时间重叠也不能拼接或覆盖历史。
            pd.DataFrame({'time': times, 'pred_tt2': 260.}).to_csv(future, index=False)
            after = compute_raw_design_fingerprint(config, base_dir=root, origin=origin, generators={})
            self.assertEqual(before, after)
            actual_future, _ = builder.forecast_designs(origin, data_phase='future')
            np.testing.assert_array_equal(actual_future[0][..., positions], 260.)
            historical_again, _ = builder.forecast_designs(origin)
            np.testing.assert_array_equal(historical_again[0], predicted[0])
            # 完整历史生命周期（含末次留出预测）不读 future；使用真正 Ridge 拟合。
            absent = replace(weather, future_path=str(root / 'never-created.csv'), inference_columns={'rt_tt2': 'pred_tt2'})
            config = replace(config, data=replace(config.data, sources=(*base.data.sources, absent)))
            result = run_canonical_config(config, output_root=root / 'results')
            self.assertTrue(result.bundle is not None)
            self.assertEqual(len(list((root / 'results').rglob('prediction.csv'))), 1)


if __name__ == '__main__':
    unittest.main()
