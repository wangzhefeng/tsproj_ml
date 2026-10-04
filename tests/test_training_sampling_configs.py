"""迁移配置的模型组采样合同；真实首步诊断不替代完整业务回测。"""
from collections import Counter
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from config.config_loader import load_yaml_config
from data_loading import BUILTIN_GENERATORS, SourceRegistry
from forecasting_core.specs import ForecastConfigSpec
from forecasting_core.specs.temporal import history_start
from model_pipeline.supervised_design import SupervisedDesignBuilder, supervised_candidate_origins
from model_training.strategies import target_plan_for_config
from models.wrappers.lightgbm import LightGBMModel

ROOT = Path(__file__).resolve().parents[1]
LIANTONG = ROOT / 'config/aidc_electricity_computility/electricity/2026-08-31/liantong_IT'
SCENARIOS = ('aidc_load_15min_daily', 'aidc_load_15min_rolling', 'aidc_load_15min_short')


class TrainingSamplingConfigsTest(unittest.TestCase):
    def test_unshared_model_groups_keep_dense_origins_across_the_physical_matrix(self):
        failures = []
        counts = Counter()
        roots = [ROOT / 'config' / name / route
                 for name in SCENARIOS for route in ('route_A', 'route_B', 'route_AB')] + [LIANTONG]
        for root in roots:
            for path in sorted(root.rglob('*.yaml')):
                config = load_yaml_config(path)
                if not isinstance(config, ForecastConfigSpec) or config.validation.get('training_window') is None:
                    continue
                calls = Counter(target_plan_for_config(config).model_indices)
                sampling = config.validation.get('training', {}).get('origin_sampling')
                shared = max(calls.values()) > 1
                counts[shared] += 1
                # 每个模型只有一个调用块时，不把每日发报频率当作训练频率。
                if (sampling is not None) != shared:
                    failures.append(str(path.relative_to(ROOT)))
        self.assertGreater(counts[False], 0)
        self.assertGreater(counts[True], 0)
        self.assertEqual(failures, [], f'{len(failures)} sampling mismatches; first: {failures[:5]}')

    def test_liantong_direct_first_model_learns_with_unchanged_leaf_parameters(self):
        config = load_yaml_config(LIANTONG / 'add_weather/lgbm_direct.yaml')
        assert isinstance(config, ForecastConfigSpec)
        origin = pd.Timestamp(config.validation['forecast_origin'])
        assert isinstance(origin, pd.Timestamp)
        offset = pd.tseries.frequencies.to_offset(config.problem.freq)
        registry = SourceRegistry(config.data, ROOT, generators=BUILTIN_GENERATORS)
        builder = SupervisedDesignBuilder(config, registry,
            history_start=history_start(config.validation, origin, offset))
        origins = supervised_candidate_origins(builder, origin)
        frames = []
        # 仅编译第一个独立模型，避免测试物化全部288个模型的设计。
        for start in range(0, len(origins), 128):
            requests = tuple(builder.request(value) for value in origins[start:start + 128])
            # 与生产training_rows一致：物化训练实测天气，编译仍按各原点做as-of校验。
            training_requests = tuple(builder.request(value, target_access='supervised_labels')
                                      for value in origins[start:start + 128])
            compiled = builder.compiler.compile_batch(
                tuple(registry.materialize(request) for request in training_requests), requests, horizon_steps=(1,))
            frames.extend(item.frame.loc[:, item.schema.feature_names] for item in compiled)
        X = pd.concat(frames, ignore_index=True)
        first_designs, _ = builder.training_row(origins[0])
        np.testing.assert_allclose(X.iloc[:1].to_numpy(), first_designs[0], rtol=0, atol=1e-9)
        source = config.data.sources[0]
        assert source.history_path is not None and source.time_col is not None
        target = pd.read_csv(ROOT / source.history_path, parse_dates=[source.time_col]).set_index(source.time_col)
        y = target.loc[pd.DatetimeIndex(origins) + offset, config.problem.targets[0]].to_numpy()
        model = LightGBMModel({**dict(config.estimator.params), 'n_jobs': 1}, log_params=False)
        model.fit(X, y)
        self.assertGreater(len(np.unique(y)), 1)
        self.assertEqual(model.model.get_params()['min_child_samples'], 20)
        self.assertTrue(any(tree['num_leaves'] > 1 for tree in model.model.booster_.dump_model()['tree_info']))
        self.assertGreater(len(np.unique(model.predict(X))), 1)
