"""生产 runner 的紧凑训练设计，与独立 single 编译数值对照。"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_loading import SourceRegistry
from forecasting_core.design import IndexedDesign
from forecasting_core.specs import ColumnSpec
from forecasting_core.runtime_resources import RuntimeResourceBudget
from model_pipeline.runner import CanonicalBaseModelRunner
from tests.test_raw_history_window import make_config
from tests.test_liantong_optimization import synthetic_config


class IndexedTrainingTest(unittest.TestCase):
    def test_independent_forecast_compiles_all_calls_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.csv"
            times = pd.date_range("2026-01-01", periods=52, freq="h")
            pd.DataFrame({"time": times, "load": 100 + np.arange(52.) ** 1.5}).to_csv(path, index=False)
            config = make_config(path, strategy="direct")
            runner = CanonicalBaseModelRunner(config, SourceRegistry(config.data, directory), times[47])
            builder = runner.builder
            expected = builder.compiler.compile(builder.registry.materialize(builder.request(runner.origin)),
                                                builder.request(runner.origin)).frame
            with patch.object(builder.compiler, "compile", side_effect=AssertionError("repeated single compile")), \
                 patch.object(builder.compiler, "compile_batch", wraps=builder.compiler.compile_batch) as batch:
                designs, provider = builder.forecast_designs(runner.origin)
                actual = np.concatenate((designs[0], provider(1, (), (), {})))
            self.assertEqual(batch.call_count, 1)
            np.testing.assert_allclose(actual, expected.loc[:, builder.feature_schema].to_numpy(), rtol=1e-10, atol=1e-10)

    def test_native_history_does_not_compile_supervised_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, times = synthetic_config(root, weather=False)
            config = replace(config,
                features=replace(config.features, target_lags={}, datetime_features=(), transformations={}),
                estimator=replace(config.estimator, model_type="ets", params={"seasonal_periods": 12, "candidates": ["ANN"]}),
                validation={**dict(config.validation), "history_steps": 90, "train_history_steps": 48,
                            "train_window_steps": 44, "fold_count": 2})
            with patch("feature_engineering.FeatureCompiler.compile", side_effect=AssertionError("native model must not compile")), \
                 patch("feature_engineering.FeatureCompiler.compile_batch", side_effect=AssertionError("native model must not compile")):
                runner = CanonicalBaseModelRunner(config, SourceRegistry(config.data, root), times[-1])
            self.assertEqual(runner.feature_schema, ())
            self.assertEqual(runner.X_all[0].shape, (44, 0))

    def test_budget_rejects_before_full_design_compilation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.csv"
            times = pd.date_range("2026-01-01", periods=52, freq="h")
            pd.DataFrame({"time": times, "load": np.arange(52.) + 100}).to_csv(path, index=False)
            config = make_config(path, strategy="direct")
            budget = RuntimeResourceBudget(2, 2, 2, 1)
            with patch("model_pipeline.runner._supervised_arrays", side_effect=AssertionError("full allocation before planning")):
                with self.assertRaisesRegex(ValueError, "memory budget"):
                    CanonicalBaseModelRunner(config, SourceRegistry(config.data, directory), times[47], resource_budget=budget)

    def test_source_with_target_and_observed_columns_preserves_both_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.csv"
            times = pd.date_range("2026-01-01", periods=52, freq="h")
            pd.DataFrame({"time": times, "load": np.arange(52.) + 100,
                          "temperature": np.sin(np.arange(52.))}).to_csv(path, index=False)
            config = make_config(path, strategy="direct")
            source = replace(config.data.sources[0], columns=(*config.data.sources[0].columns,
                             ColumnSpec("temperature", "observed_past")), provider="persistence")
            config = replace(config, data=replace(config.data, sources=(source,)),
                             features=replace(config.features, observed_past_lags={"temperature": (2,)},
                                              transformations=config.features.canonical_payload()["transformations"]))
            runner = CanonicalBaseModelRunner(config, SourceRegistry(config.data, directory), times[47])
            for index, origin in enumerate(runner.supervised_origins):
                expected, _ = runner.builder.training_row(origin)
                for actual, want in zip(runner.X_all, expected):
                    np.testing.assert_allclose(actual[index], want[0], rtol=1e-10, atol=1e-10)

    def test_runner_retains_indexed_blocks_and_matches_single_compiler(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.csv"
            times = pd.date_range("2026-01-01", periods=52, freq="h")
            pd.DataFrame({"time": times, "load": 100 + np.arange(52) ** 1.5}).to_csv(path, index=False)
            config = make_config(path, strategy="direct")
            runner = CanonicalBaseModelRunner(config, SourceRegistry(config.data, directory), times[47])
            self.assertIsInstance(runner.X_all[0], IndexedDesign)
            accounting = runner.runtime_resources_payload()["design_storage"]
            # 此 fixture 每个 horizon 一个模型，不应把跨模型总行数算成单模型行数。
            self.assertEqual(accounting["largest_model_rows"], len(runner.Y_all))
            self.assertLess(accounting["retained_array_bytes"], accounting["logical_design_bytes"])
            for index, origin in enumerate(runner.supervised_origins):
                expected, labels = runner.builder.training_row(origin)
                for actual, want in zip(runner.X_all, expected):
                    np.testing.assert_allclose(actual[index], want[0], rtol=1e-10, atol=1e-10)
                np.testing.assert_array_equal(runner.Y_all[index], labels)
            # Native estimator execution must accept the compact representation.
            with patch.object(IndexedDesign, "__array__", side_effect=AssertionError("premature dense conversion")):
                fitted = runner.fit(tuple(range(len(runner.supervised_origins))))
            designs, provider = runner.forecast_designs(runner.origin, fitted[0], fitted[1])
            prediction = runner.predict(fitted[-1], designs, provider, runner.forecast_times(runner.origin), fitted[1])
            self.assertEqual(prediction.values.shape, (1, 2, 1))
            self.assertTrue(np.isfinite(prediction.values).all())
