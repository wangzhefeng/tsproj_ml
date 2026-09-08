"""严格原始历史窗口：先截断再编译；合成数据真实拟合与隔离验证。"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_loading import InformationSetRequest, SourceRegistry
from forecasting_core.specs.validation import RuntimeValidationSpec
from forecasting_core.specs import ColumnSpec
from model_training.estimators.capabilities import _ModelFactoryEstimator
from model_pipeline.runner import CanonicalBaseModelRunner, run_canonical_config
from model_pipeline.supervised_design import minimum_history_rows
from tests import test_canonical_runtime_smoke as smoke


def make_config(path, *, strategy="recursive", workers=1):
    base = smoke.CanonicalRuntimeSmokeTest().build_config(path, mode="point", strategy=strategy)
    features = base.features.canonical_payload()
    features["transformations"]["advanced"] = {
        "rolling": {"columns": ["load"], "windows": [3], "stats": ["mean"]},
        "expanding": {"columns": ["load"], "stats": ["mean", "std"]},
    }
    base = replace(base, features=replace(base.features, transformations=features["transformations"]))
    return replace(base, validation={
        "forecast_origin": "2026-01-02T23:00:00", "schedule_mode": "intraday",
        "history_steps": 44, "train_history_steps": 20,
        "train_window_steps": 20 - minimum_history_rows(base) - 2 + 1,
        "fold_count": 3, "stride_steps": 2, "seasonal_naive_lag": 2,
        "performance": {"window_parallel_workers": workers, "total_thread_limit": 2},
    })


class RawHistoryWindowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "target.csv"
        self.times = pd.date_range("2026-01-01", periods=52, freq="1h")
        self.frame = pd.DataFrame({"time": self.times, "load": 100 + np.arange(52, dtype=float) ** 1.5})
        self.frame.to_csv(self.path, index=False)

    def runner(self, config=None, origin=None, **kwargs):
        config = config or make_config(self.path)
        return CanonicalBaseModelRunner(config, SourceRegistry(config.data, self.root),
                                        self.times[47] if origin is None else origin, **kwargs)

    def test_request_boundary_is_immutable_and_preserves_upper_visibility(self):
        config = smoke.CanonicalRuntimeSmokeTest().build_config(self.path, mode="point")
        request = InformationSetRequest(self.times[30], self.times[31:33], (), history_start=self.times[20])
        registry = SourceRegistry(config.data, self.root)
        history = registry.materialize(request).target_history["target_history"]
        self.assertEqual(list(history.time), list(self.times[20:31]))
        labels = registry.materialize(replace(request, target_access="supervised_labels")).target_history["target_history"]
        self.assertEqual(list(labels.time), list(self.times[20:33]))
        with self.assertRaises(AttributeError):
            request.history_start = self.times[0]
        for invalid in (pd.NaT, self.times[31]):
            with self.assertRaises(ValueError):
                InformationSetRequest(self.times[30], self.times[31:33], (), history_start=invalid)
        self.frame.assign(temperature=12.0).to_csv(self.path, index=False)
        source = replace(config.data.sources[0], columns=(*config.data.sources[0].columns,
                         ColumnSpec("temperature", "observed_past")), provider="persistence")
        observed_registry = SourceRegistry(replace(config.data, sources=(source,)), self.root)
        observed = observed_registry.materialize(request).observed_past["target_history"]
        self.assertEqual(list(observed.time), list(self.times[20:31]))

    def test_contract_rejects_invalid_or_incomplete_history_geometry(self):
        config = make_config(self.path)
        for value in (True, 0, -2, 2.5, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(config, validation={**dict(config.validation), "train_history_steps": value})
        with self.assertRaises(ValueError):
            RuntimeValidationSpec.from_mapping({"train_history_steps": 20})
        with self.assertRaises(ValueError):
            RuntimeValidationSpec.from_mapping({"horizon_mode": "calendar_month", "train_window_days": 20,
                                                "fold_count": 2, "stride_months": 1, "train_history_steps": 20})
        for changes in ({"train_window_steps": 2}, {"train_history_steps": 4}):
            with self.assertRaisesRegex(ValueError, "train_window_steps|train_history_steps"):
                self.runner(replace(config, validation={**dict(config.validation), **changes}))
        broken = self.frame.drop(index=35)
        broken.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "regular|grid|history"):
            self.runner(config)

    def test_identity_is_semantic_and_unsupported_modes_raise(self):
        config = make_config(self.path)
        plain = {k: v for k, v in dict(config.validation).items() if k != "train_history_steps"}
        baseline = replace(config, validation=plain)
        self.assertNotEqual(config.fingerprint(), baseline.fingerprint())
        self.assertEqual(baseline.fingerprint(), replace(baseline, validation=dict(baseline.validation)).fingerprint())
        with self.assertRaisesRegex(ValueError, "train_history_steps"):
            replace(config, probabilistic={"mode": "quantile", "quantiles": [0.1, 0.5, 0.9]})
        with self.assertRaisesRegex(ValueError, "train_history_steps"):
            replace(config, features=replace(config.features, transformations={"target": {"scaling": {"method": "standard"}}}))
        with patch("model_pipeline.runner.SourceRegistry", side_effect=AssertionError("must reject before source IO")):
            with self.assertRaisesRegex(ValueError, "backtest.only"):
                run_canonical_config(config)
        runner = self.runner(config)
        with self.assertRaisesRegex(ValueError, "backtest.only"):
            runner.final_bundle_inputs()

    def test_outside_window_perturbation_leaves_features_and_prediction_unchanged(self):
        config = make_config(self.path)
        baseline = self.runner(config)
        self.assertEqual(baseline.builder.history_start, self.times[28])
        n = config.validation.backtest.train_window_steps
        self.assertEqual(len(baseline.supervised_origins), n)
        fit = baseline.fit(tuple(range(n)))
        designs, provider = baseline.forecast_designs(baseline.origin, fit[0], fit[1])
        predicted = baseline.predict(fit[-1], designs, provider, baseline.forecast_times(baseline.origin), fit[1])
        changed = self.frame.copy()
        changed.loc[:27, "load"] += 1000000
        changed.loc[48:, "load"] += 1000000
        changed.to_csv(self.path, index=False)
        repeated = self.runner(config)
        for first, second in zip(baseline.X_all, repeated.X_all):
            np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(baseline.Y_all, repeated.Y_all)
        fit2 = repeated.fit(tuple(range(n)))
        designs2, provider2 = repeated.forecast_designs(repeated.origin, fit2[0], fit2[1])
        predicted2 = repeated.predict(fit2[-1], designs2, provider2, repeated.forecast_times(repeated.origin), fit2[1])
        np.testing.assert_array_equal(predicted.values, predicted2.values)
        changed.loc[28, "load"] += 10000
        changed.to_csv(self.path, index=False)
        inside = self.runner(config)
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(baseline.X_all, inside.X_all)))

    def test_expanding_resets_and_fold_contexts_do_not_share_state(self):
        runner = self.runner()
        windows = runner.backtest_windows()
        self.assertEqual(len(windows), 3)
        contexts = [runner.for_backtest_window(w) for w in windows]
        self.assertEqual(len({id(c.builder) for c in contexts}), 3)
        for context, window in zip(contexts, windows):
            start = window.origin - pd.Timedelta(hours=19)
            self.assertEqual(context.builder.history_start, start)
            self.assertEqual(window.metadata["raw_history_start"], start.isoformat())
            self.assertEqual(window.metadata["raw_history_end"], window.origin.isoformat())
            self.assertEqual(window.metadata["train_history_steps"], 20)
            self.assertEqual(context.geometry.label_end(context.supervised_origins[-1]), window.origin)
            mean_columns = [i for i, name in enumerate(context.feature_schema) if "expanding" in name and "mean" in name]
            self.assertEqual(len(mean_columns), 1, context.feature_schema)
            for row in (0, -1):
                origin = context.supervised_origins[row]
                expected = self.frame.loc[self.frame.time.between(start, origin), "load"].mean()
                self.assertAlmostEqual(float(context.X_all[0][row, mean_columns[0]]), expected)
        self.assertEqual(contexts[0].builder.history_start, windows[0].origin - pd.Timedelta(hours=19))

    def test_batch_and_single_share_the_same_history_lower_bound(self):
        config = make_config(self.path, strategy="direct")
        runner = self.runner(config)
        origins = runner.supervised_origins[:3]
        self.assertEqual(runner.training_compile["mode"], "batch")
        for index, origin in enumerate(origins):
            single, labels = runner.builder.training_row(origin)
            for call, design in enumerate(single):
                np.testing.assert_allclose(runner.X_all[call][index:index + 1], design, rtol=1e-12)
        changed = self.frame.copy()
        changed.loc[:27, "load"] += 1000000
        changed.to_csv(self.path, index=False)
        repeated = self.runner(config)
        for first, second in zip(runner.X_all, repeated.X_all):
            np.testing.assert_array_equal(first, second)

    def test_cache_and_checkpoint_are_bound_to_window(self):
        cache = self.root / "cache"
        checkpoint = self.root / "checkpoints"
        first = self.runner(compiled_cache_root=cache, checkpoint_root=checkpoint)
        second = self.runner(compiled_cache_root=cache, checkpoint_root=checkpoint)
        earlier = self.runner(origin=self.times[45], compiled_cache_root=cache, checkpoint_root=checkpoint)
        self.assertFalse(first.compiled_cache_hit)
        self.assertTrue(second.compiled_cache_hit)
        self.assertNotEqual(first.compiled_cache_fingerprint, earlier.compiled_cache_fingerprint)
        np.testing.assert_array_equal(first.X_all, second.X_all)
        indices = tuple(range(len(first.supervised_origins)))
        fit = first.fit(indices)
        with patch.object(_ModelFactoryEstimator, "fit", side_effect=AssertionError("checkpoint hit retrained")):
            hit = second.fit(indices)
        for context, result in ((first, fit), (second, hit)):
            designs, provider = context.forecast_designs(context.origin, result[0], result[1])
            prediction = context.predict(result[-1], designs, provider, context.forecast_times(context.origin), result[1])
            if context is first:
                expected = prediction.values
            else:
                np.testing.assert_array_equal(expected, prediction.values)

    def test_backtest_serial_parallel_scores_all_points_and_same_actual(self):
        frames = []
        for workers in (1, 2):
            config = make_config(self.path, workers=workers)
            original = CanonicalBaseModelRunner.for_backtest_window
            with patch.object(CanonicalBaseModelRunner, "for_backtest_window", autospec=True, side_effect=original) as factory:
                result = run_canonical_config(config, output_root=self.root / f"run{workers}", backtest_only=True)
            self.assertEqual(factory.call_count, 3)
            self.assertEqual(factory.call_args.args[0].execution_plan.window_workers, workers)
            frames.append(pd.read_csv(result.test_dir / "cv_plot_df.csv"))
            self.assertFalse(pd.read_csv(result.test_dir / "test_scores_df.csv").empty)
        pd.testing.assert_frame_equal(frames[0], frames[1])
        self.assertEqual(len(frames[0]), 6)
        self.assertFalse(frames[0].isna().any(axis=None))


if __name__ == "__main__":
    unittest.main()
