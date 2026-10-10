"""发报时间、目标区间和训练窗口的小数据端到端契约。"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_loading import SourceRegistry
from feature_engineering.design_identity import compute_raw_design_fingerprint
from model_pipeline.batch_artifacts import artifact_paths, validate_artifacts
from model_pipeline.runner import CanonicalBaseModelRunner
from tests.test_raw_history_window import make_config
import test_canonical_runtime_smoke as smoke


class ForecastScheduleWindowsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "target.csv"
        self.times = pd.date_range("2026-01-01", periods=360, freq="1h")
        self.frame = pd.DataFrame({"time": self.times, "load": 100 + np.sin(np.arange(360) / 12)})
        self.frame.to_csv(self.path, index=False)
        base = make_config(self.path, strategy="direct")
        base = replace(base, features=replace(base.features, transformations={
            **base.features.canonical_payload()["transformations"],
            "direct": {"layout": "single_model_horizon", "align_to_target": False,
                       "horizon_feature": {"enabled": True, "name": "h", "cyclical": False}}}))
        self.base = replace(base, problem=replace(base.problem, horizon=48))

    def config(self, kind="rolling", start="next_day"):
        validation = {k: v for k, v in dict(self.base.validation).items()
                      if k not in {"train_history_steps", "train_window_steps", "seasonal_naive_lag"}}
        validation.update(forecast_origin=self.times[302].isoformat(),
                          fold_count=2, stride_steps=24,
                          forecast_window={"start": start},
                          training_window=({"kind": "rolling", "history_steps": 120} if kind == "rolling"
                                           else {"kind": "expanding", "start_time": self.times[0].isoformat()}),
                          training={"origin_sampling": {"time_of_day": "14:00"}})
        return replace(self.base, validation=validation)

    def runner(self, config, origin=None):
        return CanonicalBaseModelRunner(config, SourceRegistry(config.data, self.root),
                                        self.times[302] if origin is None else origin)

    def test_next_day_multi_day_times_and_safe_sample_labels(self):
        runner = self.runner(self.config())
        times = runner.forecast_times(runner.origin)
        self.assertEqual(times[0], pd.Timestamp("2026-01-14 00:00"))
        self.assertEqual(times[-1], pd.Timestamp("2026-01-15 23:00"))
        self.assertEqual(len(times), 48)
        self.assertTrue(all(o.hour == 14 for o in runner.supervised_origins))
        self.assertTrue(all(runner.forecast_times(o)[-1] <= runner.origin for o in runner.supervised_origins))
        np.testing.assert_array_equal(runner.Y_all[-1, :, 0],
            pd.read_csv(self.path, parse_dates=["time"]).set_index("time").loc[
                runner.forecast_times(runner.supervised_origins[-1]), "load"].to_numpy())

    def test_recent_complete_folds_without_search_cap(self):
        # 独立给出黄金日期：截止1月13日14:00，次日两天目标需后退三天。
        cases = (
            ({"start": "next_day"}, 48, ("2026-01-09 14:00", "2026-01-10 14:00")),
            ({"start": "after_origin"}, 48, ("2026-01-10 14:00", "2026-01-11 14:00")),
            ({"start": "after_origin", "gap_steps": 2}, 4,
             ("2026-01-11 14:00", "2026-01-12 14:00")),
        )
        for forecast, horizon, expected in cases:
            with self.subTest(forecast=forecast):
                config = self.config()
                config = replace(config, problem=replace(config.problem, horizon=horizon),
                    validation={**dict(config.validation), "forecast_window": forecast})
                windows = self.runner(config).backtest_windows()
                self.assertEqual([w.origin for w in windows], [pd.Timestamp(t) for t in expected])
                self.assertTrue(all(w.metadata['raw_history_steps'] == 120 for w in windows))

    def test_insufficient_history_does_not_shrink_folds_or_training_window(self):
        config = self.config()
        for count, message in ((20, 'requested fold_count'), (10, 'insufficient history')):
            with self.subTest(count=count):
                changed = replace(config, validation={**dict(config.validation), 'fold_count': count})
                with self.assertRaisesRegex(ValueError, message):
                    self.runner(changed).backtest_windows()

    def test_rolling_and_expanding_refit_overlap_and_final_bundle(self):
        for kind in ("rolling", "expanding"):
            with self.subTest(kind=kind):
                config = self.config(kind)
                runner = self.runner(config)
                windows = runner.backtest_windows()
                starts = [w.metadata["raw_history_start"] for w in windows]
                self.assertEqual(starts[0] == starts[1], kind == "expanding")
                counts = [len(w.train_indices) for w in windows]
                self.assertEqual(counts[0] == counts[1], kind == "rolling")
                fitted = []
                original = CanonicalBaseModelRunner.fit
                def traced(context, indices, **kwargs):
                    result = original(context, indices, **kwargs)
                    fitted.append((context.origin, result[-1]))
                    return result
                with patch.object(CanonicalBaseModelRunner, "fit", traced):
                    result = runner.run(self.root / kind)
                self.assertEqual([x[0] for x in fitted], [w.origin for w in windows])
                self.assertIsNot(fitted[0][1], fitted[1][1])
                frame = pd.read_csv(result.test_dir / "cv_plot_df.csv")
                self.assertEqual(len(frame), 96)
                self.assertTrue(frame.time.duplicated().any())
                self.assertIn("forecast_origin", frame)
                self.assertIn("lead_steps", frame)
                self.assertTrue(list(result.model_dir.rglob("*.pkl")))
                validate_artifacts({"artifacts": artifact_paths(result),
                    "config_fingerprint": config.fingerprint(), "result_identity": result.model_dir.parent.name},
                    require_digests=False)
                # 同一历史时刻独立进行生产式final fit，必须与该折一致。
                standalone = self.runner(config, windows[0].origin)
                inputs = standalone.final_bundle_inputs()
                _, artifact, _ = standalone.fit_final(inputs[2], inputs[3])
                designs, provider = standalone.forecast_designs(standalone.origin, inputs[0], inputs[1])
                prediction = standalone.predict(artifact, designs, provider,
                    standalone.forecast_times(standalone.origin), inputs[1])
                np.testing.assert_allclose(prediction.values.reshape(-1),
                    frame.loc[frame.window == 1, "predict_value"].to_numpy(), rtol=0, atol=1e-12)

    def test_sampling_and_time_windows_have_distinct_raw_cache_keys(self):
        config = self.config()
        variants = [config, self.config("expanding"), self.config(start="after_origin"),
            replace(config, validation={**dict(config.validation),
                "training": {"origin_sampling": {"stride_steps": 12}}})]
        keys = [compute_raw_design_fingerprint(c, base_dir=self.root, origin=self.times[302], generators={})
                for c in variants]
        self.assertEqual(len(set(keys)), len(variants))

    def test_future_perturbation_does_not_change_training_or_forecast(self):
        config = self.config()
        def predict():
            runner = self.runner(config)
            inputs = runner.final_bundle_inputs()
            _, artifact, _ = runner.fit_final(inputs[2], inputs[3])
            designs, provider = runner.forecast_designs(runner.origin, inputs[0], inputs[1])
            return inputs[3], runner.predict(artifact, designs, provider,
                runner.forecast_times(runner.origin), inputs[1]).values
        before = predict()
        self.frame.loc[303:, "load"] += 100000
        self.frame.to_csv(self.path, index=False)
        after = predict()
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)

    def test_after_origin_gap_and_sparse_stride_are_not_resampled_twice(self):
        config = self.config(start="after_origin")
        config = replace(config, problem=replace(config.problem, horizon=4),
            validation={**dict(config.validation), "forecast_window": {"start": "after_origin", "gap_steps": 2},
                        "training": {"origin_sampling": {"stride_steps": 12, "anchor_time": "2026-01-13T14:00:00"}}})
        runner = self.runner(config)
        self.assertEqual(runner.forecast_times(runner.origin)[0], pd.Timestamp("2026-01-13 17:00"))
        self.assertEqual(runner.forecast_times(runner.origin)[-1], pd.Timestamp("2026-01-13 20:00"))
        fit = runner.fit(tuple(range(len(runner.supervised_origins))))
        self.assertEqual(len(fit[3]), len(runner.supervised_origins))
        self.assertTrue(all(o.hour in (2, 14) for o in runner.supervised_origins))
        self.assertTrue(all(b - a == pd.Timedelta(hours=12)
                            for a, b in zip(runner.supervised_origins, runner.supervised_origins[1:])))
        contiguous = replace(config, validation={**dict(config.validation),
            "forecast_window": {"start": "after_origin"}})
        indexed = self.runner(contiguous)
        indexed.prepare_training()
        designs, labels = indexed.builder.training_row(indexed.supervised_origins[0])
        self.assertEqual(indexed.training_compile["mode"], "indexed")
        for actual, expected in zip(indexed.X_all, designs):
            np.testing.assert_allclose(actual[0:1], expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(indexed.Y_all[0], labels)

    def test_invalid_temporal_contracts_fail_before_io(self):
        config = self.config()
        for changes in ({"forecast_window": None}, {"forecast_window": {"start": "wrong"}},
                        {"forecast_window": {"start": "next_day", "gap_steps": 2}},
                        {"training_window": {"kind": "rolling", "history_steps": True}},
                        {"training_window": {"kind": "expanding"}},
                        {"training_window": {"kind": "expanding", "start_time": "bad"}},
                        {"train_window_steps": 3}, {"refit_every": 0}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                replace(config, validation={**dict(config.validation), **changes})


class NativeHistoryTrainingWindowTest(unittest.TestCase):
    """ETS native_history + rolling training_window 合同（旧字段退役 P0-1）。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "target.csv"
        self.times = pd.date_range("2026-01-01", periods=360, freq="1h")
        pd.DataFrame({"time": self.times,
                      "load": 100 + np.sin(np.arange(360) * 2 * np.pi / 24)}
                     ).to_csv(self.path, index=False)

    def config(self):
        base = smoke.CanonicalRuntimeSmokeTest().build_config(
            self.path, mode="point", strategy="mimo", horizon=24)
        validation = {k: v for k, v in dict(base.validation).items()
                      if k not in {"train_history_steps", "train_window_steps",
                                   "seasonal_naive_lag"}}
        validation.update(forecast_origin=self.times[302].isoformat(),
                          fold_count=2, stride_steps=24,
                          training_window={"kind": "rolling", "history_steps": 120})
        # 单次 replace：ETS 合同校验在构造期执行，validation 必须同批进入。
        return replace(base,
            estimator=replace(base.estimator, model_type="ets",
                              params={"seasonal_periods": 24, "candidates": ["ANN"]}),
            features=replace(base.features, target_lags={}, observed_past_lags={},
                             datetime_features=[], transformations={}),
            validation=validation)

    def test_ets_rolling_training_window_bounds_fold_history(self):
        config = self.config()
        runner = CanonicalBaseModelRunner(
            config, SourceRegistry(config.data, self.root), self.times[302])
        windows = runner.backtest_windows()
        self.assertEqual(len(windows), 2)
        # 各折 raw 历史起点 = origin 前含 origin 恰好 120 点（rolling 有界窗）。
        for window in windows:
            expected_start = window.origin - pd.Timedelta(hours=119)
            self.assertEqual(window.metadata["raw_history_start"],
                             expected_start.isoformat())
        from models.wrappers.ets import ETSModel
        from model_testing.fixed_step import run_fixed_step_backtest
        histories = []
        original = ETSModel.fit_history
        def traced(model, history, **kwargs):
            histories.append((len(history), kwargs.get("as_of")))
            return original(model, history, **kwargs)
        with patch.object(ETSModel, "fit_history", traced):
            with tempfile.TemporaryDirectory() as directory:
                run_fixed_step_backtest(runner, Path(directory), mode="point")
        self.assertEqual([h[0] for h in histories], [120, 120])
        self.assertEqual([h[1] for h in histories], [w.origin for w in windows])


class QuantileTrainingWindowTest(unittest.TestCase):
    """quantile + rolling training_window 合同（旧字段退役 P0-4）。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "target.csv"
        self.times = pd.date_range("2026-01-01", periods=360, freq="1h")
        pd.DataFrame({"time": self.times,
                      "load": 100 + np.sin(np.arange(360) * 2 * np.pi / 24)}
                     ).to_csv(self.path, index=False)

    def config(self, strategy="direct"):
        base = smoke.CanonicalRuntimeSmokeTest().build_config(
            self.path, mode="quantile", strategy=strategy, horizon=24)
        validation = {k: v for k, v in dict(base.validation).items()
                      if k not in {"train_history_steps", "train_window_steps",
                                   "seasonal_naive_lag"}}
        validation.update(forecast_origin=self.times[302].isoformat(),
                          fold_count=2, stride_steps=24,
                          training_window={"kind": "rolling", "history_steps": 120})
        return replace(base, validation=validation)

    def test_quantile_direct_rolling_training_window_runs_backtest(self):
        config = self.config("direct")
        runner = CanonicalBaseModelRunner(
            config, SourceRegistry(config.data, self.root), self.times[302])
        windows = runner.backtest_windows()
        self.assertEqual(len(windows), 2)
        with tempfile.TemporaryDirectory() as directory:
            _, _, audits = __import__(
                "model_testing.fixed_step", fromlist=["run_fixed_step_backtest"]
            ).run_fixed_step_backtest(runner, Path(directory), mode="quantile")
        self.assertEqual(len(audits), 2)

    def test_quantile_recursive_median_path_runs_under_training_window(self):
        # median_path：递归依赖经 point(中位)水平逐级喂特征（predictor.py:169-217），
        # 在 rolling 有界窗下逐折重拟合后仍应产出有限分位张量。
        from forecasting_core.artifacts import MarginalForecastDistribution
        config = self.config("recursive")
        runner = CanonicalBaseModelRunner(
            config, SourceRegistry(config.data, self.root), self.times[302])
        windows = runner.backtest_windows()
        self.assertEqual(len(windows), 2)
        fold_runner = runner.for_backtest_window(windows[0])
        _, _, _, _, artifact = fold_runner.fit(windows[0].train_indices)
        inputs = fold_runner.final_bundle_inputs()
        designs, provider = fold_runner.forecast_designs(
            fold_runner.origin, inputs[0], inputs[1])
        prediction = fold_runner.predict(
            artifact, designs, provider, fold_runner.forecast_times(fold_runner.origin),
            inputs[1])
        self.assertIsInstance(prediction, MarginalForecastDistribution)
        quantile_values = prediction.quantiles.values
        self.assertTrue(np.isfinite(quantile_values).all())
        # 分位单调性：median_preserving_isotonic 交叉修复后 q10 <= q90。
        levels = list(prediction.quantiles.levels)
        q10 = quantile_values[..., levels.index(0.1)]
        q90 = quantile_values[..., levels.index(0.9)]
        self.assertTrue((q10 <= q90 + 1e-9).all())

    def test_shifted_forecast_window_with_quantile_is_rejected(self):
        config = self.config("direct")
        with self.assertRaisesRegex(ValueError, "shifted forecast_window with quantile"):
            replace(config, validation={**dict(config.validation),
                "forecast_window": {"start": "next_day"}})


class TargetTransformTrainingWindowTest(unittest.TestCase):
    """target transform + rolling training_window 合同（旧字段退役 D9）。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "target.csv"
        self.times = pd.date_range("2026-01-01", periods=360, freq="1h")
        pd.DataFrame({"time": self.times,
                      "load": 100 + np.sin(np.arange(360) * 2 * np.pi / 24)}
                     ).to_csv(self.path, index=False)

    def config(self):
        base = smoke.CanonicalRuntimeSmokeTest().build_config(
            self.path, mode="point", strategy="direct", horizon=24)
        validation = {k: v for k, v in dict(base.validation).items()
                      if k not in {"train_history_steps", "train_window_steps",
                                   "seasonal_naive_lag"}}
        validation.update(forecast_origin=self.times[302].isoformat(),
                          fold_count=2, stride_steps=24,
                          training_window={"kind": "rolling", "history_steps": 120})
        # 与现役 decomp 配置同形的 target transform（scaling 用 standard 以覆盖状态拟合）
        return replace(base,
            features=replace(base.features, transformations={
                **base.features.canonical_payload()["transformations"],
                "target": {
                    "calendar_normalization": {"method": "none"},
                    "decomposition": {"method": "linear", "trend_degree": 1,
                                      "trend_forecast": "polynomial", "damping": 0.98,
                                      "trend_lookback": 4},
                    "scaling": {"method": "standard", "inverse": True},
                },
            }),
            validation=validation)

    def _fold1_prediction(self):
        config = self.config()
        runner = CanonicalBaseModelRunner(
            config, SourceRegistry(config.data, self.root), self.times[302])
        windows = runner.backtest_windows()
        fold_runner = runner.for_backtest_window(windows[0])
        _, _, _, _, artifact = fold_runner.fit(windows[0].train_indices)
        inputs = fold_runner.final_bundle_inputs()
        designs, provider = fold_runner.forecast_designs(
            fold_runner.origin, inputs[0], inputs[1])
        prediction = fold_runner.predict(
            artifact, designs, provider, fold_runner.forecast_times(fold_runner.origin),
            inputs[1])
        return windows[0], prediction.values.reshape(-1)

    def test_target_transform_runs_under_training_window(self):
        windows_first, prediction = self._fold1_prediction()
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(len(prediction), 24)

    def test_transform_state_is_bounded_by_window(self):
        # 泄漏探针：污染窗口起点之前的历史，折 1 的变换态与预测必须不变。
        windows_first, before = self._fold1_prediction()
        start = pd.Timestamp(windows_first.metadata["raw_history_start"])
        frame = pd.read_csv(self.path, parse_dates=["time"])
        frame.loc[frame.time < start, "load"] += 1e6
        frame.to_csv(self.path, index=False)
        _, after = self._fold1_prediction()
        np.testing.assert_array_equal(before, after)


class MonthlyTrainingWindowTest(unittest.TestCase):
    """月度频率（1ME）+ rolling training_window 合同（旧字段退役 D7）。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "target.csv"
        self.times = pd.date_range("2023-01-31", periods=48, freq="1ME")
        pd.DataFrame({"time": self.times,
                      "load": 100 + np.arange(48, dtype=float) * 2}
                     ).to_csv(self.path, index=False)

    def config(self):
        base = smoke.CanonicalRuntimeSmokeTest().build_config(
            self.path, mode="point", strategy="direct", horizon=3)
        features = {
            "target_lags": {"load": (1, 2, 3, 4, 5, 6)},
            "observed_past_lags": {},
            "datetime_features": (),
            "transformations": {"direct": {"layout": "independent_models",
                                           "align_to_target": False}},
            "selection": None,
        }
        return replace(base,
            problem=replace(base.problem, freq="1ME", horizon=3),
            features=replace(base.features, **features),
            validation={"forecast_origin": self.times[-1].isoformat(),
                        "fold_count": 2, "stride_steps": 2,
                        "training_window": {"kind": "rolling", "history_steps": 24}})

    def test_monthly_rolling_training_window_builds_folds(self):
        config = self.config()
        runner = CanonicalBaseModelRunner(
            config, SourceRegistry(config.data, self.root), self.times[-1])
        windows = runner.backtest_windows()
        self.assertEqual(len(windows), 2)
        for window in windows:
            self.assertLess(window.origin, self.times[-1])
            self.assertGreater(len(window.train_indices), 0)
