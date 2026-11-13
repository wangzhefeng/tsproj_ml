"""三模式公共生命周期：真实临时数据拟合，不读取正式模型产物。"""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_loading import SourceRegistry
from forecasting_core.specs.config import parse_model_config
from model_forecasting.deployment import predict_strategy_bundle
from model_pipeline.lifecycle import run_lifecycle, execute_lifecycle
from model_pipeline.production import ProductionForecastRequest
from model_pipeline.runner import CanonicalBaseModelRunner, run_canonical_config
from models.pickle_io import ModelDeployPkl
from tests.test_raw_history_window import make_config
from tests.test_calendar_month_origin_day import _payload
from tests import test_canonical_runtime_smoke as smoke


class LifecycleModesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "target.csv"
        self.times = pd.date_range("2026-01-01", periods=52, freq="1h")
        pd.DataFrame({"time": self.times, "load": 100 + np.arange(52) ** 1.5}).to_csv(self.path, index=False)
        self.config = make_config(self.path)

    def test_skip_backtest_predicts_with_insufficient_fold_history(self):
        config = replace(self.config, validation={**dict(self.config.validation), "fold_count": 50})
        with self.assertRaisesRegex(ValueError, "fold_count|insufficient history"):
            run_canonical_config(config, output_root=self.root / "full")
        with patch("model_pipeline.lifecycle.run_fixed_step_backtest", side_effect=AssertionError("backtest entered")), patch(
            "model_pipeline.lifecycle.run_calendar_month_backtest", side_effect=AssertionError("backtest entered")
        ):
            result = run_canonical_config(config, output_root=self.root / "production", skip_backtest=True)
        prediction = pd.read_csv(result.forecast_dir / "prediction.csv")
        self.assertEqual(len(prediction), config.problem.horizon)
        self.assertTrue(np.isfinite(prediction.predict_value).all())
        self.assertFalse(result.test_dir.exists())
        runtime = json.loads((result.forecast_dir / "resolved_config.json").read_text())["runtime"]
        self.assertIsNone(runtime["holdout"])
        self.assertIsNone(runtime["calibration"])
        self.assertEqual(runtime["holdout_visibility_proof"]["total_lookups"], 0)
        self.assertEqual(runtime["resources"]["stage_wall_seconds"]["backtest"], 0.0)
        self.assertEqual(result.fingerprint, config.fingerprint())

    def calendar_config(self):
        path = self.root / "daily.csv"
        times = pd.date_range("2025-10-01", "2026-07-31", freq="1D")
        pd.DataFrame({"time": times, "load": 100 + np.arange(len(times)) * 0.3}).to_csv(path, index=False)
        payload = _payload(str(path), fold_count=2)
        payload["problem"]["horizon"] = 31
        return parse_model_config(payload, source="<lifecycle-modes>")

    def test_same_final_prediction_and_saved_bundle_roundtrip(self):
        for kind, config in (("fixed", self.config), ("calendar", self.calendar_config())):
            with self.subTest(kind=kind):
                original = CanonicalBaseModelRunner.predict
                captured = []
                def record(runner, *args, **kwargs):
                    prediction = original(runner, *args, **kwargs)
                    captured.append(prediction.values.copy())
                    return prediction
                with patch.object(CanonicalBaseModelRunner, "predict", record):
                    full = run_canonical_config(config, output_root=self.root / kind / "full")
                    full_final = captured[-1]
                    skipped = run_canonical_config(config, output_root=self.root / kind / "skip", skip_backtest=True)
                    np.testing.assert_array_equal(full_final, captured[-1])
                pd.testing.assert_frame_equal(pd.read_csv(full.forecast_dir / "prediction.csv"),
                                              pd.read_csv(skipped.forecast_dir / "prediction.csv"))
                loaded = ModelDeployPkl(str(skipped.model_dir / "model.pkl")).load_model()
                origin = pd.Timestamp(config.validation["forecast_origin"])
                runner = CanonicalBaseModelRunner(config, SourceRegistry(config.data, self.root), origin)
                designs, provider = runner.builder.forecast_designs(origin, target_transform=loaded.target_transform)
                deployed = predict_strategy_bundle(loaded, designs[0], forecast_times=runner.forecast_times(origin),
                                                   raw_feature_provider=provider)
                np.testing.assert_array_equal(deployed.values, full_final)
                self.assertEqual(loaded.config_fingerprint, full.fingerprint)

    def test_no_bundle_preserves_old_files_and_uses_new_in_memory_fit(self):
        for skip in (False, True):
            with self.subTest(skip=skip):
                root = self.root / str(skip)
                original = run_canonical_config(self.config, output_root=root)
                # 保留明确损坏的旧模型：若尝试读旧模型必须失败，不能靠删除掩盖。
                old_model = original.model_dir / "model.pkl"
                old_model.write_bytes(b"stale model must never be loaded")
                before_model = {p.name: p.read_bytes() for p in original.model_dir.iterdir() if p.is_file()}
                before_test = {str(p.relative_to(original.test_dir)): p.read_bytes()
                               for p in original.test_dir.rglob("*") if p.is_file()}
                with patch.object(ModelDeployPkl, "load_model", side_effect=AssertionError("old model read")):
                    result = run_canonical_config(self.config, output_root=root, skip_backtest=skip, persist_bundle=False)
                self.assertEqual(before_model, {p.name: p.read_bytes() for p in result.model_dir.iterdir() if p.is_file()})
                if skip:
                    self.assertEqual(before_test, {str(p.relative_to(result.test_dir)): p.read_bytes()
                                                   for p in result.test_dir.rglob("*") if p.is_file()})
                self.assertEqual(json.loads((result.forecast_dir / "run_state.json").read_text())["status"], "completed")
                runtime = json.loads((result.forecast_dir / "resolved_config.json").read_text())["runtime"]
                self.assertFalse(runtime["persist_bundle"])
                if skip:
                    self.assertIsNone(runtime["holdout"])
                clean = run_canonical_config(self.config, output_root=root / "clean", skip_backtest=skip, persist_bundle=False)
                self.assertFalse(clean.model_dir.exists())
                pd.testing.assert_frame_equal(pd.read_csv(result.forecast_dir / "prediction.csv"),
                                              pd.read_csv(clean.forecast_dir / "prediction.csv"))

    def test_dynamic_origin_all_stages_and_fixed_step_not_month_sliced(self):
        for kind, config in (("fixed", self.config), ("calendar", self.calendar_config())):
            date = "2026-01-01" if kind == "fixed" else "2026-07-10"
            # 固定锚点故意过早：动态运行必须忽略它，也不按 issue_date 截断源数据。
            config = replace(config, validation={**dict(config.validation), "forecast_origin": "2025-01-01"})
            fingerprint = config.fingerprint()
            request = ProductionForecastRequest(date)
            for mode in ("backtest-only", "backtest-fit-forecast", "production"):
                with self.subTest(kind=kind, mode=mode):
                    result = run_canonical_config(config, output_root=self.root / kind / mode,
                        production_request=request, backtest_only=mode == "backtest-only", skip_backtest=mode == "production")
                    self.assertEqual(result.fingerprint, fingerprint)
                    if mode == "backtest-only":
                        self.assertTrue((result.test_dir / "test_scores_df.csv").exists())
                        self.assertEqual(list((self.root / kind / mode).rglob("model.pkl")), [])
                        self.assertEqual(list((self.root / kind / mode).rglob("prediction.csv")), [])
                        continue
                    runtime = json.loads((result.forecast_dir / "resolved_config.json").read_text())["runtime"]
                    expected_origin = self.times[-1] if kind == "fixed" else pd.Timestamp("2026-07-31")
                    self.assertEqual(pd.Timestamp(runtime["forecast_origin"]), expected_origin)
                    prediction = pd.read_csv(result.forecast_dir / "prediction.csv", parse_dates=["time"])
                    self.assertEqual(len(prediction), 2 if kind == "fixed" else 31)
                    self.assertEqual(prediction.time.iloc[0], expected_origin + pd.tseries.frequencies.to_offset(config.problem.freq))
                    if kind == "calendar":
                        self.assertEqual(prediction.time.iloc[-1], pd.Timestamp("2026-08-31"))
                    self.assertEqual(result.forecast_dir.parent.name, request.run_scope)
                    self.assertEqual(runtime["production"]["issue_date"], date)
                    self.assertTrue(runtime["production"]["sources"][0]["sha256"])

    def test_calendar_skip_does_not_require_backtest_folds(self):
        config = self.calendar_config()
        # 自然月既有合同允许少折；120天只够final训练，不够一个完整回测月。
        path = Path(config.data.sources[0].history_path)
        pd.read_csv(path).iloc[-120:].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "at least one complete fold"):
            run_canonical_config(config, output_root=self.root / "full-month")
        result = run_canonical_config(config, output_root=self.root / "skip-month", skip_backtest=True)
        self.assertEqual(len(pd.read_csv(result.forecast_dir / "prediction.csv")), 31)
        self.assertFalse(result.test_dir.exists())

    def test_training_history_and_incomplete_month_still_raise(self):
        too_short = replace(self.config, validation={**dict(self.config.validation),
                            "training_window": {"kind": "rolling", "history_steps": 100}})
        with self.assertRaisesRegex(ValueError, "history|coverage"):
            run_canonical_config(too_short, output_root=self.root / "short", skip_backtest=True)
        config = self.calendar_config()
        short_horizon = replace(config, problem=replace(config.problem, horizon=28))
        with self.assertRaisesRegex(ValueError, "cover|month"):
            run_canonical_config(short_horizon, output_root=self.root / "short-month", skip_backtest=True)
        states = list((self.root / "short-month").rglob("run_state.json"))
        self.assertEqual(len(states), 1)
        self.assertEqual(json.loads(states[0].read_text())["status"], "failed")

    def test_public_api_guards_precede_training_and_writes(self):
        prob = smoke.CanonicalRuntimeSmokeTest().build_config(self.path, mode="quantile")
        prob = replace(prob, probabilistic={**dict(prob.probabilistic), "calibration": {
            "method": "cqr", "interval": "q10_q90", "target_coverage": 0.8,
            "calibration_windows": 3, "min_windows": 1, "min_scores": 1,
            "label_availability_delay_steps": 0}})
        seasonal = replace(self.config, features=replace(self.config.features, transformations={
            **self.config.features.canonical_payload()["transformations"],
            "seasonal_baseline": {"column": "load", "period": 2, "days": 2}}))
        # 在配置入口/runner.run/两个生命周期公共入口逐一验证，禁止旁路。
        from types import SimpleNamespace
        ets = replace(self.config,
            strategy=replace(self.config.strategy, name="mimo"),
            features=replace(self.config.features, target_lags={}, datetime_features=(), transformations={}),
            estimator=replace(self.config.estimator, model_type="ets", params={"seasonal_periods": 2, "candidates": ["ANN"]}))
        cases = [(self.config, {"backtest_only": True, "skip_backtest": True}, "cannot be combined"),
                 (self.config, {"backtest_only": True, "persist_bundle": False}, "requires.*final fit"),
                 (prob, {"skip_backtest": True}, "CQR"),
                 (seasonal, {"persist_bundle": False}, "seasonal_baseline"),
                 (ets, {}, "ETS")]
        for config, options, message in cases:
            with self.subTest(message=message):
                fake = SimpleNamespace(config=config, execution_plan=SimpleNamespace(model_threads=1))
                calls = [lambda: run_lifecycle(fake, self.root / "guard", **options),
                         lambda: execute_lifecycle(fake, self.root / "guard", **options),
                         lambda: CanonicalBaseModelRunner.run(fake, self.root / "guard", **options)]
                calls.append(lambda: run_canonical_config(config, output_root=self.root / "guard", **options))
                for call in calls:
                    with self.assertRaisesRegex(ValueError, message):
                        call()
                    self.assertFalse((self.root / "guard").exists())

    def test_execution_failure_records_failed_without_bundle(self):
        error = RuntimeError("injected final-fit failure")
        with patch.object(CanonicalBaseModelRunner, "fit_final", side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                run_canonical_config(self.config, output_root=self.root / "failure", skip_backtest=True, persist_bundle=False)
        self.assertIs(caught.exception, error)
        states = list((self.root / "failure").rglob("run_state.json"))
        self.assertEqual(len(states), 1)
        self.assertEqual(json.loads(states[0].read_text())["status"], "failed")
        self.assertEqual(list((self.root / "failure").rglob("model.pkl")), [])
