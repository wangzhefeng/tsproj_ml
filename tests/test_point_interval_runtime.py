"""点模型区间主链、持久化与部署验收。"""
from dataclasses import replace
from pathlib import Path
import pickle
import tempfile
import unittest
from typing import cast

import numpy as np
import pandas as pd

from data_loading import SourceRegistry
from forecasting_core.point_intervals import PointIntervalForecast
from model_predicting.loops.deployment import predict_strategy_bundle
from pipeline.batch_artifacts import artifact_paths, artifact_digests, validate_artifacts
from pipeline.runner import run_canonical_config
from pipeline.lifecycle import CanonicalRuntimeResult
from pipeline.supervised_design import SupervisedDesignBuilder
import test_canonical_runtime_smoke as fixtures


class PointIntervalRuntimeTest(unittest.TestCase):
    def test_real_backtest_final_bundle_and_deployment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "load.csv"
            pd.DataFrame({"time": pd.date_range("2026-01-01", periods=52, freq="h"),
                          "load": 100 + np.arange(52) + np.sin(np.arange(52))}).to_csv(source, index=False)
            base = fixtures.CanonicalRuntimeSmokeTest().build_config(source, mode="point", strategy="mimo")
            config = replace(base, validation={
                "forecast_origin": "2026-01-02T23:00:00", "history_steps": 35,
                "train_window_steps": 12, "fold_count": 6, "stride_steps": 2,
                "refit_every": 0,
            }, probabilistic={"mode": "point", "calibration": {
                "method": "absolute_residual", "target_coverage": 0.5,
                "calibration_windows": 4, "min_windows": 2, "min_scores": 2,
            }})
            result = run_canonical_config(config, output_root=root / "results")
            assert isinstance(result, CanonicalRuntimeResult)
            self.assertIsNotNone(result.bundle.calibration_state)
            assert result.bundle.calibration_state is not None
            self.assertEqual(result.bundle.calibration_state["method"], "absolute_residual")
            backtest = pd.read_csv(result.test_dir / "cv_plot_df.csv")
            self.assertFalse(backtest.loc[backtest["window"] <= 2, "pi_available"].any())
            self.assertTrue(backtest.loc[backtest["window"] >= 3, "pi_available"].all())
            prediction = pd.read_csv(result.forecast_dir / "prediction.csv")
            self.assertTrue(prediction["pi_available"].all())
            self.assertFalse(any(column.startswith("predict_q") for column in prediction))
            scores = pd.read_csv(result.test_dir / "test_scores_probabilistic_df.csv")
            self.assertTrue({"interval_coverage", "interval_width", "interval_winkler", "coverage_gap"} <= set(scores["metric"]))
            paths = artifact_paths(result)
            validate_artifacts({"artifacts": paths, "artifact_sha256": artifact_digests(paths),
                                "config_fingerprint": config.fingerprint(), "result_identity": config.result_identity()})
            with (result.model_dir / "model.pkl").open("rb") as handle:
                bundle = pickle.load(handle)
            builder = SupervisedDesignBuilder(config, SourceRegistry(config.data, root))
            origin = cast(pd.Timestamp, pd.Timestamp(config.validation["forecast_origin"]))
            request = builder.request(origin)
            info = builder.registry.materialize(request)
            compiled = builder.compiler.compile(info, request, horizon_steps=[1])
            design = compiled.frame[bundle.input_schema["columns"]].to_numpy(dtype=float)
            deployed = predict_strategy_bundle(bundle, design, forecast_times=request.forecast_times)
            self.assertIsInstance(deployed, PointIntervalForecast)
            assert isinstance(deployed, PointIntervalForecast)
            np.testing.assert_allclose(deployed.point.values.reshape(-1), prediction["predict_value"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(deployed.lower.reshape(-1), prediction["predict_pi50_lower"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(deployed.upper.reshape(-1), prediction["predict_pi50_upper"], rtol=0, atol=1e-12)
            # 不能通过伪造 available=true 放过 NaN 区间；结构校验独立于摘要。
            backtest.loc[0, "pi_available"] = True
            backtest.to_csv(result.test_dir / "cv_plot_df.csv", index=False)
            with self.assertRaisesRegex(ValueError, "bounds/availability"):
                validate_artifacts({"artifacts": paths, "config_fingerprint": config.fingerprint(),
                                    "result_identity": config.result_identity()}, require_digests=False)


if __name__ == "__main__":
    unittest.main()
