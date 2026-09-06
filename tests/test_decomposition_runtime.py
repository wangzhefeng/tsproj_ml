"""真实 Ridge/QR 的分解训练—回测—预测—bundle 部署往返（合成测试数据）。"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import test_canonical_runtime_smoke as smoke
from data_loading import SourceRegistry
from model_forecasting.deployment import predict_strategy_bundle
from model_pipeline.runner import run_canonical_config
from model_pipeline.supervised_design import SupervisedDesignBuilder
from models.pickle_io import ModelDeployPkl


class DecompositionRuntimeTest(unittest.TestCase):
    def test_seasonal_nondefault_runtime_and_deployment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x = np.arange(96, dtype=float)
            data_path = root / "targets.csv"
            pd.DataFrame({"time": pd.date_range("2026-01-01", periods=len(x), freq="h"),
                          "load": 100 + 0.01 * x ** 2 + (2 + 0.01 * x) * np.sin(2 * np.pi * x / 6),
                          "power": 10000 + 0.5 * x ** 2 + 100 * np.cos(2 * np.pi * x / 12)}).to_csv(data_path, index=False)
            for mode in ("point", "quantile"):
                for method, periods in (("stl", [6]), ("mstl", [6, 12])):
                    with self.subTest(mode=mode, method=method):
                        config = smoke.CanonicalRuntimeSmokeTest().build_transformed_k2_config(data_path, mode=mode)
                        transformations = dict(config.features.transformations)
                        target = dict(transformations["target"])
                        target["decomposition"] = {"method": method, "periods": periods, "trend_degree": 2,
                                                   "trend_forecast": "damped", "trend_lookback": 5, "seasonal_cycles": 1}
                        transformations["target"] = target
                        config = replace(config, features=replace(config.features, transformations=transformations))
                        result = run_canonical_config(config, output_root=root / "results")
                        prediction = pd.read_csv(result.forecast_dir / "prediction.csv")
                        scores = pd.read_csv(result.test_dir / "test_scores_df.csv")
                        self.assertFalse(scores.empty)
                        self.assertTrue((result.model_dir / "resolved_model.json").is_file())
                        self.assertEqual(len(prediction), 4)
                        self.assertTrue(np.isfinite(prediction["predict_value"]).all())
                        loaded = ModelDeployPkl(str(result.model_dir / "model.pkl")).load_model()
                        self.assertEqual(loaded.config_fingerprint, config.fingerprint())
                        self.assertEqual(set(loaded.target_transform.fitted_keys), {("__local__", "load"), ("__local__", "power")})
                        origin = pd.Timestamp(config.validation["forecast_origin"])
                        builder = SupervisedDesignBuilder(config, SourceRegistry(config.data, root))
                        designs, provider = builder.forecast_designs(origin, target_transform=loaded.target_transform)
                        times = pd.date_range(origin + pd.Timedelta(hours=1), periods=2, freq="h")
                        deployed = predict_strategy_bundle(loaded, designs[0], forecast_times=times, raw_feature_provider=provider)
                        point = deployed if mode == "point" else deployed.point
                        for k, target_name in enumerate(config.problem.targets):
                            rows = prediction.loc[prediction.target == target_name].sort_values("time")
                            np.testing.assert_allclose(point.values[0, :, k], rows.predict_value.to_numpy(), rtol=1e-12, atol=1e-8)
                            if mode == "quantile":
                                for q, column in enumerate(("predict_q10", "predict_q50", "predict_q90")):
                                    np.testing.assert_allclose(deployed.quantiles.values[0, :, k, q], rows[column].to_numpy(), rtol=1e-12, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
