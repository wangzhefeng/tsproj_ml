"""新增窗口能力的独立黄金值及真实训练/在线/部署闭环。"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import pickle
import numpy as np
import pandas as pd
from tests import test_canonical_runtime_smoke as smoke
from data_loading import SourceRegistry
from feature_engineering import FeatureCompiler
from pipeline.supervised_design import SupervisedDesignBuilder
from pipeline.runner import run_canonical_config
from pipeline.online import RollingForecastSession


class WindowExtensionsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.times = pd.date_range("2026-01-01", periods=60, freq="h")
        self.frame = pd.DataFrame({"time": self.times, "load": np.arange(60.) ** 2 + 1})
        path = self.root / "history.csv"
        self.frame.to_csv(path, index=False)
        base = smoke.CanonicalRuntimeSmokeTest().build_config(path, mode="point", strategy="direct")
        self.config = replace(base, estimator=replace(base.estimator, params={"alpha": 1e-12}),
        features=replace(base.features, datetime_features=(), transformations={"feature_scaling": {"method": "standard"}, "advanced": {
            "rolling_quantile": {"columns": ["load"], "windows": [4], "quantiles": [0.25, 0.75]},
            "lagged_rolling": {"columns": ["load"], "windows": [3], "offsets": [2], "stats": ["mean"]},
        }, "interactions": {"window_product": ["load_rolling_quantile_0.25_4", "load_rolling_mean_3_offset_2"]}}),
        validation={"forecast_origin": self.times[47].isoformat(), "history_steps": 30,
                    "train_window_steps": 12, "fold_count": 2, "stride_steps": 2})

    def test_single_batch_values_and_window_bounds(self):
        builder = SupervisedDesignBuilder(self.config, SourceRegistry(self.config.data, self.root))
        request = builder.request(self.times[30])
        info = builder.registry.materialize(request)
        compiler = FeatureCompiler(self.config)
        single = compiler.compile(info, request)
        batch = compiler.compile_batch((info,), (request,))[0]
        pd.testing.assert_frame_equal(single.frame, batch.frame, check_exact=True)
        expected_quantile = np.quantile(self.frame.load.iloc[27:31], 0.25)
        expected_lagged = np.mean(self.frame.load.iloc[26:29])
        for name, expected in (("load_rolling_quantile_0.25_4", expected_quantile),
                               ("load_rolling_mean_3_offset_2", expected_lagged),
                               ("window_product", expected_quantile * expected_lagged)):
            np.testing.assert_array_equal(single.frame[name], [expected, expected])
        short = replace(request, history_start=self.times[28])
        with self.assertRaisesRegex(ValueError, "history|window"):
            compiler.compile(builder.registry.materialize(short), short)

    def test_real_fit_online_restore_and_nonlinear_prediction(self):
        result = run_canonical_config(self.config, output_root=self.root / "results")
        bundle = pickle.loads(pickle.dumps(result.bundle))
        self.assertIn("load_rolling_quantile_0.25_4", bundle.input_schema["columns"])
        lineage = {item["feature"]: item for item in bundle.feature_lineage}
        self.assertIn("derivation", lineage["window_product"])
        self.assertEqual(lineage["window_product"]["derivation"]["inputs"],
                         ["load_rolling_quantile_0.25_4", "load_rolling_mean_3_offset_2"])
        self.assertEqual(lineage["load_rolling_mean_3_offset_2"]["derivation"]["parameters"]["offsets"], [2])
        session = RollingForecastSession(self.config, bundle, self.frame.iloc[:48], origin=self.times[47])
        first = session.predict().values
        restored = RollingForecastSession.from_state(self.config, bundle, pickle.loads(pickle.dumps(session.state())))
        np.testing.assert_array_equal(first, restored.predict().values)
        expected = self.frame.load.iloc[48:50].to_numpy()
        np.testing.assert_allclose(first.reshape(-1), expected, rtol=1e-5, atol=1e-3)
        session.update(self.frame.iloc[48:50], origin=self.times[49])
        self.assertEqual(session.retention_steps, 5)
        self.assertEqual(session.origin, self.times[49])


if __name__ == "__main__":
    unittest.main()
