"""真实 bundle 的有界追加、重建和恢复，不建立第二套特征/预测链。"""
from dataclasses import replace
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd

from data_loading import SourceRegistry
from model_predicting.loops.deployment import predict_strategy_bundle
from pipeline.online import RollingForecastSession
from pipeline.runner import run_canonical_config
from pipeline.lifecycle import CanonicalRuntimeResult
from forecasting_core.tensors.point import PointForecastTensor
from pipeline.supervised_design import SupervisedDesignBuilder
import test_canonical_runtime_smoke as fixtures


class OnlinePredictionTest(unittest.TestCase):
    def test_append_restore_rebuild_and_full_history_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "load.csv"
            frame = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=60, freq="h"),
                                  "load": 100 + np.arange(60) + np.sin(np.arange(60))})
            frame.to_csv(source, index=False)
            frame = pd.read_csv(source, parse_dates=["time"])
            config = fixtures.CanonicalRuntimeSmokeTest().build_config(source, mode="point", strategy="recursive")
            config = replace(config, features=replace(config.features, transformations={
                "advanced": {
                    "rolling": {"windows": [3, 5], "stats": ["mean", "std"], "columns": ["load"]},
                    "ewm": {"halflives": [2.0], "stats": ["mean", "std"], "columns": ["load"]},
                    "time_since": {"events": ["peak", "trough"], "columns": ["load"]},
                    "expanding": {"stats": ["median", "mean", "std", "min", "max_diff"], "columns": ["load"]},
                },
            }), validation={"forecast_origin": "2026-01-02T23:00:00", "history_steps": 30,
                            "train_window_steps": 12, "fold_count": 2, "stride_steps": 2})
            result = run_canonical_config(config, output_root=root / "results")
            assert isinstance(result, CanonicalRuntimeResult)
            bundle = result.bundle
            session = RollingForecastSession(config, bundle, frame.iloc[:48], origin=frame.time.iloc[47])
            previous = 48
            for stop in (48, 49, 52, 56):
                if stop > 48:
                    session.update(frame.iloc[previous:stop], origin=frame.time.iloc[stop - 1])
                previous = stop
                prediction = session.predict()
                builder = SupervisedDesignBuilder(config, SourceRegistry(config.data, root))
                origin = frame.time.iloc[stop - 1]
                designs, provider = builder.forecast_designs(origin, target_transform=bundle.target_transform)
                reference = predict_strategy_bundle(bundle, designs[0], forecast_times=builder.request(origin).forecast_times,
                                                     raw_feature_provider=provider)
                assert isinstance(prediction, PointForecastTensor) and isinstance(reference, PointForecastTensor)
                np.testing.assert_array_equal(prediction.values, reference.values)
                self.assertLessEqual(len(session.state()["history"]), session.retention_steps)
                self.assertEqual(len(session.state()["statistics"].expanding["load"].prefix), stop)
                self.assertEqual(session.state()["statistics"].state_policy["ewm"], "bounded")
                self.assertEqual(session.state()["statistics"].state_policy["expanding:load"], "growing_exact_prefix")
                restored = RollingForecastSession.from_state(config, bundle, pickle.loads(pickle.dumps(session.state())))
                np.testing.assert_array_equal(restored.predict().values, prediction.values)
                self.assertTrue(session.last_audit)
                for compiled in session.last_audit:
                    for lineage in compiled.source_lineage:
                        self.assertIsNone(lineage.path)
                        self.assertTrue(lineage.path_version.startswith("online:"))
            before = session.state()
            nonfinite = frame.iloc[56:57].copy()
            nonfinite.loc[56, "load"] = np.nan
            for bad, bad_origin in ((frame.iloc[55:56], frame.time.iloc[55]),
                                    (frame.iloc[58:59], frame.time.iloc[58]),
                                    (frame.iloc[56:59].iloc[[0, 2, 1]], frame.time.iloc[57]),
                                    (nonfinite, frame.time.iloc[56]),
                                    (frame.iloc[57:58], frame.time.iloc[56])):
                with self.assertRaises(ValueError):
                    session.update(bad, origin=bad_origin)
                pd.testing.assert_frame_equal(before["history"], session.state()["history"])
                self.assertEqual(before["origin"], session.state()["origin"])
            revised = frame.iloc[:56].copy()
            revised.loc[55, "load"] += 50
            session.rebuild(revised, origin=frame.time.iloc[55])
            self.assertFalse(np.array_equal(session.predict().values, prediction.values))
            corrupted = {**session.state(), "config_fingerprint": "0" * 64}
            with self.assertRaisesRegex(ValueError, "identity"):
                RollingForecastSession.from_state(config, bundle, corrupted)
            incompatible = fixtures.CanonicalRuntimeSmokeTest().build_config(source, mode="point", strategy="recursive")
            with self.assertRaisesRegex(ValueError, "identity"):
                RollingForecastSession(incompatible, bundle, frame.iloc[:48], origin=frame.time.iloc[47])


if __name__ == "__main__":
    unittest.main()
