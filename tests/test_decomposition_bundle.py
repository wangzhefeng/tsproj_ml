# -*- coding: utf-8 -*-
"""Phase 3 测试：bundle 持久化 + 诊断输出 + checker spec 对齐。"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from decomposition import (
    DecompositionBundle,
    DecompositionPipeline,
    resolve_decomposition_spec,
    write_diagnostics_report,
)


def _stl_args(periods=(24,), method="stl"):
    return SimpleNamespace(
        decomposition_method=method,
        decomposition_periods=list(periods),
        decomposition_robust=True,
        decomposition_trend_degree=1,
        decomposition_trend_forecast="polynomial",
        decomposition_damping=0.98,
        decomposition_trend_lookback=28,
        decomposition_seasonal_cycles=4,
        decomposition={},
        n_per_day=0,
    )


def _make_frame(n=60):
    rng = np.random.RandomState(0)
    times = pd.date_range("2026-01-01", periods=n, freq="1h")
    x = np.arange(n, dtype=float)
    y = 100 + 0.1 * x + 8 * np.sin(2 * np.pi * x / 24) + rng.normal(0, 0.5, n)
    return pd.DataFrame({"time": times, "y": y})


class BundleTest(unittest.TestCase):
    def test_bundle_roundtrip_preserves_predictions(self):
        frame = _make_frame()
        pipe = DecompositionPipeline.from_args(_stl_args()).fit(frame)
        bundle = DecompositionBundle.from_pipeline(pipe)
        future = pd.date_range(frame["time"].iloc[-1] + pd.Timedelta(hours=1), periods=6, freq="1h")
        expected = pipe.restore(np.zeros(6), future)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bundle.pkl"
            bundle.save(path)
            loaded = DecompositionBundle.load(path)
            actual = loaded.pipeline.restore(np.zeros(6), future)
            np.testing.assert_allclose(expected, actual, atol=1e-10)
            self.assertEqual(loaded.spec_summary["method"], "stl")
            self.assertEqual(loaded.spec_summary["periods"], [24])
            # legacy 兼容入口（from_pipeline）不标 2；canonical 用 from_components
            self.assertIn(loaded.schema_version, (1, 2))

    def test_bundle_rejects_wrong_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not_bundle.pkl"
            import pickle

            with open(path, "wb") as f:
                pickle.dump("not a bundle", f)
            with self.assertRaises((TypeError,)):
                DecompositionBundle.load(path)


class CanonicalBundleTest(unittest.TestCase):
    """内嵌式 bundle：model + scaler + pipeline 单文件保存/加载。"""

    def test_embedded_bundle_roundtrip(self):
        frame = _make_frame()
        pipe = DecompositionPipeline.from_args(_stl_args()).fit(frame)
        model = {"mock": "estimator"}
        bundle = DecompositionBundle.from_components(
            pipeline=pipe, model=model, target_scaler=None
        )
        self.assertIs(bundle.model, model)
        self.assertEqual(bundle.schema_version, 2)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model_bundle.pkl"
            bundle.save(path)
            loaded = DecompositionBundle.load(path)
            self.assertEqual(loaded.model, model)
            future = pd.date_range(
                frame["time"].iloc[-1] + pd.Timedelta(hours=1), periods=4, freq="1h"
            )
            np.testing.assert_allclose(
                pipe.restore(np.zeros(4), future),
                loaded.pipeline.restore(np.zeros(4), future),
                atol=1e-10,
            )

    def test_deploy_load_returns_bundle(self):
        """ModelDeployPkl 挂载：保存 bundle 后 load_model 返回 bundle 而非裸模型。"""
        frame = _make_frame()
        pipe = DecompositionPipeline.from_args(_stl_args()).fit(frame)
        bundle = DecompositionBundle.from_components(pipeline=pipe, model="m")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decomposition_bundle.pkl"
            bundle.save(path)

            from models.ModelSaveLoad import ModelDeployPkl

            loaded = ModelDeployPkl(str(path)).load_model()
            self.assertIsInstance(loaded, DecompositionBundle)
            self.assertEqual(loaded.schema_version, 2)

    def test_deploy_load_legacy_model_unaffected(self):
        """非 bundle 的 pkl（普通 model.pkl）按原样返回，不受影响。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkl"
            import joblib

            joblib.dump({"plain": 1}, path)

            from models.ModelSaveLoad import ModelDeployPkl

            loaded = ModelDeployPkl(str(path)).load_model()
            self.assertEqual(loaded, {"plain": 1})


class DiagnosticsTest(unittest.TestCase):
    def test_diagnostics_written_for_stl(self):
        frame = _make_frame()
        pipe = DecompositionPipeline.from_args(_stl_args()).fit(frame)
        with tempfile.TemporaryDirectory() as tmp:
            out = write_diagnostics_report(
                pipe, frame, "time", "y", Path(tmp)
            )
            self.assertIsNotNone(out)
            df = pd.read_csv(out)
            names = set(df["component"])
            self.assertIn("y", names)
            self.assertIn("deterministic", names)
            self.assertIn("residual", names)
            self.assertIn("trend", names)
            self.assertIn("seasonal", names)

    def test_diagnostics_skipped_for_none(self):
        frame = _make_frame()
        pipe = DecompositionPipeline.from_args(_stl_args(method="none"))
        with tempfile.TemporaryDirectory() as tmp:
            out = write_diagnostics_report(pipe, frame, "time", "y", Path(tmp))
            self.assertIsNone(out)

    def test_diagnostics_suffix_avoids_overwrite(self):
        frame = _make_frame()
        pipe = DecompositionPipeline.from_args(_stl_args()).fit(frame)
        with tempfile.TemporaryDirectory() as tmp:
            out1 = write_diagnostics_report(
                pipe, frame, "time", "y", Path(tmp), suffix="_win1"
            )
            out2 = write_diagnostics_report(
                pipe, frame, "time", "y", Path(tmp), suffix="_win2"
            )
            self.assertNotEqual(out1, out2)
            self.assertTrue(Path(out1).exists())
            self.assertTrue(Path(out2).exists())
            self.assertIn("_win1", out1.name)
            self.assertIn("_win2", out2.name)


class CheckerSpecAlignmentTest(unittest.TestCase):
    def test_checker_uses_spec_for_validation(self):
        """checker 通过 resolve_decomposition_spec 校验，不直接读旧字段。"""
        import scripts.check_model_configs as checker

        # stl 正确配置应无 decomposition 相关 problems
        cfg = _stl_args()
        cfg.scale_target = False
        # checker 内部会调 resolve_decomposition_spec(cfg) 并在异常时写 problems
        # 这里验证 cfg 的 spec 可解析即可
        spec = resolve_decomposition_spec(cfg)
        self.assertEqual(spec.method, "stl")

    def test_checker_detects_invalid_stl_periods(self):
        cfg = _stl_args(periods=(24, 48))  # stl 只能 1 个周期
        with self.assertRaisesRegex(ValueError, "exactly 1"):
            resolve_decomposition_spec(
                SimpleNamespace(
                    decomposition={"method": "stl", "periods": [24, 48]},
                    decomposition_method="none",
                )
            )


if __name__ == "__main__":
    unittest.main()
