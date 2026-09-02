# -*- coding: utf-8 -*-
"""诊断输出 + checker spec 对齐 + pkl 加载透传测试。

历史 DecompositionBundle 持久化测试已随 bundle.py 一并删除
（canonical 产物统一走 ForecastModelBundle）。
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from decomposition import (
    DecompositionPipeline,
    resolve_decomposition_spec,
    write_diagnostics_report,
)


def _stl_args(periods=(24,), method="stl"):
    raw: dict = {"method": method}
    if method in ("stl", "mstl"):
        raw["periods"] = list(periods)
    return SimpleNamespace(decomposition=raw)


def _make_frame(n=60):
    rng = np.random.RandomState(0)
    times = pd.date_range("2026-01-01", periods=n, freq="1h")
    x = np.arange(n, dtype=float)
    y = 100 + 0.1 * x + 8 * np.sin(2 * np.pi * x / 24) + rng.normal(0, 0.5, n)
    return pd.DataFrame({"time": times, "y": y})


class DiagnosticsTest(unittest.TestCase):
    def test_diagnostics_written_for_stl(self):
        frame = _make_frame()
        pipe = DecompositionPipeline.from_args(_stl_args()).fit(frame)
        with tempfile.TemporaryDirectory() as tmp:
            out = write_diagnostics_report(
                pipe, frame, "time", "y", Path(tmp)
            )
            self.assertIsNotNone(out)
            assert out is not None
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
            assert out1 is not None and out2 is not None
            self.assertTrue(Path(out1).exists())
            self.assertTrue(Path(out2).exists())
            self.assertIn("_win1", out1.name)
            self.assertIn("_win2", out2.name)


class ModelDeployPassthroughTest(unittest.TestCase):
    def test_deploy_load_plain_pkl_unaffected(self):
        """非 bundle 的 pkl（普通 model.pkl）按原样返回，不受影响。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkl"
            import joblib

            joblib.dump({"plain": 1}, path)

            from models.ModelSaveLoad import ModelDeployPkl

            loaded = ModelDeployPkl(str(path)).load_model()
            self.assertEqual(loaded, {"plain": 1})


class CheckerSpecAlignmentTest(unittest.TestCase):
    def test_checker_uses_spec_for_validation(self):
        """checker 通过 resolve_decomposition_spec 校验，语义与 spec 层一致。"""
        import scripts.check_model_configs as checker  # noqa: F401

        spec = resolve_decomposition_spec(_stl_args())
        self.assertEqual(spec.method, "stl")

    def test_invalid_stl_periods_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly 1"):
            resolve_decomposition_spec(
                SimpleNamespace(decomposition={"method": "stl", "periods": [24, 48]})
            )


if __name__ == "__main__":
    unittest.main()
