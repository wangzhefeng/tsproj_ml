# -*- coding: utf-8 -*-
"""Phase 2 等价性验证：新 DecompositionPipeline vs 旧 TargetDecomposer。

旧实现冻结于 tests/fixtures/legacy_target_decomposition.py
（git show e4502fd:features/TargetDecomposition.py）。
除 Phase 0 两处已修复缺陷（linear+damped lookback、MSTL robust）外，
历史 transform 与未来 restore 输出必须逐位一致。

冻结核仍消费旧写法散列字段（``_legacy_args``，这是它的历史输入契约）；
新 pipeline 一侧一律走 ``decomposition`` mapping（``_new_cfg``）。
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from legacy_target_decomposition import TargetDecomposer  # noqa: E402

from decomposition import build_pipeline, resolve_decomposition_spec  # noqa: E402


def _legacy_args(method="none", periods=None, degree=1, forecast="polynomial",
                 damping=0.98, lookback=28, cycles=4, n_per_day=0):
    """冻结旧实现 TargetDecomposer 的历史输入契约（散列字段），保持不变。"""
    return SimpleNamespace(
        decomposition_method=method,
        decomposition_periods=list(periods or []),
        decomposition_robust=True,
        decomposition_trend_degree=degree,
        decomposition_trend_forecast=forecast,
        decomposition_damping=damping,
        decomposition_trend_lookback=lookback,
        decomposition_seasonal_cycles=cycles,
        decomposition={},
        n_per_day=n_per_day,
    )


def _new_cfg(method="none", periods=None, degree=1, forecast="polynomial",
             damping=0.98, lookback=28, cycles=4, robust=None):
    """与 _legacy_args 语义一致的新写法 mapping（喂 resolve_decomposition_spec）。"""
    raw: dict = {"method": method}
    if method == "linear":
        raw.update(trend_degree=degree, trend_forecast=forecast,
                   damping=damping, trend_lookback=lookback)
    elif method in ("stl", "mstl"):
        raw["periods"] = list(periods or [])
        if robust is not None:
            raw["robust"] = robust
        raw.update(trend_degree=degree, trend_forecast=forecast,
                   damping=damping, trend_lookback=lookback,
                   seasonal_cycles=cycles)
    return SimpleNamespace(decomposition=raw)


def _make_series(n=120, period=24, base=100.0, slope=0.1, amp=8.0, seed=42):
    rng = np.random.RandomState(seed)
    times = pd.date_range("2026-01-01", periods=n, freq="1h")
    x = np.arange(n, dtype=float)
    y = base + slope * x + amp * np.sin(2.0 * np.pi * x / period) + rng.normal(0, 0.5, n)
    return pd.DataFrame({"time": times, "y": y})

class EquivalenceNoneTest(unittest.TestCase):
    def test_none_identity(self):
        df = _make_series()
        old = TargetDecomposer(_legacy_args("none")).fit_transform(df.copy())
        new_spec = resolve_decomposition_spec(_new_cfg("none"))
        new = build_pipeline(new_spec).fit_transform(df.copy())
        np.testing.assert_array_equal(old["y"].values, new["y"].values)
        np.testing.assert_array_equal(old["y"].values, df["y"].values)


class EquivalenceLinearTest(unittest.TestCase):
    def _assert_linear_equal(self, **kwargs):
        df = _make_series(**{k: v for k, v in kwargs.pop("_series", {}).items()})
        method = kwargs.pop("method", "linear")
        old_dec = TargetDecomposer(_legacy_args(method, **kwargs))
        old = old_dec.fit(df.copy()).transform(df.copy())
        old_restore = old_dec.restore(np.zeros(10), pd.date_range("2026-01-06", periods=10, freq="1h"))

        new_spec = resolve_decomposition_spec(_new_cfg(method, **kwargs))
        new_pipe = build_pipeline(new_spec)
        new = new_pipe.fit_transform(df.copy())
        new_restore = new_pipe.restore(np.zeros(10), pd.date_range("2026-01-06", periods=10, freq="1h"))

        np.testing.assert_allclose(old["y"].values, new["y"].values, atol=1e-10)
        np.testing.assert_allclose(old_restore, new_restore, atol=1e-10)

    def test_linear_polynomial(self):
        self._assert_linear_equal(method="linear", degree=1)

    def test_linear_quadratic_polynomial(self):
        self._assert_linear_equal(method="linear", degree=2)


class EquivalenceSTLTest(unittest.TestCase):
    def _assert_stl_equal(self, **kwargs):
        series_kwargs = kwargs.pop("_series", {})
        df = _make_series(**series_kwargs)
        method = kwargs.pop("method", "stl")
        old_dec = TargetDecomposer(_legacy_args(method, **kwargs))
        old = old_dec.fit(df.copy()).transform(df.copy())
        old_restore = old_dec.restore(np.zeros(12), pd.date_range("2026-01-06", periods=12, freq="1h"))

        new_spec = resolve_decomposition_spec(_new_cfg(method, **kwargs))
        new_pipe = build_pipeline(new_spec)
        new = new_pipe.fit_transform(df.copy())
        new_restore = new_pipe.restore(np.zeros(12), pd.date_range("2026-01-06", periods=12, freq="1h"))

        np.testing.assert_allclose(old["y"].values, new["y"].values, atol=1e-6)
        np.testing.assert_allclose(old_restore, new_restore, atol=1e-6)

    def test_stl96_robust_true(self):
        self._assert_stl_equal(method="stl", periods=[24])

    def test_stl96_robust_false(self):
        # robust=false 需要显式设置；旧 args 默认 True，这里直接构造
        df = _make_series()
        legacy = _legacy_args("stl", periods=[24])
        legacy.decomposition_robust = False
        old_dec = TargetDecomposer(legacy)
        old = old_dec.fit(df.copy()).transform(df.copy())

        spec = resolve_decomposition_spec(_new_cfg("stl", periods=[24], robust=False))
        new_pipe = build_pipeline(spec)
        new = new_pipe.fit_transform(df.copy())
        np.testing.assert_allclose(old["y"].values, new["y"].values, atol=1e-6)


class EquivalenceMSTLTest(unittest.TestCase):
    def _make_mstl_series(self, n=400):
        rng = np.random.RandomState(7)
        times = pd.date_range("2026-01-01", periods=n, freq="1h")
        x = np.arange(n, dtype=float)
        y = (
            100.0
            + 0.05 * x
            + 6.0 * np.sin(2.0 * np.pi * x / 24)
            + 10.0 * np.sin(2.0 * np.pi * x / 168)
            + rng.normal(0, 0.4, n)
        )
        return pd.DataFrame({"time": times, "y": y})

    def _assert_mstl_equal(self, **kwargs):
        df = self._make_mstl_series()
        old_dec = TargetDecomposer(_legacy_args("mstl", **kwargs))
        old = old_dec.fit(df.copy()).transform(df.copy())
        future = pd.date_range("2026-01-13", periods=24, freq="1h")
        old_restore = old_dec.restore(np.zeros(len(future)), future)

        new_spec = resolve_decomposition_spec(_new_cfg("mstl", **kwargs))
        new_pipe = build_pipeline(new_spec)
        new = new_pipe.fit_transform(df.copy())
        new_restore = new_pipe.restore(np.zeros(len(future)), future)

        np.testing.assert_allclose(old["y"].values, new["y"].values, atol=1e-6)
        np.testing.assert_allclose(old_restore, new_restore, atol=1e-6)

    def test_mstl_two_periods_robust_true(self):
        self._assert_mstl_equal(periods=[24, 168])

    def test_mstl_two_periods_robust_false(self):
        df = self._make_mstl_series()
        legacy = _legacy_args("mstl", periods=[24, 168])
        legacy.decomposition_robust = False
        old_dec = TargetDecomposer(legacy)
        old = old_dec.fit(df.copy()).transform(df.copy())

        spec = resolve_decomposition_spec(_new_cfg("mstl", periods=[24, 168], robust=False))
        new_pipe = build_pipeline(spec)
        new = new_pipe.fit_transform(df.copy())
        np.testing.assert_allclose(old["y"].values, new["y"].values, atol=1e-6)


class PickleTest(unittest.TestCase):
    def test_pipeline_pickle_roundtrip(self):
        import pickle

        df = _make_series(n=60)
        spec = resolve_decomposition_spec(_new_cfg("stl", periods=[24]))
        pipe = build_pipeline(spec).fit(df)
        data = pickle.dumps(pipe)
        restored = pickle.loads(data)
        future = pd.date_range("2026-01-06", periods=6, freq="1h")
        np.testing.assert_allclose(pipe.restore(np.zeros(6), future), restored.restore(np.zeros(6), future))


class RegistryFailFastTest(unittest.TestCase):
    def test_custom_rejected_at_registry(self):
        from decomposition.construction.registry import build_pipeline

        with self.assertRaisesRegex(ValueError, "not implemented"):
            build_pipeline(resolve_decomposition_spec(
                SimpleNamespace(decomposition={"method": "custom"})
            ))


if __name__ == "__main__":
    unittest.main()
