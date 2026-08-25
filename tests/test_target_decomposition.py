# -*- coding: utf-8 -*-

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import statsmodels.tsa.seasonal

from decomposition import DecompositionPipeline
from models.ModelForecasting import Forecaster
from models.ModelSaveLoad import ModelDeployPkl
from models.ModelTraining import Trainer


class TargetDecomposerTest(unittest.TestCase):

    @staticmethod
    def _args(method, periods=None, degree=1, **extra):
        base = dict(
            decomposition_method=method,
            decomposition_periods=list(periods or []),
            decomposition_trend_degree=degree,
            decomposition_trend_forecast="polynomial",
            decomposition_damping=0.98,
        )
        base.update(extra)
        return SimpleNamespace(**base)

    @staticmethod
    def _make_decomposer(args):
        return DecompositionPipeline.from_args(args)

    def test_linear_damped_forecast_consumes_trend_lookback(self):
        # 前段斜率 0.1，末端 lookback 段斜率 10.0：damped 外推应使用末端斜率
        times = pd.date_range("2026-01-01", periods=40, freq="1D")
        x = np.arange(40, dtype=float)
        y = 100.0 + 0.1 * x + 9.9 * np.maximum(0.0, x - 29.0)
        frame = pd.DataFrame({"time": times, "y": y})
        future_times = pd.date_range(times[-1] + pd.Timedelta(days=1), periods=5, freq="1D")

        def damped_future(lookback):
            args = self._args(
                "linear",
                decomposition_trend_forecast="damped",
                decomposition_trend_lookback=lookback,
                decomposition_damping=0.98,
            )
            decomposer = TargetDecomposerTest._make_decomposer(args).fit(frame)
            return decomposer.restore(np.zeros(len(future_times)), future_times)

        recent = damped_future(7)
        long_window = damped_future(40)
        # lookback=7 只看到末端陡峭段（局部斜率 10），lookback=40 覆盖全窗平缓段
        self.assertGreater(recent[-1] - recent[0], 30.0)
        # lookback=40 时阻尼后的全窗平均斜率仍给出温和外推，但起点显著低于陡峭段
        self.assertLess(long_window[-1] - long_window[0], 10.0)
        self.assertNotAlmostEqual(float(recent[0]), float(long_window[0]))

    def test_linear_decomposition_reconstructs_history_and_future(self):
        times = pd.date_range("2026-01-01", periods=40, freq="1D")
        y = 100.0 + 2.5 * np.arange(40, dtype=float)
        frame = pd.DataFrame({"time": times, "y": y})

        decomposer = TargetDecomposerTest._make_decomposer(self._args("linear"))
        transformed = decomposer.fit_transform(frame)

        np.testing.assert_allclose(transformed["y"], 0.0, atol=1e-10)
        np.testing.assert_allclose(
            decomposer.restore(transformed["y"], times),
            y,
            atol=1e-10,
        )

        future_times = pd.date_range("2026-02-10", periods=3, freq="1D")
        np.testing.assert_allclose(
            decomposer.restore(np.zeros(3), future_times),
            [200.0, 202.5, 205.0],
            atol=1e-10,
        )

    def test_stl_decomposition_reconstructs_history_and_repeats_seasonality(self):
        period = 12
        times = pd.date_range("2026-01-01", periods=period * 10, freq="1h")
        x = np.arange(len(times), dtype=float)
        y = 50.0 + 0.1 * x + 8.0 * np.sin(2.0 * np.pi * x / period)
        frame = pd.DataFrame({"time": times, "y": y})

        decomposer = TargetDecomposerTest._make_decomposer(self._args("stl", [period]))
        transformed = decomposer.fit_transform(frame)

        np.testing.assert_allclose(
            decomposer.restore(transformed["y"], times),
            y,
            atol=1e-8,
        )
        future_times = pd.date_range(times[-1] + pd.Timedelta(hours=1), periods=period, freq="1h")
        future_level = decomposer.restore(np.zeros(period), future_times)
        self.assertTrue(np.isfinite(future_level).all())
        self.assertGreater(np.ptp(future_level), 10.0)

    def test_stl_allows_exactly_two_complete_cycles(self):
        period = 12
        times = pd.date_range("2026-01-01", periods=period * 2, freq="1h")
        x = np.arange(len(times), dtype=float)
        frame = pd.DataFrame(
            {"time": times, "y": 50.0 + 5.0 * np.sin(2.0 * np.pi * x / period)}
        )

        transformed = TargetDecomposerTest._make_decomposer(self._args("stl", [period])).fit_transform(frame)

        self.assertTrue(np.isfinite(transformed["y"]).all())

    def test_mstl_passes_robust_to_statsmodels(self):
        periods = [6, 12]
        times = pd.date_range("2026-01-01", periods=48, freq="1h")
        x = np.arange(len(times), dtype=float)
        frame = pd.DataFrame(
            {
                "time": times,
                "y": 50.0
                + 5.0 * np.sin(2.0 * np.pi * x / periods[0])
                + 3.0 * np.sin(2.0 * np.pi * x / periods[1]),
            }
        )

        for robust in (True, False):
            with self.subTest(robust=robust):
                captured = {}

                real_mstl = statsmodels.tsa.seasonal.MSTL

                def spy_mstl(y, periods=None, **kwargs):
                    captured["stl_kwargs"] = kwargs.get("stl_kwargs")
                    return real_mstl(y, periods=periods, **kwargs)

                args = self._args("mstl", periods, decomposition_robust=robust)
                with patch.object(statsmodels.tsa.seasonal, "MSTL", spy_mstl):
                    TargetDecomposerTest._make_decomposer(args).fit(frame)

                self.assertEqual(captured["stl_kwargs"], {"robust": robust})

    def test_forecaster_restores_point_and_quantile_outputs_with_same_component(self):
        times = pd.date_range("2026-01-01", periods=10, freq="1D")
        frame = pd.DataFrame({"time": times, "y": 100.0 + np.arange(10, dtype=float)})
        decomposer = TargetDecomposerTest._make_decomposer(self._args("linear")).fit(frame)
        future_times = pd.date_range("2026-01-11", periods=2, freq="1D")

        forecaster = Forecaster.__new__(Forecaster)
        forecaster.df_future = pd.DataFrame({"time": future_times})
        forecaster.target_decomposer = decomposer
        forecaster._quantile_outputs = {
            0.1: np.array([-2.0, -2.0]),
            0.5: np.array([0.0, 0.0]),
            0.9: np.array([2.0, 2.0]),
        }

        restored = forecaster._restore_target_decomposition(np.array([0.0, 0.0]))

        np.testing.assert_allclose(restored, [110.0, 111.0], atol=1e-10)
        np.testing.assert_allclose(forecaster.quantile_outputs[0.1], [108.0, 109.0], atol=1e-10)
        np.testing.assert_allclose(forecaster.quantile_outputs[0.5], [110.0, 111.0], atol=1e-10)
        np.testing.assert_allclose(forecaster.quantile_outputs[0.9], [112.0, 113.0], atol=1e-10)

    def test_model_save_persists_fitted_target_decomposer(self):
        times = pd.date_range("2026-01-01", periods=10, freq="1D")
        frame = pd.DataFrame({"time": times, "y": 100.0 + np.arange(10, dtype=float)})
        decomposer = TargetDecomposerTest._make_decomposer(self._args("linear")).fit(frame)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer.__new__(Trainer)
            trainer.args = SimpleNamespace(
                checkpoints_dir=Path(tmpdir),
                model_type="lightgbm",
                pred_method="univariate-single-multistep-recursive",
                horizon=2,
                lags=[1],
            )
            trainer.log_prefix = "[test]"

            trainer.model_save({"model": "fake"}, target_decomposer=decomposer)

            model_path = Path(tmpdir) / "model.pkl"
            self.assertTrue(model_path.exists())
            self.assertFalse(Path(tmpdir, "decomposition_bundle.pkl").exists())
            from probabilistic.types import ForecastModelBundle

            bundle = ModelDeployPkl(model_path).load_model()
            self.assertIsInstance(bundle, ForecastModelBundle)
            self.assertEqual(bundle.schema_version, 1)
            np.testing.assert_allclose(
                bundle.target_transform.restore(
                    np.zeros(2), pd.date_range("2026-01-11", periods=2, freq="1D")
                ),
                [110.0, 111.0],
                atol=1e-10,
            )


if __name__ == "__main__":
    unittest.main()
