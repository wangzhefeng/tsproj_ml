# -*- coding: utf-8 -*-

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from features.TargetDecomposition import TargetDecomposer
from models.ModelForecasting import Forecaster
from models.ModelSaveLoad import ModelDeployPkl
from models.ModelTraining import Trainer


class TargetDecomposerTest(unittest.TestCase):

    @staticmethod
    def _args(method, periods=None, degree=1):
        return SimpleNamespace(
            decomposition_method=method,
            decomposition_periods=list(periods or []),
            decomposition_trend_degree=degree,
            decomposition_trend_forecast="polynomial",
            decomposition_damping=0.98,
        )

    def test_linear_decomposition_reconstructs_history_and_future(self):
        times = pd.date_range("2026-01-01", periods=40, freq="1D")
        y = 100.0 + 2.5 * np.arange(40, dtype=float)
        frame = pd.DataFrame({"time": times, "y": y})

        decomposer = TargetDecomposer(self._args("linear"))
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

        decomposer = TargetDecomposer(self._args("stl", [period]))
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

        transformed = TargetDecomposer(self._args("stl", [period])).fit_transform(frame)

        self.assertTrue(np.isfinite(transformed["y"]).all())

    def test_forecaster_restores_point_and_quantile_outputs_with_same_component(self):
        times = pd.date_range("2026-01-01", periods=10, freq="1D")
        frame = pd.DataFrame({"time": times, "y": 100.0 + np.arange(10, dtype=float)})
        decomposer = TargetDecomposer(self._args("linear")).fit(frame)
        future_times = pd.date_range("2026-01-11", periods=2, freq="1D")

        forecaster = Forecaster.__new__(Forecaster)
        forecaster.df_future = pd.DataFrame({"time": future_times})
        forecaster.target_decomposer = decomposer
        forecaster.quantile_outputs = {
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
        decomposer = TargetDecomposer(self._args("linear")).fit(frame)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer.__new__(Trainer)
            trainer.args = SimpleNamespace(checkpoints_dir=Path(tmpdir))
            trainer.log_prefix = "[test]"

            trainer.model_save({"model": "fake"}, target_decomposer=decomposer)

            saved_path = Path(tmpdir) / "target_decomposer.pkl"
            self.assertTrue(saved_path.exists())
            restored = ModelDeployPkl(saved_path).load_model()
            self.assertTrue(restored.is_fitted)
            np.testing.assert_allclose(
                restored.restore(np.zeros(2), pd.date_range("2026-01-11", periods=2, freq="1D")),
                [110.0, 111.0],
                atol=1e-10,
            )


if __name__ == "__main__":
    unittest.main()
