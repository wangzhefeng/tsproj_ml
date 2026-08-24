# -*- coding: utf-8 -*-
"""共享目标变换栈的顺序与可逆性测试。"""

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from features.TargetTransformation import TargetTransformPipeline
from models.ModelForecasting import Forecaster


class TargetTransformPipelineTest(unittest.TestCase):
    @staticmethod
    def _args():
        return SimpleNamespace(
            target_calendar_normalization="per_calendar_day",
            decomposition_method="linear",
            decomposition_periods=[],
            decomposition_robust=True,
            decomposition_trend_damping=1.0,
            decomposition_trend_degree=1,
            decomposition_trend_lookback=4,
            decomposition_seasonal_cycles=2,
            decomposition={},
            scale_target=True,
            inverse_target=True,
            target_scaler_type="standard",
        )

    def test_calendar_decomposition_scaler_roundtrip_uses_exact_reverse_order(self):
        times = pd.date_range("2026-01-30", periods=4, freq="1D")
        original = np.array([310.0, 341.0, 336.0, 364.0])
        frame = pd.DataFrame({"time": times, "y": original})
        pipeline = TargetTransformPipeline.from_args(self._args())

        transformed_history = pipeline.fit_transform_history(frame)
        transformed_targets = pipeline.fit_transform_targets(
            transformed_history[["y"]],
        )
        restored = pipeline.restore(
            transformed_targets["y"].to_numpy(),
            times,
            target_columns=["y"],
        )

        self.assertEqual(
            pipeline.training_steps,
            ("calendar_normalization", "decomposition", "target_scaling"),
        )
        np.testing.assert_allclose(restored, original, atol=1e-10)

    def test_point_and_quantile_matrix_share_the_same_restorer(self):
        times = pd.date_range("2026-01-01", periods=5, freq="1D")
        frame = pd.DataFrame(
            {"time": times, "y": 100.0 + np.arange(5, dtype=float) * 2.0}
        )
        args = self._args()
        args.target_calendar_normalization = "none"
        pipeline = TargetTransformPipeline.from_args(args)
        transformed = pipeline.fit_transform_history(frame)
        scaled = pipeline.fit_transform_targets(transformed[["y"]])["y"].to_numpy()
        quantile_matrix = np.column_stack([scaled - 0.1, scaled, scaled + 0.1])

        restored_point = pipeline.restore(scaled, times, target_columns=["y"])
        restored_quantiles = pipeline.restore_quantile_matrix(
            quantile_matrix,
            times,
            target_columns=["y"],
        )

        np.testing.assert_allclose(restored_point, frame["y"], atol=1e-10)
        np.testing.assert_allclose(restored_quantiles[:, 1], restored_point, atol=1e-10)
        self.assertEqual(restored_quantiles.shape, (5, 3))

    def test_restore_rejects_time_and_value_length_mismatch(self):
        pipeline = TargetTransformPipeline.from_args(self._args())
        frame = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=3, freq="1D"),
                "y": [1.0, 2.0, 3.0],
            }
        )
        transformed = pipeline.fit_transform_history(frame)
        pipeline.fit_transform_targets(transformed[["y"]])

        with self.assertRaisesRegex(ValueError, "length mismatch"):
            pipeline.restore(
                np.array([1.0, 2.0]),
                frame["time"],
                target_columns=["y"],
            )

    def test_forecaster_restores_point_and_all_quantiles_through_one_pipeline(self):
        args = self._args()
        args.target_calendar_normalization = "none"
        args.decomposition_method = "none"
        frame = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=3, freq="1D"),
                "y": [10.0, 20.0, 30.0],
            }
        )
        pipeline = TargetTransformPipeline.from_args(args)
        transformed = pipeline.fit_transform_history(frame)
        pipeline.fit_transform_targets(transformed[["y"]])

        forecaster = Forecaster.__new__(Forecaster)
        forecaster.df_future = pd.DataFrame(
            {"time": pd.date_range("2026-01-04", periods=2, freq="1D")}
        )
        forecaster.target_transform = pipeline
        forecaster.prediction_target_columns = ["y"]
        forecaster.target_decomposer = None
        forecaster._quantile_outputs = {
            0.1: np.array([-1.0, 0.0]),
            0.5: np.array([0.0, 1.0]),
            0.9: np.array([1.0, 2.0]),
        }

        restored = forecaster._restore_target_transform(np.array([0.0, 1.0]))

        expected_point = pipeline.restore(
            np.array([0.0, 1.0]),
            forecaster.df_future["time"],
            target_columns=["y"],
        )
        np.testing.assert_allclose(restored, expected_point, atol=1e-12)
        np.testing.assert_allclose(
            forecaster.quantile_outputs[0.5],
            expected_point,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
