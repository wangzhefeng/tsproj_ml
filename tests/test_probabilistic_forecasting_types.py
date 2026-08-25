# -*- coding: utf-8 -*-
"""Forecaster 统一 ForecastDistribution 输出契约。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from models.ModelForecasting import Forecaster
from probabilistic.spec import ProbabilisticSpec
from probabilistic.types import ForecastDistribution, ProbabilisticModelBundle


class ForecastDistributionOutputTest(unittest.TestCase):
    def test_quantile_predict_by_method_returns_distribution_not_side_channel_tuple(self):
        spec = ProbabilisticSpec(
            mode="quantile",
            quantiles=(0.1, 0.5, 0.9),
            point_quantile=0.5,
            recursive_propagation="median_path",
            crossing_method="none",
            crossing_report_raw=True,
            intervals=(),
            calibration=None,
        )
        bundle = ProbabilisticModelBundle(
            schema_version=1,
            spec=spec,
            model_type="lightgbm",
            pred_method="univariate-single-multistep-direct-pointwise",
            models_by_quantile={0.1: object(), 0.5: object(), 0.9: object()},
            recursive_propagation="median_path",
        )
        forecaster = Forecaster.__new__(Forecaster)
        forecaster.args = SimpleNamespace(
            pred_method="univariate-single-multistep-direct-pointwise",
            predict_type="quantile",
        )
        forecaster.horizon = 2
        forecaster.model = bundle
        forecaster.df_future = pd.DataFrame(
            {"time": pd.date_range("2026-08-01", periods=2, freq="1D")}
        )
        forecaster.target_transform = None
        forecaster.target_decomposer = None
        forecaster.log_prefix = "[test]"

        def predict_method():
            forecaster._quantile_outputs = {
                0.1: np.array([8.0, 9.0]),
                0.5: np.array([10.0, 11.0]),
                0.9: np.array([12.0, 13.0]),
            }
            return np.array([10.0, 11.0])

        executor = SimpleNamespace(execute=lambda _context: predict_method())
        with patch("models.ModelForecasting.get_executor", return_value=executor):
            result = forecaster._predict_by_method()

        self.assertIsInstance(result, ForecastDistribution)
        self.assertEqual(result.space, "model")
        self.assertEqual(result.quantile_stage, "raw")
        np.testing.assert_array_equal(result.point, [10.0, 11.0])
        np.testing.assert_array_equal(result.quantile_values[:, 0], [8.0, 9.0])


if __name__ == "__main__":
    unittest.main()
