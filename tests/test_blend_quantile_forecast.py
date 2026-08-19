# -*- coding: utf-8 -*-

from types import SimpleNamespace
from typing import Any
import unittest

import numpy as np
import pandas as pd

from models.ModelForecasting import Forecaster


class BlendQuantileForecastTest(unittest.TestCase):
    def test_blend_quantile_uses_shared_weights_and_median_as_point_prediction(self):
        forecaster: Any = Forecaster.__new__(Forecaster)
        direct_q10, recursive_q10 = object(), object()
        direct_q50, recursive_q50 = object(), object()
        direct_q90, recursive_q90 = object(), object()
        original_bundle = {
            "models": {
                0.1: {"direct": direct_q10, "recursive": recursive_q10},
                0.5: {"direct": direct_q50, "recursive": recursive_q50},
                0.9: {"direct": direct_q90, "recursive": recursive_q90},
            },
            "median_quantile": 0.5,
        }
        predictions = {
            direct_q10: np.array([1.0, 2.0]),
            recursive_q10: np.array([3.0, 4.0]),
            direct_q50: np.array([10.0, 20.0]),
            recursive_q50: np.array([30.0, 40.0]),
            direct_q90: np.array([100.0, 200.0]),
            recursive_q90: np.array([300.0, 400.0]),
        }
        forecaster.args = SimpleNamespace(
            pred_method="univariate-single-multistep-blend-direct-recursive"
        )
        forecaster.model = original_bundle
        forecaster.df_history = pd.DataFrame({"y": [1.0, 2.0]})
        forecaster.max_lag = 1
        forecaster._recursive_schema_cache = {}
        forecaster._is_quantile_bundle = lambda: True
        forecaster._resolve_blend_weights = lambda: np.array([0.25, 0.75])
        forecaster.univariate_single_multi_step_direct_forecast = lambda: predictions[forecaster.model]
        forecaster.univariate_single_multi_step_recursive_forecast = lambda: predictions[forecaster.model]
        forecaster.log_prefix = "[test]"

        point_prediction = forecaster._blend_forecast_quantile()

        np.testing.assert_allclose(forecaster.quantile_outputs[0.1], [2.5, 3.5])
        np.testing.assert_allclose(forecaster.quantile_outputs[0.5], [25.0, 35.0])
        np.testing.assert_allclose(forecaster.quantile_outputs[0.9], [250.0, 350.0])
        np.testing.assert_allclose(point_prediction, forecaster.quantile_outputs[0.5])
        np.testing.assert_allclose(forecaster.blend_direct_pred, [10.0, 20.0])
        np.testing.assert_allclose(forecaster.blend_recursive_pred, [30.0, 40.0])
        self.assertIs(forecaster.model, original_bundle)


if __name__ == "__main__":
    unittest.main()
