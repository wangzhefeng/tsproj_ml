# -*- coding: utf-8 -*-

from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from models.ModelForecasting import Forecaster
from models.multistep.executors.blend import BlendExecutor
from probabilistic.spec import ProbabilisticSpec
from probabilistic.types import BlendQuantileModel, ProbabilisticModelBundle


class BlendQuantileForecastTest(unittest.TestCase):
    def test_blend_quantile_uses_shared_weights_and_median_as_point_prediction(self):
        forecaster: Any = Forecaster.__new__(Forecaster)
        direct_q10, recursive_q10 = object(), object()
        direct_q50, recursive_q50 = object(), object()
        direct_q90, recursive_q90 = object(), object()
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
        original_bundle = ProbabilisticModelBundle(
            schema_version=1,
            spec=spec,
            model_type="lightgbm",
            pred_method="univariate-single-multistep-blend-direct-recursive",
            models_by_quantile={
                0.1: BlendQuantileModel(direct_q10, recursive_q10),
                0.5: BlendQuantileModel(direct_q50, recursive_q50),
                0.9: BlendQuantileModel(direct_q90, recursive_q90),
            },
            recursive_propagation="median_path",
        )
        predictions = {
            direct_q10: np.array([1.0, 2.0]),
            recursive_q10: np.array([3.0, 4.0]),
            direct_q50: np.array([10.0, 20.0]),
            recursive_q50: np.array([30.0, 40.0]),
            direct_q90: np.array([100.0, 200.0]),
            recursive_q90: np.array([300.0, 400.0]),
        }
        forecaster.args = SimpleNamespace(
            pred_method="univariate-single-multistep-blend-direct-recursive",
            predict_type="quantile",
        )
        forecaster.horizon = 2
        forecaster.model = original_bundle
        forecaster.df_history = pd.DataFrame({"y": [1.0, 2.0]})
        forecaster.df_history_for_lags = forecaster.df_history.copy(deep=True)
        forecaster.max_lag = 1
        forecaster.history_context_length = 1
        forecaster._recursive_schema_cache = {}
        forecaster._is_quantile_bundle = lambda: True
        forecaster._resolve_blend_weights = lambda: np.array([0.25, 0.75])
        forecaster.log_prefix = "[test]"

        with patch(
            "models.multistep.executors.blend.DirectExecutor.execute",
            side_effect=lambda context: predictions[context.model],
        ), patch(
            "models.multistep.executors.blend.RecursiveExecutor.execute",
            side_effect=lambda context: predictions[context.model],
        ):
            point_prediction = BlendExecutor().execute(forecaster)

        np.testing.assert_allclose(forecaster.quantile_outputs[0.1], [2.5, 3.5])
        np.testing.assert_allclose(forecaster.quantile_outputs[0.5], [25.0, 35.0])
        np.testing.assert_allclose(forecaster.quantile_outputs[0.9], [250.0, 350.0])
        np.testing.assert_allclose(point_prediction, forecaster.quantile_outputs[0.5])
        np.testing.assert_allclose(forecaster.blend_direct_pred, [10.0, 20.0])
        np.testing.assert_allclose(forecaster.blend_recursive_pred, [30.0, 40.0])
        self.assertIs(forecaster.model, original_bundle)

    def test_msbr_typed_quantile_bundle_returns_multivariate_blend_shape(self):
        forecaster: Any = Forecaster.__new__(Forecaster)
        direct_models = {level: object() for level in (0.1, 0.5, 0.9)}
        recursive_models = {level: object() for level in (0.1, 0.5, 0.9)}
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
            pred_method="multivariate-single-multistep-blend-direct-recursive",
            models_by_quantile={
                level: BlendQuantileModel(
                    direct_models[level],
                    recursive_models[level],
                )
                for level in spec.quantiles
            },
            recursive_propagation="median_path",
        )
        predictions = {
            direct_models[0.1]: np.array([1.0, 2.0]),
            recursive_models[0.1]: np.array([3.0, 4.0]),
            direct_models[0.5]: np.array([10.0, 20.0]),
            recursive_models[0.5]: np.array([30.0, 40.0]),
            direct_models[0.9]: np.array([100.0, 200.0]),
            recursive_models[0.9]: np.array([300.0, 400.0]),
        }
        forecaster.args = SimpleNamespace(
            pred_method="multivariate-single-multistep-blend-direct-recursive",
            predict_type="quantile",
        )
        forecaster.horizon = 2
        forecaster.model = bundle
        forecaster.df_history = pd.DataFrame({"y": [1.0, 2.0]})
        forecaster.df_history_for_lags = forecaster.df_history.copy(deep=True)
        forecaster.max_lag = 1
        forecaster.history_context_length = 1
        forecaster._recursive_schema_cache = {}
        forecaster._is_quantile_bundle = lambda: True
        forecaster._resolve_blend_weights = lambda: np.array([0.5, 0.5])
        forecaster.log_prefix = "[test]"

        with patch(
            "models.multistep.executors.blend.DirectExecutor.execute",
            side_effect=lambda context: predictions[context.model],
        ), patch(
            "models.multistep.executors.blend.RecursiveExecutor.execute",
            side_effect=lambda context: predictions[context.model],
        ):
            point_prediction = BlendExecutor().execute(forecaster)

        self.assertEqual(point_prediction.shape, (2,))
        self.assertEqual(
            np.column_stack(
                [forecaster.quantile_outputs[level] for level in spec.quantiles]
            ).shape,
            (2, 3),
        )
        np.testing.assert_allclose(point_prediction, [20.0, 30.0])
        self.assertIs(forecaster.model, bundle)


if __name__ == "__main__":
    unittest.main()
