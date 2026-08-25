#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd

from features.FeatureEngineering import FeatureEngineer
from models.ModelForecasting import Forecaster
from models.multistep.executors.pointwise import PointwiseExecutor


class _LagEchoModel:
    def __init__(self, column: str):
        self.column = column
        self.last_X = None

    def predict(self, X):
        self.last_X = X.copy()
        return X[self.column].to_numpy(dtype=float)


class UsmdpSafeLagTest(unittest.TestCase):
    @staticmethod
    def _args(lags):
        return SimpleNamespace(
            pred_method="univariate-single-multistep-direct-pointwise",
            freq="1h",
            lags=list(lags),
            enable_lags_features=True,
            align_direct_features_to_target=True,
            enable_date_features=False,
            enable_weather_features=False,
            enable_datetime_features=False,
            datetime_features=[],
            datetime_categorical_features=[],
            custom_features=[],
            use_horizon_exogenous_for_direct=False,
            direct_strategy="multioutput",
            enable_global_training=False,
            series_id_feature="series_id",
            enable_advanced_features=False,
        )

    def _forecaster(self, lags):
        args = self._args(lags)
        history = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01 00:00:00", periods=5, freq="1h"),
                "y": [10.0, 11.0, 12.0, 13.0, 14.0],
            }
        )
        future = pd.DataFrame(
            {"time": pd.date_range("2026-01-01 05:00:00", periods=3, freq="1h")}
        )
        model = _LagEchoModel(f"y_lag_{lags[0]}")
        forecaster = cast(Any, Forecaster.__new__(Forecaster))
        forecaster.args = args
        forecaster.horizon = 3
        forecaster.model = model
        forecaster.df_history_for_lags = history.copy()
        forecaster.df_future = future
        forecaster.df_date_future = None
        forecaster.df_weather_future = None
        forecaster.df_custom_future = []
        forecaster.endogenous_features = ["y"]
        forecaster.target_feature = "y"
        forecaster.selected_features = None
        forecaster.categorical_features = []
        forecaster.feature_engineer = FeatureEngineer(args, "[test]", verbose=False)
        forecaster._quantile_outputs = None
        forecaster.log_prefix = "[test]"
        forecaster._slice_future_aux_by_forecast = lambda df_forecast: (None, None)
        forecaster._transform_features = lambda X, categorical_features: X
        return forecaster, model

    def test_multistep_safe_lag_uses_only_known_history(self):
        forecaster, model = self._forecaster([3, 4])

        prediction = PointwiseExecutor().execute(forecaster)

        np.testing.assert_array_equal(prediction, np.array([12.0, 13.0, 14.0]))
        model_input = cast(pd.DataFrame, model.last_X)
        self.assertEqual(model_input["y_lag_3"].tolist(), [12.0, 13.0, 14.0])
        self.assertEqual(model_input["y_lag_4"].tolist(), [11.0, 12.0, 13.0])

    def test_multistep_safe_lag_rejects_lag_shorter_than_horizon(self):
        forecaster, _model = self._forecaster([2, 3])

        with self.assertRaisesRegex(
            ValueError,
            r"USMDP safe-lag requires min\(lags\) >= horizon; got min_lag=2, horizon=3",
        ):
            PointwiseExecutor().execute(forecaster)


if __name__ == "__main__":
    unittest.main()
