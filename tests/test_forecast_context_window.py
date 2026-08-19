# -*- coding: utf-8 -*-

from types import SimpleNamespace
from typing import Any
import unittest

import pandas as pd

from models.ModelForecasting import Forecaster


class ForecastContextWindowTest(unittest.TestCase):
    def test_context_length_covers_enabled_advanced_feature_lookback(self):
        args: Any = SimpleNamespace(
            lags=[288, 2016],
            enable_advanced_features=True,
            enable_rolling_features=True,
            rolling_windows=[2016, 4032, 8064],
            enable_diff_features=True,
            diff_periods=[288, 2016, 8064],
            enable_pct_change_features=False,
            pct_change_periods=[1, 7],
        )

        self.assertEqual(Forecaster._resolve_history_context_length(args), 8064)

    def test_initial_history_buffer_uses_feature_context_length(self):
        args: Any = SimpleNamespace(
            lags=[288, 2016],
            enable_advanced_features=True,
            enable_rolling_features=True,
            rolling_windows=[2016, 4032, 8064],
            enable_diff_features=True,
            diff_periods=[288, 2016, 8064],
            enable_pct_change_features=False,
            pct_change_periods=[],
        )
        history = pd.DataFrame({"y": range(9000)})
        forecaster = Forecaster(
            args=args,
            horizon=288,
            model=object(),
            feature_scaler=None,
            target_scaler=None,
            df_history=history,
            df_future=pd.DataFrame(),
            df_date_future=pd.DataFrame(),
            df_weather_future=pd.DataFrame(),
            endogenous_features=[],
            target_feature="y",
            target_output_features=[],
            categorical_features=[],
        )

        self.assertEqual(forecaster.history_context_length, 8064)
        self.assertEqual(len(forecaster.df_history_for_lags), 8064)
        self.assertEqual(forecaster.df_history_for_lags["y"].iloc[0], 936)

    def test_recursive_append_retains_feature_context_length(self):
        forecaster: Any = Forecaster.__new__(Forecaster)
        forecaster.max_lag = 2
        forecaster.history_context_length = 5
        forecaster.df_history_for_lags = pd.DataFrame({"y": [0.0, 1.0, 2.0, 3.0, 4.0]})

        forecaster._append_history_row(pd.DataFrame({"y": [5.0]}))

        self.assertEqual(len(forecaster.df_history_for_lags), 5)
        self.assertEqual(forecaster.df_history_for_lags["y"].tolist(), [1.0, 2.0, 3.0, 4.0, 5.0])


if __name__ == "__main__":
    unittest.main()
