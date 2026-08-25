# -*- coding: utf-8 -*-
"""派生天气与 custom future 的预测接线回归测试。"""
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from features.FeatureEngineering import ExogenousFeatureEngineer, FeatureEngineer
from models.ModelForecasting import Forecaster
from models.multistep.executors.recursive import RecursiveExecutor


class NativeDerivedWeatherTest(unittest.TestCase):
    def test_precomputed_rh_and_derived_columns_are_preserved(self):
        args = SimpleNamespace(
            weather_features=[
                "rt_tt2",
                "cal_rh",
                "rt_ssr",
                "rt_ws10",
                "tt2_mean_3h",
                "tt2_diff_1h",
                "ssr_mean_3h",
            ],
            weather_categorical_features=[],
        )
        times = pd.date_range("2026-07-29", periods=3, freq="5min")
        target = pd.DataFrame({"time": times})
        weather = pd.DataFrame({
            "time": times,
            "rt_tt2": [300.0, 301.0, 302.0],
            "cal_rh": [61.0, 62.0, 63.0],
            "rt_ssr": [0.0, 10.0, 20.0],
            "rt_ws10": [1.0, 1.1, 1.2],
            "tt2_mean_3h": [299.0, 300.0, 301.0],
            "tt2_diff_1h": [0.5, 0.6, 0.7],
            "ssr_mean_3h": [0.0, 5.0, 10.0],
        })

        engineer = ExogenousFeatureEngineer(args, verbose=False)
        featured = engineer.extend_weather_feature(target, weather, col_ts="time")

        self.assertIn("time", featured.columns)
        self.assertEqual(featured["time"].tolist(), times.tolist())
        for column in args.weather_features:
            self.assertIn(column, featured.columns)
            np.testing.assert_allclose(featured[column], weather[column])
        self.assertEqual(engineer.get_generated_features()[0], args.weather_features)

    def test_future_preserves_all_precomputed_derived_columns(self):
        args = SimpleNamespace(
            weather_features=[
                "rt_tt2", "cal_rh", "rt_ssr", "rt_ws10",
                "tt2_mean_3h", "tt2_diff_1h", "ssr_mean_3h",
            ],
            weather_categorical_features=[],
        )
        times = pd.date_range("2026-07-29", periods=3, freq="5min")
        target = pd.DataFrame({"time": times})
        weather = pd.DataFrame({
            "time": times,
            "rt_tt2": [300.0, 301.0, 302.0],
            "cal_rh": [61.0, 62.0, 63.0],
            "rt_ssr": [0.0, 10.0, 20.0],
            "rt_ws10": [1.0, 1.1, 1.2],
            "tt2_mean_3h": [299.0, 300.0, 301.0],
            "tt2_diff_1h": [0.5, 0.6, 0.7],
            "ssr_mean_3h": [0.0, 5.0, 10.0],
        })

        engineer = ExogenousFeatureEngineer(args, verbose=False)
        featured = engineer.extend_future_weather_feature(target, weather, col_ts="time")

        self.assertIn("time", featured.columns)
        for column in args.weather_features:
            self.assertIn(column, featured.columns)
            np.testing.assert_allclose(featured[column], weather[column])
        self.assertEqual(engineer.get_generated_features()[0], args.weather_features)


class ExogenousFillIsolationTest(unittest.TestCase):
    def test_exogenous_fill_never_interpolates_target_y(self):
        args = SimpleNamespace(
            enable_date_features=False,
            enable_weather_features=False,
            enable_datetime_features=True,
            datetime_features=["hour", "minute"],
            datetime_categorical_features=[],
            strict_weather_information_set=False,
            strict_date_information_set=False,
            custom_features=[],
            freq="5min",
        )
        frame = pd.DataFrame({
            "time": pd.date_range("2026-07-29", periods=3, freq="5min"),
            "y": [1.0, np.nan, 3.0],
        })
        featured, _, _ = FeatureEngineer(args, "[test]", False).create_exogenouse_features(
            df=frame,
            df_date_history=None,
            df_date_future=None,
            df_weather_history=None,
            df_weather_future=None,
        )
        self.assertTrue(pd.isna(featured.loc[1, "y"]))


class RecursiveCustomFutureWiringTest(unittest.TestCase):
    def test_usmr_passes_custom_future_to_feature_engineering(self):
        captured = {}

        class FeatureEngineerSpy:
            def create_features(self, **kwargs):
                captured.update(kwargs)
                frame = kwargs["df_series"].copy()
                frame["pcs_plan"] = 123.0
                return frame, ["pcs_plan"], ["y_shift_0"], []

        forecaster = object.__new__(Forecaster)
        forecaster.args = SimpleNamespace(
            pred_method="univariate-single-multistep-recursive",
            enable_step_logging=False,
            forecast_log_interval=1,
        )
        forecaster.horizon = 1
        forecaster.df_future = pd.DataFrame({
            "time": [pd.Timestamp("2026-07-29 00:00:00")],
        })
        forecaster.df_history_for_lags = pd.DataFrame({
            "time": [pd.Timestamp("2026-07-28 23:55:00")],
            "y": [10.0],
        })
        forecaster.df_date_future = None
        forecaster.df_weather_future = None
        forecaster.df_custom_future = [{
            "name": "pcs_plan",
            "ts_col": "time",
            "columns": ["pcs_plan"],
            "categorical_columns": [],
            "df": pd.DataFrame({
                "time": [pd.Timestamp("2026-07-29 00:00:00")],
                "pcs_plan": [123.0],
            }),
        }]
        forecaster.endogenous_features = ["y"]
        forecaster.target_feature = "y"
        forecaster.feature_engineer = FeatureEngineerSpy()
        forecaster.selected_features = None
        forecaster.log_prefix = "[test]"
        forecaster._quantile_outputs = None
        forecaster._slice_future_aux_by_forecast = lambda frame: (None, None)
        forecaster._get_recursive_schema = lambda key: None
        forecaster._set_recursive_schema = lambda *args, **kwargs: None
        forecaster._apply_selected_feature_subset = lambda predictors, categorical: (predictors, categorical)
        forecaster._transform_features = lambda frame, categorical: frame
        forecaster._predict_point_and_quantiles = lambda frame: (np.array([11.0]), {})
        forecaster._record_quantile_recursive_step = lambda store, preds: None
        forecaster._append_history_row = lambda row: None
        forecaster._finalize_recursive_quantiles = lambda store: None

        prediction = RecursiveExecutor().execute(forecaster)

        np.testing.assert_allclose(prediction, [11.0])
        self.assertIs(captured.get("df_custom_future"), forecaster.df_custom_future)


if __name__ == "__main__":
    unittest.main()
