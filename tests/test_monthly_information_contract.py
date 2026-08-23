# -*- coding: utf-8 -*-
"""月频信息集、特征时间对齐和目标归一化回归测试。"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from features.FeatureEngineering import FeatureEngineer
from features.TargetNormalization import CalendarDayTargetNormalizer
from models.ModelTesting import Tester
from utils.weather_contract import (
    validate_weather_availability,
    validate_weather_coverage,
    validate_weather_information_contract,
)


class CalendarDayTargetNormalizerTests(unittest.TestCase):
    def test_per_calendar_day_transform_and_restore(self):
        args = SimpleNamespace(target_calendar_normalization="per_calendar_day")
        normalizer = CalendarDayTargetNormalizer(args)
        history = pd.DataFrame({
            "time": pd.to_datetime(["2026-01-31", "2026-02-28"]),
            "y": [310.0, 280.0],
        })

        transformed = normalizer.transform_history(history, time_col="time", target_col="y")
        restored = normalizer.restore_predictions(
            np.array([10.0, 10.0]),
            pd.to_datetime(["2026-03-31", "2026-04-30"]),
        )

        np.testing.assert_allclose(transformed["y"], [10.0, 10.0])
        np.testing.assert_allclose(restored, [310.0, 300.0])
        np.testing.assert_allclose(history["y"], [310.0, 280.0])


class MonthlyFeatureAlignmentTests(unittest.TestCase):
    @staticmethod
    def _args(method: str):
        return SimpleNamespace(
            pred_method=method,
            lags=[1, 2, 3],
            enable_lags_features=True,
            align_direct_features_to_target=True,
            enable_date_features=False,
            enable_weather_features=True,
            weather_ts_feat="ts",
            weather_features=["rt_tt2"],
            weather_categorical_features=[],
            enable_datetime_features=False,
            datetime_features=[],
            datetime_categorical_features=[],
            custom_features=[],
            use_horizon_exogenous_for_direct=False,
            enable_global_training=False,
            enable_advanced_features=False,
        )

    @staticmethod
    def _series_and_weather():
        times = pd.to_datetime(["2026-01-31", "2026-02-28", "2026-03-31", "2026-04-30"])
        series = pd.DataFrame({"time": times, "y": [100.0, 200.0, 300.0, 400.0]})
        weather = pd.DataFrame({
            "ts": times,
            "rt_tt2": [283.15, 293.15, 303.15, 313.15],
            "rt_dt": [278.15, 288.15, 298.15, 308.15],
        })
        return series, weather

    def test_usmd_lag1_is_current_month_and_weather_is_target_month(self):
        series, weather = self._series_and_weather()
        featured, predictors, targets, _ = FeatureEngineer(self._args(
            "univariate-single-multistep-direct"
        ), "[test]", verbose=False).create_features(
            df_series=series,
            df_weather_history=weather,
            endogenous_features_with_target=["y"],
            target_feature="y",
            horizon=1,
        )

        row = featured.loc[pd.Timestamp("2026-02-28")]
        self.assertEqual(row["y_lag_1"], 200.0)
        self.assertEqual(row["rt_tt2"], 303.15)
        self.assertEqual(row[targets[0]], 300.0)
        self.assertIn("y_lag_1", predictors)

    def test_usmdp_generates_lags_when_target_alignment_enabled(self):
        series, weather = self._series_and_weather()
        featured, predictors, targets, _ = FeatureEngineer(self._args(
            "univariate-single-multistep-direct-pointwise"
        ), "[test]", verbose=False).create_features(
            df_series=series,
            df_weather_history=weather,
            endogenous_features_with_target=["y"],
            target_feature="y",
            horizon=1,
        )

        row = featured.loc[pd.Timestamp("2026-02-28")]
        self.assertEqual(row["y_lag_1"], 100.0)
        self.assertEqual(row["rt_tt2"], 293.15)
        self.assertEqual(row[targets[0]], 200.0)
        self.assertIn("y_lag_1", predictors)


class WeatherInformationContractTests(unittest.TestCase):
    def test_future_weather_available_after_origin_is_rejected(self):
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2026-08-31"]),
            "available_at": pd.to_datetime(["2026-08-14"]),
        })
        with self.assertRaisesRegex(ValueError, "after forecast origin"):
            validate_weather_availability(
                df,
                ts_col="ts",
                label="Future weather",
                forecast_origin=pd.Timestamp("2026-07-31"),
            )

    def test_backtest_weather_available_inside_target_month_is_rejected(self):
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2026-07-31"]),
            "available_at": pd.to_datetime(["2026-07-14"]),
        })
        with self.assertRaisesRegex(ValueError, "before target month"):
            validate_weather_availability(
                df,
                ts_col="ts",
                label="Backtest weather",
                require_before_target_month=True,
            )

    def test_forecast_mode_rejects_future_actual_weather(self):
        args = SimpleNamespace(
            strict_weather_information_set=True,
            enable_weather_features=True,
            is_testing=True,
            is_forecasting=True,
            forecast_mode="forecast",
            weather_history_source="actual",
            weather_backtest_path="backtest.csv",
            weather_backtest_source="proxy",
            weather_future_path="future.csv",
            weather_future_source="actual",
        )
        with self.assertRaisesRegex(ValueError, "future weather source"):
            validate_weather_information_contract(args)

    def test_strict_testing_rejects_actual_backtest_weather(self):
        args = SimpleNamespace(
            strict_weather_information_set=True,
            enable_weather_features=True,
            is_testing=True,
            is_forecasting=False,
            forecast_mode="forecast",
            weather_history_source="actual",
            weather_backtest_path="backtest.csv",
            weather_backtest_source="actual",
            weather_future_path=None,
            weather_future_source="",
        )
        with self.assertRaisesRegex(ValueError, "backtest weather source"):
            validate_weather_information_contract(args)

    def test_coverage_requires_every_target_timestamp(self):
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2026-05-31", "2026-07-31"]),
            "rt_tt2": [1.0, 2.0],
        })
        expected = pd.to_datetime(["2026-05-31", "2026-06-30", "2026-07-31"])
        with self.assertRaisesRegex(ValueError, "2026-06-30"):
            validate_weather_coverage(df, expected, "ts", "Backtest weather")

    def test_window_weather_uses_backtest_frame_not_history_actuals(self):
        history = pd.DataFrame({"ts": pd.to_datetime(["2026-07-31"]), "rt_tt2": [999.0]})
        backtest = pd.DataFrame({
            "ts": pd.to_datetime(["2026-07-31"]),
            "available_at": pd.to_datetime(["2026-06-30"]),
            "rt_tt2": [300.0],
        })
        test = pd.DataFrame({"time": pd.to_datetime(["2026-07-31"]), "y": [1.0]})
        args = SimpleNamespace(
            strict_weather_information_set=True,
            enable_weather_features=True,
            weather_ts_feat="ts",
        )
        resolved = Tester._resolve_window_weather_future(
            args=args,
            df_history_test=test,
            df_weather_history=history,
            df_weather_backtest=backtest,
            fold_origin=pd.Timestamp("2026-06-30"),
            log_prefix="[test]",
        )
        self.assertEqual(resolved.loc[0, "rt_tt2"], 300.0)


if __name__ == "__main__":
    unittest.main()
