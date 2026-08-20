# -*- coding: utf-8 -*-
"""origin-frozen custom feature 的方法感知时间对齐回归测试。"""
import unittest
from types import SimpleNamespace

import pandas as pd

from features.FeatureEngineering import FeatureEngineer


def _args(method: str, use_horizon: bool = False):
    return SimpleNamespace(
        pred_method=method,
        freq="1D",
        lags=[1],
        enable_lags_features=True,
        align_direct_features_to_target=False,
        enable_date_features=False,
        enable_weather_features=False,
        enable_datetime_features=False,
        datetime_features=[],
        datetime_categorical_features=[],
        custom_features=[],
        use_horizon_exogenous_for_direct=use_horizon,
        direct_strategy="multioutput",
        enable_global_training=False,
        enable_advanced_features=False,
    )


def _source():
    return [{
        "name": "load_state",
        "ts_col": "time",
        "columns": ["state_value"],
        "categorical_columns": [],
        "future_strategy": "freeze_last_observation",
        "availability": "end_of_period",
        "df": pd.DataFrame({
            "time": pd.date_range("2026-01-01", periods=6, freq="1D"),
            "state_value": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }),
    }]


class OriginFrozenCustomFeatureTest(unittest.TestCase):
    def test_recursive_training_row_uses_previous_period_state(self):
        series = pd.DataFrame({
            "time": pd.date_range("2026-01-02", periods=4, freq="1D"),
            "y": [100.0, 110.0, 120.0, 130.0],
        })
        featured, predictors, targets, _ = FeatureEngineer(
            _args("univariate-single-multistep-recursive"), "[test]", verbose=False
        ).create_features(
            df_series=series,
            df_custom_history=_source(),
            endogenous_features_with_target=["y"],
            target_feature="y",
            horizon=1,
        )

        self.assertIn("state_value", predictors)
        # 01-02 的目标只能看到 01-01 日末状态 10，而非同日状态 20。
        self.assertEqual(featured.loc[pd.Timestamp("2026-01-02"), "state_value"], 10.0)
        self.assertEqual(featured.loc[pd.Timestamp("2026-01-03"), "state_value"], 20.0)

    def test_direct_keeps_origin_state_and_does_not_generate_state_horizon_columns(self):
        series = pd.DataFrame({
            "time": pd.date_range("2026-01-01", periods=6, freq="1D"),
            "y": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
        })
        featured, predictors, targets, _ = FeatureEngineer(
            _args("univariate-single-multistep-direct", use_horizon=True),
            "[test]",
            verbose=False,
        ).create_features(
            df_series=series,
            df_custom_history=_source(),
            endogenous_features_with_target=["y"],
            target_feature="y",
            horizon=2,
        )

        self.assertIn("state_value", predictors)
        self.assertNotIn("state_value_h1", predictors)
        self.assertNotIn("state_value_h2", predictors)
        self.assertEqual(featured.loc[pd.Timestamp("2026-01-02"), "state_value"], 20.0)

    def test_contemporaneous_custom_feature_still_expands_by_horizon(self):
        source = _source()
        source[0]["future_strategy"] = "explicit"
        source[0]["availability"] = "contemporaneous"
        series = pd.DataFrame({
            "time": pd.date_range("2026-01-01", periods=6, freq="1D"),
            "y": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
        })
        _, predictors, _, _ = FeatureEngineer(
            _args("univariate-single-multistep-direct", use_horizon=True),
            "[test]",
            verbose=False,
        ).create_features(
            df_series=series,
            df_custom_history=source,
            endogenous_features_with_target=["y"],
            target_feature="y",
            horizon=2,
        )

        self.assertIn("state_value_h1", predictors)
        self.assertIn("state_value_h2", predictors)


if __name__ == "__main__":
    unittest.main()
