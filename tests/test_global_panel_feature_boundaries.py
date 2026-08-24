#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from features.FeatureEngineering import FeatureEngineer


class GlobalPanelFeatureBoundaryTest(unittest.TestCase):
    @staticmethod
    def _panel():
        return pd.DataFrame(
            {
                "time": list(pd.date_range("2026-01-01", periods=3, freq="1h")) * 2,
                "series_id": ["A", "A", "A", "B", "B", "B"],
                "x": [10.0, 20.0, 30.0, 1000.0, 2000.0, 3000.0],
                "y": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
            }
        )

    @staticmethod
    def _basic_args(pred_method):
        return SimpleNamespace(
            pred_method=pred_method,
            enable_lags_features=True,
            lags=[1],
            align_direct_features_to_target=False,
            enable_global_training=True,
            series_id_feature="series_id",
        )

    def test_univariate_lag_and_direct_target_do_not_cross_series(self):
        engineer = FeatureEngineer(
            self._basic_args("univariate-single-multistep-direct"),
            "[test]",
            verbose=False,
        )

        featured, _predictors, targets = engineer.create_endogenous_basic_features(
            df_series=self._panel(),
            target_feature="y",
            endogenous_features_with_target=["y"],
            horizon=1,
        )

        self.assertTrue(np.isnan(featured.iloc[3]["y_lag_1"]))
        self.assertTrue(np.isnan(featured.iloc[2][targets[0]]))
        self.assertEqual(featured.iloc[3][targets[0]], 200.0)

    def test_multivariate_lags_do_not_cross_series(self):
        engineer = FeatureEngineer(
            self._basic_args("multivariate-single-multistep-direct"),
            "[test]",
            verbose=False,
        )

        featured, _predictors, _targets = engineer.create_endogenous_basic_features(
            df_series=self._panel(),
            target_feature="y",
            endogenous_features_with_target=["x", "y"],
            horizon=1,
        )

        self.assertTrue(np.isnan(featured.iloc[3]["x_lag_1"]))
        self.assertTrue(np.isnan(featured.iloc[3]["y_lag_1"]))
        self.assertEqual(featured.iloc[4]["x_lag_1"], 1000.0)
        self.assertEqual(featured.iloc[4]["y_lag_1"], 100.0)

    def test_advanced_sequential_features_restart_for_each_series(self):
        args = SimpleNamespace(
            enable_global_training=True,
            series_id_feature="series_id",
            enable_advanced_features=True,
            enable_rolling_features=True,
            rolling_columns=["y"],
            rolling_windows=[2],
            rolling_stats=["mean"],
            enable_expanding_features=True,
            expanding_columns=["y"],
            expanding_stats=["mean"],
            enable_diff_features=True,
            diff_columns=["y"],
            diff_periods=[1],
            enable_pct_change_features=True,
            pct_change_columns=["y"],
            pct_change_periods=[1],
            enable_time_since_features=False,
            enable_cyclical_features=False,
            enable_interaction_features=False,
            enable_polynomial_features=False,
        )
        engineer = FeatureEngineer(args, "[test]", verbose=False)

        featured, _features = engineer.create_endogenous_advanced_features(self._panel())

        self.assertEqual(featured.iloc[3]["y_rolling_mean_2"], 100.0)
        self.assertEqual(featured.iloc[3]["y_expanding_mean"], 100.0)
        self.assertTrue(np.isnan(featured.iloc[3]["y_diff_1"]))
        self.assertTrue(np.isnan(featured.iloc[3]["y_pct_change_1"]))

    def test_time_since_feature_restarts_for_each_series(self):
        panel = self._panel()
        panel["y"] = [1.0, 3.0, 1.0, 100.0, 200.0, 300.0]
        args = SimpleNamespace(
            enable_global_training=True,
            series_id_feature="series_id",
            enable_advanced_features=True,
            enable_rolling_features=False,
            enable_expanding_features=False,
            enable_diff_features=False,
            enable_pct_change_features=False,
            enable_time_since_features=True,
            time_since_columns=["y"],
            time_since_events=["peak"],
            enable_cyclical_features=False,
            enable_interaction_features=False,
            enable_polynomial_features=False,
        )
        engineer = FeatureEngineer(args, "[test]", verbose=False)

        featured, _features = engineer.create_endogenous_advanced_features(panel)

        self.assertEqual(featured.iloc[3]["y_time_since_peak"], 0.0)

    def test_global_mode_requires_series_id_column(self):
        panel = self._panel().drop(columns=["series_id"])
        engineer = FeatureEngineer(
            self._basic_args("univariate-single-multistep-direct"),
            "[test]",
            verbose=False,
        )

        with self.assertRaisesRegex(
            ValueError,
            r"enable_global_training=true requires series_id column 'series_id'",
        ):
            engineer.create_endogenous_basic_features(
                df_series=panel,
                target_feature="y",
                endogenous_features_with_target=["y"],
                horizon=1,
            )

    def test_horizon_exogenous_shift_does_not_cross_series(self):
        args = SimpleNamespace(
            enable_global_training=True,
            series_id_feature="series_id",
        )
        engineer = FeatureEngineer(args, "[test]", verbose=False)
        panel = self._panel().rename(columns={"x": "weather"})

        featured, _features = engineer._expand_horizon_exogenous_for_direct(
            panel,
            exogenous_features=["weather"],
            horizon=2,
        )

        self.assertTrue(np.isnan(featured.iloc[2]["weather_h1"]))
        self.assertEqual(featured.iloc[3]["weather_h1"], 2000.0)


if __name__ == "__main__":
    unittest.main()
