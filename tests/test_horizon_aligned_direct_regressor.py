# -*- coding: utf-8 -*-
"""Direct多输出的目标horizon外生隔离测试。"""
import unittest

import numpy as np
import pandas as pd

from config.config_loader import load_yaml_config
from features.FeatureEngineering import FeatureEngineer
from models.ModelTraining import HorizonAlignedDirectRegressor
from pathlib import Path


class RecordingEstimator:
    def __init__(self):
        self.fit_columns = None
        self.fit_first_row = None

    def fit(self, X, y, **kwargs):
        self.fit_columns = list(X.columns)
        self.fit_first_row = X.iloc[0].to_dict()
        return self

    def predict(self, X):
        # 直接返回目标horizon plan，便于验证predict同样按输出隔离。
        return X["pcs_plan"].to_numpy(dtype=float)


class HorizonAlignedDirectRegressorTest(unittest.TestCase):
    def test_each_output_only_reads_its_target_horizon_exogenous(self):
        X = pd.DataFrame({
            "y_lag_288": [1.0, 2.0],
            "date_type": [9, 9],
            "pcs_plan": [999.0, 999.0],
            "date_type_h1": [1, 1],
            "pcs_plan_h1": [10.0, 11.0],
            "date_type_h2": [2, 2],
            "pcs_plan_h2": [20.0, 21.0],
        })
        Y = pd.DataFrame({"y_shift_1": [100.0, 101.0], "y_shift_2": [200.0, 201.0]})
        model = HorizonAlignedDirectRegressor(
            estimator_factory=RecordingEstimator,
            n_jobs=1,
            log_prefix="[test]",
        ).fit(X, Y)

        self.assertEqual(
            model.estimators_[0].fit_columns,
            ["y_lag_288", "date_type", "pcs_plan"],
        )
        self.assertEqual(model.estimators_[0].fit_first_row["date_type"], 1)
        self.assertEqual(model.estimators_[0].fit_first_row["pcs_plan"], 10.0)
        self.assertEqual(model.estimators_[1].fit_first_row["date_type"], 2)
        self.assertEqual(model.estimators_[1].fit_first_row["pcs_plan"], 20.0)

        pred = model.predict(X.iloc[:1])
        np.testing.assert_allclose(pred, [[10.0, 20.0]])

    def test_missing_horizon_column_fails_instead_of_using_base_value(self):
        X = pd.DataFrame({
            "y_lag_288": [1.0],
            "pcs_plan": [999.0],
            "pcs_plan_h1": [10.0],
        })
        Y = pd.DataFrame({"y_shift_1": [100.0], "y_shift_2": [200.0]})
        with self.assertRaisesRegex(ValueError, "horizon 2"):
            HorizonAlignedDirectRegressor(
                estimator_factory=RecordingEstimator,
                n_jobs=1,
                log_prefix="[test]",
            ).fit(X, Y)


class DirectRecursiveBlockContractTest(unittest.TestCase):
    def test_usmdr_trains_block_size_outputs_not_outer_horizon(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_yaml_config(str(
            root / "config/aidc_ess_selfuse_load/route_A/add_exogenous_plan_strategy/"
            "lgbm_usmdr_prob_mean_plan.yaml"
        ))
        cfg.block_size = 4
        cfg.lags = [1]
        cfg.enable_advanced_features = False
        times = pd.date_range("2026-01-01", periods=20, freq="5min")
        series = pd.DataFrame({"time": times, "y": np.arange(20, dtype=float)})
        custom = [{
            "name": "pcs_plan",
            "ts_col": "time",
            "columns": ["pcs_plan"],
            "categorical_columns": [],
            "future_strategy": "explicit",
            "availability": "forecast_origin",
            "df": pd.DataFrame({"time": times, "pcs_plan": np.arange(20, dtype=float)}),
        }]
        _, predictors, targets, _ = FeatureEngineer(cfg, "[test]", False).create_features(
            df_series=series,
            df_date_history=None,
            df_date_future=None,
            df_weather_history=None,
            df_weather_future=None,
            df_custom_history=custom,
            endogenous_features_with_target=["y"],
            target_feature="y",
            horizon=12,
        )
        self.assertEqual(targets, [f"y_shift_{h}" for h in range(1, 5)])
        self.assertIn("pcs_plan_h4", predictors)
        self.assertNotIn("pcs_plan_h5", predictors)


if __name__ == "__main__":
    unittest.main()
