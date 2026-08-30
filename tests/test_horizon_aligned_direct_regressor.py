# -*- coding: utf-8 -*-
"""Direct多输出的目标horizon外生隔离测试。"""
import unittest
import tempfile

import numpy as np
import pandas as pd

from model_training.trainer import HorizonAlignedDirectRegressor


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


if __name__ == "__main__":
    unittest.main()
