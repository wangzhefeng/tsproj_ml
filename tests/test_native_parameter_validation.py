"""原生模型参数不得静默丢弃或被 wrapper 默认值覆盖。"""

import unittest
from unittest.mock import patch

import catboost as cab
import lightgbm as lgb
import numpy as np
import pandas as pd

from models.ModelFactory import ModelFactory


class NativeParameterValidationTest(unittest.TestCase):
    def test_lightgbm_unknown_parameter_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown_parameter"):
            ModelFactory().create_model("lgb", {"unknown_parameter": 1}, log_params=False)

    def test_lightgbm_native_alias_is_preserved(self):
        model = ModelFactory().create_model("lgb", {"min_data_in_leaf": 7}, log_params=False)
        self.assertEqual(model.get_params()["min_data_in_leaf"], 7)

    def test_lightgbm_validation_failure_is_not_silently_skipped(self):
        with patch.object(lgb.basic._ConfigAliases, "_get_all_param_aliases", side_effect=RuntimeError("unavailable")):
            with self.assertRaisesRegex(RuntimeError, "parameter validation"):
                ModelFactory().create_model("lgb", {}, log_params=False)

    def test_catboost_keeps_valid_num_leaves_and_explicit_seed(self):
        model = ModelFactory().create_model(
            "cat", {"num_leaves": 8, "random_state": 7, "grow_policy": "Lossguide"},
            log_params=False,
        )
        params = model.get_params()
        self.assertEqual(params.get("num_leaves", params.get("max_leaves")), 8)
        self.assertEqual(params.get("random_seed", params.get("random_state")), 7)

    def test_catboost_foreign_parameter_is_not_discarded(self):
        with self.assertRaisesRegex(ValueError, "feature_fraction"):
            ModelFactory().create_model("cat", {"feature_fraction": 0.8}, log_params=False)

    def test_catboost_conflicting_explicit_iteration_aliases_raise(self):
        with self.assertRaises(cab.CatBoostError):
            ModelFactory().create_model("cat", {"iterations": 3, "n_estimators": 4}, log_params=False)

    def test_catboost_aliases_match_native_fit(self):
        X = pd.DataFrame({"x": np.arange(20, dtype=float)})
        y = X["x"] * 2
        supplied = {"n_estimators": 3, "random_state": 7, "max_depth": 2, "thread_count": 1, "silent": True, "allow_writing_files": False}
        model = ModelFactory().create_model("cat", supplied, log_params=False)
        model.fit(X, y)
        native = cab.CatBoostRegressor(
            **supplied, learning_rate=0.05, loss_function="MAE", eval_metric="MAE",
        ).fit(X, y)
        np.testing.assert_array_equal(model.predict(X), native.predict(X))
        assert model.model is not None
        self.assertEqual(model.model.get_all_params()["random_seed"], 7)


if __name__ == "__main__":
    unittest.main()
