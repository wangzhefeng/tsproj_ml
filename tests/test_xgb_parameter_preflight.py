"""XGBoost 原生配置预检在真实拟合之前，告警捕获不得污染父线程。"""

import unittest
import warnings
import pickle
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pandas as pd
import xgboost as xgb

from models.wrappers.xgboost import XGBoostModel
from models import xgb_validation


class XGBParameterPreflightTest(unittest.TestCase):
    def test_unknown_parameter_rejected_before_fit_even_when_validation_disabled(self):
        X = pd.DataFrame(np.arange(24, dtype=float).reshape(12, 2))
        y = X[0] / 2
        model = XGBoostModel(
            {"n_estimators": 1, "n_jobs": 1, "unknown_parameter": 1, "validate_parameters": False},
            log_params=False,
        )
        with self.assertRaisesRegex(ValueError, "unknown_parameter"):
            model.fit(X, y)
        self.assertFalse(model.is_fitted)

    def test_valid_native_configuration_fits_without_parent_warning_changes(self):
        X = pd.DataFrame(np.arange(24, dtype=float).reshape(12, 2))
        y = X[0] / 2
        filters = list(warnings.filters)
        showwarning = warnings.showwarning
        model = XGBoostModel(
            {"n_estimators": 2, "n_jobs": 1, "max_depth": 2}, log_params=False,
        )
        model.fit(X, y)
        self.assertEqual(warnings.filters, filters)
        self.assertIs(warnings.showwarning, showwarning)
        self.assertTrue(np.isfinite(model.predict(X)).all())
        self.assertEqual(model.parameter_validation["status"], "validated")
        self.assertEqual(model.parameter_validation["num_features"], 2)
        restored = pickle.loads(pickle.dumps(model))
        with patch.object(xgb_validation.subprocess, "run", side_effect=AssertionError("deployment must not preflight")):
            np.testing.assert_array_equal(restored.predict(X), model.predict(X))
        self.assertEqual(restored.parameter_validation["fitted_config"], model.parameter_validation["fitted_config"])

    def test_silent_mode_cannot_hide_unused_parameters(self):
        with self.assertRaisesRegex(ValueError, "unknown_parameter"):
            xgb_validation.validate_xgb_parameters(
                {"unknown_parameter": 1, "verbosity": 0, "n_jobs": 1}, num_features=2,
            )

    def test_preflight_does_not_advance_random_generator(self):
        X = pd.DataFrame(np.arange(24, dtype=float).reshape(12, 2))
        y = X[0] / 2
        model = XGBoostModel(
            {"n_estimators": 2, "n_jobs": 1, "random_state": np.random.default_rng(7)},
            log_params=False,
        )
        native_params = dict(model.params)
        native_params["random_state"] = np.random.default_rng(7)
        native = xgb.XGBRegressor(**native_params).fit(X, y)
        model.fit(X, y)
        assert isinstance(model.model.random_state, np.random.Generator)
        assert isinstance(native.random_state, np.random.Generator)
        self.assertEqual(model.model.random_state.bit_generator.state, native.random_state.bit_generator.state)
        np.testing.assert_array_equal(model.predict(X), native.predict(X))

    def test_parallel_identical_preflight_reuses_one_subprocess(self):
        xgb_validation._cached_preflight.cache_clear()
        with patch.object(xgb_validation.subprocess, "run", wraps=xgb_validation.subprocess.run) as run:
            with ThreadPoolExecutor(max_workers=3) as pool:
                results = list(pool.map(
                    lambda _: xgb_validation.validate_xgb_parameters(
                        {"n_jobs": 1, "max_depth": 3}, num_features=2,
                    ), range(3),
                ))
            self.assertEqual(run.call_count, 1)
            xgb_validation.validate_xgb_parameters({"n_jobs": 1, "max_depth": 3}, num_features=3)
            self.assertEqual(run.call_count, 2)
        self.assertEqual(results[0], results[1])
        self.assertIsNot(results[0], results[1])

    def test_preflight_process_failure_is_not_silently_skipped(self):
        xgb_validation._cached_preflight.cache_clear()
        with patch.object(xgb_validation.subprocess, "run", side_effect=OSError("cannot start")):
            with self.assertRaisesRegex(OSError, "cannot start"):
                xgb_validation.validate_xgb_parameters({"n_jobs": 1}, num_features=2)


if __name__ == "__main__":
    unittest.main()
