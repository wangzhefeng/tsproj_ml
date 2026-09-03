# -*- coding: utf-8 -*-
"""Canonical runtime array selection fast-path tests."""
from __future__ import annotations

import unittest
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pandas as pd

from feature_engineering.compiler import (
    CompiledFeatures,
    FeatureSchema,
    VisibilityProof,
)
from model_forecasting.runtime import _holdout_proof_summary, _sample_selector
from model_training.estimators import make_model_factory


class SampleSelectorTest(unittest.TestCase):
    def test_contiguous_origin_range_uses_slice_for_global_rows(self) -> None:
        self.assertEqual(
            _sample_selector((2, 3, 4), n_series=2),
            slice(4, 10),
        )

    def test_noncontiguous_origins_keep_explicit_sample_indices(self) -> None:
        self.assertEqual(
            _sample_selector((1, 3), n_series=2),
            (2, 3, 6, 7),
        )

    def test_empty_origins_return_empty_indices(self) -> None:
        self.assertEqual(_sample_selector((), n_series=3), ())


class EstimatorInputFastPathTest(unittest.TestCase):
    def test_numeric_model_keeps_ndarray_input(self) -> None:
        estimator: Any = make_model_factory(
            "ridge",
            {"alpha": 1.0},
            feature_names=("lag", "hour"),
        )()
        values = np.arange(12.0).reshape(6, 2)

        self.assertIs(estimator._input(values), values)

    def test_seasonal_template_keeps_feature_names_in_dataframe(self) -> None:
        estimator: Any = make_model_factory(
            "st",
            {},
            feature_names=("load__lag_24", "dt_day_of_week"),
        )()
        values = np.arange(12.0).reshape(6, 2)

        frame = estimator._input(values)

        self.assertIsInstance(frame, pd.DataFrame)
        self.assertEqual(list(frame.columns), ["load__lag_24", "dt_day_of_week"])

    def test_seasonal_template_factory_suppresses_per_output_fit_logs(self) -> None:
        estimator: Any = make_model_factory(
            "st",
            {},
            feature_names=("load__lag_24", "dt_day_of_week"),
        )()
        step = np.arange(30.0)
        values = np.column_stack((100.0 + step, step % 7.0))
        target = values[:, 0] * 2.0

        with patch("models.ModelFactory.logger.info") as log_info:
            estimator.fit(values, target)

        log_info.assert_not_called()


    def test_all_supported_model_families_fit_fast_path_input(self) -> None:
        cases = {
            "st": {},
            "ridge": {"alpha": 1.0},
            "lasso": {"alpha": 0.01, "max_iter": 100},
            "enet": {"alpha": 0.01, "max_iter": 100},
            "lightgbm": {"n_estimators": 2, "n_jobs": 1, "verbose": -1},
            "xgboost": {"n_estimators": 2, "n_jobs": 1, "max_depth": 2},
            "catboost": {"iterations": 2, "thread_count": 1, "verbose": False},
            "randomforest": {"n_estimators": 2, "n_jobs": 1, "max_depth": 2},
            "histgb": {"max_iter": 2, "max_depth": 2},
        }
        step = np.arange(30.0)
        values = np.column_stack((100.0 + step, step % 7.0))
        target = 2.0 * values[:, 0] - values[:, 1]
        feature_names = ("load__lag_24", "dt_day_of_week")

        for model_type, params in cases.items():
            with self.subTest(model_type=model_type):
                estimator: Any = make_model_factory(
                    model_type,
                    params,
                    feature_names=feature_names,
                )()
                estimator.fit(values, target)
                prediction = np.asarray(estimator.predict(values[:3]), dtype=float)
                self.assertEqual(prediction.shape, (3,))
                self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
