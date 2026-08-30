# -*- coding: utf-8 -*-
"""Canonical combination contracts on the reference-based ensemble (v4)."""

import unittest

import numpy as np
import pandas as pd

from model_ensemble.methods import averaging
from model_ensemble.methods.linear_blending import fit_nonnegative_stacking_weights
from model_forecasting.specs.strategy import ForecastStrategySpec


class CanonicalCombinationContractTest(unittest.TestCase):
    def test_canonical_point_combination_uses_member_weights(self):
        from model_forecasting.tensors import PointForecastTensor

        times = pd.date_range("2026-01-01", periods=1, freq="1h")
        a = PointForecastTensor(
            values=np.array([[[10.0]]]),
            series_ids=("s",),
            forecast_times=times,
            targets=("load",),
        )
        b = PointForecastTensor(
            values=np.array([[[30.0]]]),
            series_ids=("s",),
            forecast_times=times,
            targets=("load",),
        )
        from model_ensemble.artifacts import EqualWeightsArtifact

        values = averaging.combine_averaging(
            EqualWeightsArtifact(),
            member_values={"a": a.values, "b": b.values},
        )
        self.assertEqual(float(values[0, 0, 0]), 20.0)

    def test_blend_is_not_a_canonical_strategy_name(self):
        with self.assertRaises(ValueError):
            ForecastStrategySpec("blend")


class NNLSFunctionContractTest(unittest.TestCase):
    def test_nnls_weights_nonnegative_and_normalized(self):
        y = np.array([[10.0, 12.0], [20.0, 18.0]])
        e = np.array([[1.0, -1.0], [1.0, -1.0]])
        weights = fit_nonnegative_stacking_weights(
            {"a": y + e, "b": y - e},
            y,
            fallback_weights={"a": 0.5, "b": 0.5},
        )
        self.assertTrue(all(w >= 0.0 for w in weights))
        self.assertAlmostEqual(sum(weights), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
