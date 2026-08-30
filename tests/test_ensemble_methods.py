# -*- coding: utf-8 -*-
"""E4: fusion method contracts (v4 §3 fixed semantics)."""

from __future__ import annotations

import unittest

import numpy as np

from model_ensemble.methods import (
    averaging,
    linear_blending,
    stacking,
    weighted,
)


def _two_member_oof():
    # member A tracks the target with slope 0.9, member B over-predicts 1.1
    rng = np.random.default_rng(7)
    actual = 50.0 + rng.normal(scale=2.0, size=(20, 3, 2))
    member_a = actual * 0.95
    member_b = actual * 1.10
    return member_a, member_b, actual


class AveragingTest(unittest.TestCase):
    def test_mean_of_members(self):
        _, _, actual = _two_member_oof()
        artifact = averaging.fit_averaging({"a": actual, "b": actual * 3}, actual)
        combined = averaging.combine_averaging(
            artifact, {"a": actual, "b": actual * 3}
        )
        np.testing.assert_allclose(combined, actual * 2.0)

    def test_requires_members(self):
        artifact = averaging.fit_averaging({}, np.zeros((1, 1, 1)))
        with self.assertRaises(ValueError):
            averaging.combine_averaging(artifact, {})


class WeightedTest(unittest.TestCase):
    def test_better_member_gets_more_weight(self):
        member_a, member_b, actual = _two_member_oof()
        artifact = weighted.fit_weighted({"a": member_a, "b": member_b}, actual)
        wa = artifact.weights_by_target["target_0"]
        self.assertGreater(wa[0], wa[1])
        for key in artifact.weights_by_target:
            self.assertAlmostEqual(sum(artifact.weights_by_target[key]), 1.0, places=12)

    def test_default_metric_rmse_and_mae_option(self):
        member_a, member_b, actual = _two_member_oof()
        self.assertEqual(
            weighted.fit_weighted({"a": member_a, "b": member_b}, actual).metric,
            "rmse",
        )
        artifact = weighted.fit_weighted(
            {"a": member_a, "b": member_b}, actual, metric="mae"
        )
        self.assertEqual(artifact.metric, "mae")

    def test_invalid_metric_raises(self):
        member_a, _, actual = _two_member_oof()
        with self.assertRaises(ValueError):
            weighted.fit_weighted({"a": member_a, "b": member_a}, actual, metric="smape")

    def test_combine_broadcasts_same_weights_across_horizon(self):
        member_a, member_b, actual = _two_member_oof()
        artifact = weighted.fit_weighted({"a": member_a, "b": member_b}, actual)
        combined = weighted.combine_weighted(
            artifact, {"a": member_a, "b": member_b}
        )
        self.assertEqual(combined.shape, actual.shape)
        # uniform target slice: weights constant across h
        w0 = artifact.weights_by_target["target_0"]
        np.testing.assert_allclose(
            combined[:, :, 0],
            w0[0] * member_a[:, :, 0] + w0[1] * member_b[:, :, 0],
        )

    def test_perfect_member_gets_full_weight(self):
        rng = np.random.default_rng(3)
        actual = 10.0 + rng.normal(scale=1.0, size=(15, 2, 1))
        perfect = actual.copy()
        bad = actual + 5.0
        artifact = weighted.fit_weighted({"p": perfect, "b": bad}, actual)
        self.assertGreater(artifact.weights_by_target["target_0"][0], 0.99)


class LinearBlendingTest(unittest.TestCase):
    def test_nonnegative_and_normalized(self):
        member_a, member_b, actual = _two_member_oof()
        artifact = linear_blending.fit_linear_blending(
            {"a": member_a, "b": member_b}, actual
        )
        for key, weights in artifact.weights_by_target.items():
            self.assertTrue(all(w >= 0.0 for w in weights), key)
            self.assertAlmostEqual(sum(weights), 1.0, places=12)

    def test_degenerate_total_uses_fallback(self):
        actual = np.ones((5, 2, 1))
        zeros = np.zeros((5, 2, 1))
        artifact = linear_blending.fit_linear_blending(
            {"a": zeros, "b": zeros * 2}, actual,
            fallback_weights={"a": 0.3, "b": 0.7},
        )
        self.assertEqual(artifact.weights_by_target["target_0"], (0.3, 0.7))

    def test_combine_matches_manual_convex_combination(self):
        member_a, member_b, actual = _two_member_oof()
        artifact = linear_blending.fit_linear_blending(
            {"a": member_a, "b": member_b}, actual
        )
        combined = linear_blending.combine_linear_blending(
            artifact, {"a": member_a, "b": member_b}
        )
        w = artifact.weights_by_target["target_1"]
        np.testing.assert_allclose(
            combined[:, :, 1],
            w[0] * member_a[:, :, 1] + w[1] * member_b[:, :, 1],
        )


class StackingTest(unittest.TestCase):
    def test_fit_and_combine_round_trip_point(self):
        member_a, member_b, actual = _two_member_oof()
        artifact = stacking.fit_stacking({"a": member_a, "b": member_b}, actual)
        self.assertFalse(artifact.fit_intercept)
        self.assertEqual(artifact.alpha, 1.0)
        combined = stacking.combine_stacking(
            artifact, {"a": member_a, "b": member_b}
        )
        self.assertEqual(combined.shape, actual.shape)
        # in-sample on training OOF should track the target far better than
        # either member alone
        err_combined = float(np.mean(np.abs(combined - actual)))
        err_a = float(np.mean(np.abs(member_a - actual)))
        self.assertLess(err_combined, err_a)

    def test_predict_with_different_order_raises(self):
        member_a, member_b, actual = _two_member_oof()
        artifact = stacking.fit_stacking({"a": member_a, "b": member_b}, actual)
        with self.assertRaises(ValueError):
            stacking.combine_stacking(
                artifact, {"b": member_b, "a": member_a}
            )

    def test_zero_variance_member_standardized_safely(self):
        actual = np.ones((6, 2, 1)) * 5.0
        flat = np.zeros((6, 2, 1))
        artifact = stacking.fit_stacking({"a": flat, "b": flat * 0.0}, actual)
        combined = stacking.combine_stacking(artifact, {"a": flat, "b": flat})
        self.assertTrue(np.isfinite(combined).all())


if __name__ == "__main__":
    unittest.main()
