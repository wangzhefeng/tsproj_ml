# -*- coding: utf-8 -*-
"""Quantile linear blending 必须保持 (N,H,K,Q) 并学习每 target simplex 权重。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from model_ensemble.methods.linear_blending import (
    combine_linear_blending,
    fit_linear_blending,
)


class QuantileLinearBlendingTest(unittest.TestCase):
    def test_quantile_fit_and_combine_support_k2_q3(self):
        rng = np.random.default_rng(11)
        actual = rng.normal(loc=20.0, scale=2.0, size=(8, 4, 2))
        offsets = np.array([-2.0, 0.0, 2.0])[None, None, None, :]
        member_a = actual[..., None] + offsets
        member_b = actual[..., None] + 0.5 + offsets * 1.2

        artifact = fit_linear_blending(
            {"a": member_a, "b": member_b},
            actual,
            quantile_levels=(0.1, 0.5, 0.9),
        )
        combined = combine_linear_blending(
            artifact,
            {"a": member_a, "b": member_b},
        )

        self.assertEqual(combined.shape, member_a.shape)
        self.assertTrue(np.isfinite(combined).all())
        self.assertEqual(set(artifact.weights_by_target), {"target_0", "target_1"})
        self.assertEqual(artifact.quantile_levels, (0.1, 0.5, 0.9))
        self.assertEqual(
            artifact.n_samples_by_target,
            {"target_0": 32, "target_1": 32},
        )
        self.assertEqual(
            artifact.optimizer_success_by_target,
            {"target_0": True, "target_1": True},
        )
        self.assertEqual(set(artifact.optimizer_status_by_target), {"target_0", "target_1"})
        self.assertEqual(set(artifact.optimizer_message_by_target), {"target_0", "target_1"})
        self.assertEqual(
            artifact.fallback_reason_by_target,
            {"target_0": None, "target_1": None},
        )
        for weights in artifact.weights_by_target.values():
            self.assertTrue(all(weight >= 0.0 for weight in weights))
            self.assertAlmostEqual(sum(weights), 1.0, places=10)

    def test_optimizer_failure_records_fallback_reason(self):
        actual = np.arange(12, dtype=float).reshape(3, 4, 1)
        member_a = np.repeat(actual[..., None], 3, axis=-1)
        member_b = member_a + 1.0
        failed = SimpleNamespace(
            x=np.array([0.25, 0.75]),
            success=False,
            status=9,
            message="iteration limit reached",
        )

        with patch("model_ensemble.methods.linear_blending.minimize", return_value=failed):
            artifact = fit_linear_blending(
                {"a": member_a, "b": member_b},
                actual,
                quantile_levels=(0.1, 0.5, 0.9),
            )

        self.assertEqual(artifact.weights_by_target["target_0"], (0.5, 0.5))
        self.assertFalse(artifact.optimizer_success_by_target["target_0"])
        self.assertEqual(artifact.optimizer_status_by_target["target_0"], 9)
        self.assertEqual(
            artifact.fallback_reason_by_target["target_0"],
            "optimizer_failed: iteration limit reached",
        )


if __name__ == "__main__":
    unittest.main()
