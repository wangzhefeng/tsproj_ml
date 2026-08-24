# -*- coding: utf-8 -*-
"""概率预测语义契约测试。"""

import unittest
from types import SimpleNamespace

from probabilistic.spec import (
    validate_cqr_params,
    validate_interval_quantiles,
    validate_probabilistic_args,
    validate_quantile_grid,
)


class QuantileGridValidationTest(unittest.TestCase):
    def test_valid_quantile_grid_is_normalized_to_strict_float_tuple(self):
        levels = validate_quantile_grid([0.1, 0.5, 0.9], point_quantile=0.5)

        self.assertEqual(levels, (0.1, 0.5, 0.9))
        self.assertTrue(all(isinstance(level, float) for level in levels))

    def test_malformed_quantile_grids_fail_fast(self):
        cases = [
            ([], "must not be empty"),
            ([0.1, 0.1, 0.9], "must be unique"),
            ([0.5, 0.1, 0.9], "strictly increasing"),
            ([0.0, 0.5, 0.9], "inside \(0, 1\)"),
            ([0.1, 0.9], "point_quantile=0.5"),
        ]
        for levels, message in cases:
            with self.subTest(levels=levels):
                with self.assertRaisesRegex(ValueError, message):
                    validate_quantile_grid(levels, point_quantile=0.5)

    def test_invalid_interval_and_cqr_parameters_fail_fast(self):
        levels = (0.1, 0.5, 0.9)
        with self.assertRaisesRegex(ValueError, "lower_quantile must be < upper_quantile"):
            validate_interval_quantiles(0.9, 0.1, levels)
        with self.assertRaisesRegex(ValueError, "must be present"):
            validate_interval_quantiles(0.2, 0.9, levels)
        for alpha in (0.0, 1.0, float("nan")):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(ValueError, "alpha must be finite and inside"):
                    validate_cqr_params(alpha=alpha, min_scores=30)
        with self.assertRaisesRegex(ValueError, "min_scores must be > 0"):
            validate_cqr_params(alpha=0.1, min_scores=0)

    def test_runtime_contract_rejects_conformal_point_mode(self):
        args = SimpleNamespace(
            predict_type="point",
            quantiles=[0.1, 0.5, 0.9],
            enable_conformal_calibration=True,
            conformal_alpha=0.1,
            conformal_calibration_windows=5,
            conformal_min_scores=30,
        )

        with self.assertRaisesRegex(ValueError, "requires predict_type=quantile"):
            validate_probabilistic_args(args)


if __name__ == "__main__":
    unittest.main()
