# -*- coding: utf-8 -*-
"""概率预测语义契约测试。"""

import unittest
from types import SimpleNamespace

from forecasting_core.probabilistic_spec import (
    CalibrationSpec,
    IntervalSpec,
    ProbabilisticSpec,
    resolve_probabilistic_spec,
    validate_cqr_params,
    validate_interval_quantiles,
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
            resolve_probabilistic_spec(args)

    def test_runtime_contract_rejects_invalid_asof_calibration_limits(self):
        args = SimpleNamespace(
            predict_type="quantile",
            quantiles=[0.1, 0.5, 0.9],
            enable_conformal_calibration=True,
            conformal_alpha=0.1,
            conformal_calibration_windows=5,
            conformal_min_windows=0,
            conformal_min_scores=30,
            conformal_label_availability_delay_steps=-1,
        )

        with self.assertRaisesRegex(ValueError, "conformal_min_windows must be > 0"):
            resolve_probabilistic_spec(args)
        args.conformal_min_windows = 1
        with self.assertRaisesRegex(
            ValueError,
            "conformal_label_availability_delay_steps must be >= 0",
        ):
            resolve_probabilistic_spec(args)


class ProbabilisticSpecResolverTest(unittest.TestCase):
    def test_legacy_quantile_config_normalizes_to_explicit_spec(self):
        args = SimpleNamespace(
            predict_type="quantile",
            quantiles=[0.1, 0.5, 0.9],
            quantile_monotone=True,
            enable_conformal_calibration=True,
            conformal_alpha=0.1,
            conformal_calibration_windows=5,
            conformal_min_windows=3,
            conformal_min_scores=30,
            conformal_label_availability_delay_steps=2,
            probabilistic={},
        )

        spec = resolve_probabilistic_spec(args)

        self.assertEqual(
            spec,
            ProbabilisticSpec(
                mode="quantile",
                quantiles=(0.1, 0.5, 0.9),
                point_quantile=0.5,
                recursive_propagation="median_path",
                crossing_method="rearrangement",
                crossing_report_raw=True,
                intervals=(IntervalSpec("q10_q90", 0.1, 0.9),),
                calibration=CalibrationSpec(
                    method="cqr",
                    interval_name="q10_q90",
                    target_coverage=0.9,
                    calibration_windows=5,
                    min_windows=3,
                    min_scores=30,
                    label_availability_delay_steps=2,
                    allow_interval_shrink=False,
                    grouping="pooled",
                ),
                schema_version=1,
            ),
        )

    def test_new_mapping_normalizes_nested_fields(self):
        args = SimpleNamespace(
            predict_type="point",
            quantiles=[0.1, 0.5, 0.9],
            quantile_monotone=False,
            enable_conformal_calibration=False,
            probabilistic={
                "mode": "quantile",
                "quantiles": [0.1, 0.5, 0.9],
                "point_quantile": 0.5,
                "crossing": {
                    "method": "median_preserving_isotonic",
                    "report_raw": True,
                },
                "intervals": [
                    {
                        "name": "p10_p90",
                        "lower_quantile": 0.1,
                        "upper_quantile": 0.9,
                    }
                ],
                "calibration": {
                    "method": "cqr",
                    "interval": "p10_p90",
                    "target_coverage": 0.8,
                    "calibration_windows": 5,
                    "min_windows": 3,
                    "min_scores": 30,
                    "label_availability_delay_steps": 0,
                    "allow_interval_shrink": False,
                    "grouping": "pooled",
                },
            },
        )

        spec = resolve_probabilistic_spec(args)

        self.assertEqual(spec.mode, "quantile")
        self.assertEqual(spec.crossing_method, "median_preserving_isotonic")
        self.assertEqual(spec.intervals[0].nominal_coverage, 0.8)
        self.assertEqual(spec.calibration.target_coverage, 0.8)

    def test_old_and_new_conflict_fails_fast(self):
        args = SimpleNamespace(
            predict_type="quantile",
            quantiles=[0.1, 0.5, 0.9],
            quantile_monotone=True,
            enable_conformal_calibration=False,
            probabilistic={
                "mode": "quantile",
                "quantiles": [0.2, 0.5, 0.8],
                "point_quantile": 0.5,
            },
        )

        with self.assertRaisesRegex(ValueError, "legacy and probabilistic config conflict"):
            resolve_probabilistic_spec(args)

    def test_unknown_nested_key_and_point_calibration_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "Unknown probabilistic.crossing key"):
            resolve_probabilistic_spec(
                SimpleNamespace(
                    probabilistic={
                        "mode": "quantile",
                        "quantiles": [0.1, 0.5, 0.9],
                        "point_quantile": 0.5,
                        "crossing": {"method": "none", "typo": True},
                    }
                )
            )
        with self.assertRaisesRegex(ValueError, "point mode forbids"):
            resolve_probabilistic_spec(
                SimpleNamespace(
                    probabilistic={
                        "mode": "point",
                        "calibration": {"method": "cqr"},
                    }
                )
            )

    def test_calibration_interval_resolves_from_canonical_spec(self):
        spec = ProbabilisticSpec(
            mode="quantile",
            quantiles=(0.05, 0.1, 0.5, 0.9, 0.95),
            point_quantile=0.5,
            recursive_propagation="median_path",
            crossing_method="none",
            crossing_report_raw=True,
            intervals=(IntervalSpec("central80", 0.1, 0.9),),
            calibration=CalibrationSpec(
                method="cqr",
                interval_name="central80",
                target_coverage=0.8,
                calibration_windows=7,
                min_windows=4,
                min_scores=40,
                label_availability_delay_steps=2,
                allow_interval_shrink=True,
                grouping="pooled",
            ),
        )

        self.assertEqual(spec.calibration_interval, IntervalSpec("central80", 0.1, 0.9))
        assert spec.calibration is not None
        self.assertEqual(spec.calibration.target_coverage, 0.8)
        self.assertEqual(spec.calibration.calibration_windows, 7)
        self.assertEqual(spec.calibration.min_windows, 4)
        self.assertEqual(spec.calibration.min_scores, 40)
        self.assertEqual(spec.calibration.label_availability_delay_steps, 2)
        self.assertTrue(spec.calibration.allow_interval_shrink)


if __name__ == "__main__":
    unittest.main()
