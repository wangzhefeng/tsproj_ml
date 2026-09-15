# -*- coding: utf-8 -*-
"""概率预测语义契约测试。"""

import unittest
from types import SimpleNamespace

from forecasting_core.probabilistic_spec import (
    CalibrationSpec,
    IntervalSpec,
    ProbabilisticSpec,
    probabilistic_spec_from_mapping,
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
        with self.assertRaisesRegex(ValueError, "point mode forbids calibration"):
            probabilistic_spec_from_mapping({
                "mode": "point", "calibration": {"method": "cqr"},
            })

    def test_runtime_contract_rejects_invalid_asof_calibration_limits(self):
        calibration = {
            "method": "cqr", "interval": "q10_q90", "target_coverage": 0.8,
            "calibration_windows": 5, "min_windows": 3, "min_scores": 30,
            "label_availability_delay_steps": 0,
        }
        cases = [
            ({"min_windows": 0}, "min_windows must be > 0"),
            ({"label_availability_delay_steps": -1}, "label_availability_delay_steps must be >= 0"),
            ({"calibration_windows": 0}, "calibration_windows must be > 0"),
            ({"min_windows": 6}, "min_windows must be <= calibration_windows"),
            ({"min_scores": 0}, "min_scores must be > 0"),
        ]
        for override, message in cases:
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, message):
                    probabilistic_spec_from_mapping({
                        "mode": "quantile", "quantiles": [0.1, 0.5, 0.9],
                        "calibration": {**calibration, **override},
                    })


class ProbabilisticSpecResolverTest(unittest.TestCase):
    def test_canonical_quantile_config_normalizes_to_explicit_spec(self):
        # 保留原独立期望值；只将构造输入迁移到生产 mapping 入口。
        with self.assertWarnsRegex(RuntimeWarning, "target coverage differs"):
            spec = probabilistic_spec_from_mapping({
                "mode": "quantile", "quantiles": [0.1, 0.5, 0.9],
                "crossing": {"method": "rearrangement", "report_raw": True},
                "calibration": {
                    "method": "cqr", "interval": "q10_q90", "target_coverage": 0.9,
                    "calibration_windows": 5, "min_windows": 3, "min_scores": 30,
                    "label_availability_delay_steps": 2,
                },
            })

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
        spec = probabilistic_spec_from_mapping({
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
        })
        self.assertEqual(spec.mode, "quantile")
        self.assertEqual(spec.crossing_method, "median_preserving_isotonic")
        self.assertEqual(spec.intervals[0].nominal_coverage, 0.8)
        assert spec.calibration is not None
        self.assertEqual(spec.calibration.target_coverage, 0.8)

    def test_legacy_flat_keys_and_args_objects_fail_fast(self):
        for key in ("predict_type", "quantile_monotone", "enable_conformal_calibration", "conformal", "crossing_method"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "Unknown probabilistic key"):
                    probabilistic_spec_from_mapping({
                        "mode": "quantile", "quantiles": [0.1, 0.5, 0.9], key: True,
                    })
        self.assertRaises(
            TypeError, probabilistic_spec_from_mapping, SimpleNamespace(predict_type="point"),
        )

    def test_unknown_nested_key_and_point_calibration_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "Unknown probabilistic.crossing key"):
            probabilistic_spec_from_mapping({
                "mode": "quantile", "quantiles": [0.1, 0.5, 0.9],
                "point_quantile": 0.5, "crossing": {"method": "none", "typo": True},
            })
        with self.assertRaisesRegex(ValueError, "point mode forbids"):
            probabilistic_spec_from_mapping({
                "mode": "point", "calibration": {"method": "cqr"},
            })

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
