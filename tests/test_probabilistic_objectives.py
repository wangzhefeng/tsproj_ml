# -*- coding: utf-8 -*-
"""Quantile objective capability mapping tests。"""

import unittest
from pathlib import Path

from config.config_loader import load_yaml_config
from forecasting_core.specs import EstimatorSpec, ForecastConfigSpec
from probabilistic.objectives import (
    supports_quantile_objective,
    validate_quantile_model_support,
)
from forecasting_core.probabilistic_spec import ProbabilisticSpec, resolve_probabilistic_spec
from models.catalog import quantile_parameters


class QuantileObjectiveMappingTest(unittest.TestCase):
    def test_supported_models_receive_exact_objective_parameters(self):
        cases = {
            "lightgbm": {
                "objective": "quantile",
                "alpha": 0.2,
            },
            "xgb": {
                "objective": "reg:quantileerror",
                "quantile_alpha": 0.2,
            },
            "catboost": {"loss_function": "Quantile:alpha=0.2"},
            "histgradientboosting": {"loss": "quantile", "quantile": 0.2},
            "qr": {"quantile": 0.2},
        }
        for model_type, expected in cases.items():
            with self.subTest(model_type=model_type):
                original = {"keep": 1}
                actual = quantile_parameters(model_type, original, 0.2)
                self.assertEqual(actual, {"keep": 1, **expected})
                self.assertEqual(original, {"keep": 1})
                self.assertTrue(supports_quantile_objective(model_type))

    def test_unsupported_model_and_invalid_level_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "does not declare scalar quantile support"):
            quantile_parameters("randomforest", {}, 0.5)
        with self.assertRaisesRegex(ValueError, "inside \(0, 1\)"):
            quantile_parameters("lightgbm", {}, 1.0)
        with self.assertRaisesRegex(ValueError, "unknown model_type"):
            quantile_parameters("unknown", {}, 0.5)
        self.assertFalse(supports_quantile_objective("unknown"))

    def test_point_spec_does_not_require_quantile_capability(self):
        point_spec = ProbabilisticSpec(
            mode="point",
            quantiles=(),
            point_quantile=0.5,
            recursive_propagation="median_path",
            crossing_method="none",
            crossing_report_raw=True,
            intervals=(),
            calibration=None,
        )

        validate_quantile_model_support("randomforest", point_spec)

    def test_quantile_spec_rejects_unsupported_model_at_construction_boundary(self):
        quantile_spec = ProbabilisticSpec(
            mode="quantile",
            quantiles=(0.1, 0.5, 0.9),
            point_quantile=0.5,
            recursive_propagation="median_path",
            crossing_method="none",
            crossing_report_raw=True,
            intervals=(),
            calibration=None,
        )

        with self.assertRaisesRegex(ValueError, "does not support native quantile"):
            validate_quantile_model_support("ridge", quantile_spec)

    def test_canonical_config_uses_new_spec_for_capability_and_identity(self):
        root = Path(__file__).resolve().parent.parent
        config_path = (
            root
            / "config/aidc_power_month/route_A/freq_1month/window_length_9/"
            / "ridge_usmd_mean.yaml"
        )
        base = load_yaml_config(config_path)
        probabilistic = {
            "mode": "quantile",
            "quantiles": [0.1, 0.5, 0.9],
            "point_quantile": 0.5,
        }

        def build(model_type):
            return ForecastConfigSpec(
                problem=base.problem,
                data=base.data,
                features=base.features,
                strategy=base.strategy,
                estimator=EstimatorSpec(
                    model_type=model_type,
                    target_adapter=base.estimator.target_adapter,
                    params=base.estimator.params,
                ),
                probabilistic=probabilistic,
                validation=base.validation,
                output=base.output,
            )

        ridge = build("ridge")
        ridge_spec = resolve_probabilistic_spec(ridge)
        with self.assertRaisesRegex(ValueError, "does not support native quantile"):
            validate_quantile_model_support(ridge.estimator.model_type, ridge_spec)

        lightgbm = build("lightgbm")
        lightgbm_spec = resolve_probabilistic_spec(lightgbm)
        validate_quantile_model_support(lightgbm.estimator.model_type, lightgbm_spec)

        self.assertEqual(lightgbm_spec.mode, "quantile")
        self.assertEqual(lightgbm.probabilistic["mode"], "quantile")
        self.assertNotEqual(base.fingerprint(), lightgbm.fingerprint())


if __name__ == "__main__":
    unittest.main()
