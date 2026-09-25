# -*- coding: utf-8 -*-
"""Quantile 原生参数注入与概率 spec 身份测试。

quantile 能力拒绝的生产唯一入口是 `models/catalog.py::quantile_parameters`
（参数注入边界 RAISE）；原 `model_training/objectives.py` 包装层无生产消费者，
已于 2026-09-25 退役。
"""

import unittest
from pathlib import Path

from config.config_loader import load_yaml_config
from forecasting_core.specs import EstimatorSpec, ForecastConfigSpec
from forecasting_core.probabilistic_spec import probabilistic_spec_from_mapping
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

    def test_unsupported_model_and_invalid_level_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "does not declare scalar quantile support"):
            quantile_parameters("randomforest", {}, 0.5)
        with self.assertRaisesRegex(ValueError, "inside \(0, 1\)"):
            quantile_parameters("lightgbm", {}, 1.0)
        with self.assertRaisesRegex(ValueError, "unknown model_type"):
            quantile_parameters("unknown", {}, 0.5)

    def test_canonical_config_uses_new_spec_for_capability_and_identity(self):
        root = Path(__file__).resolve().parent.parent
        config_path = (
            root
            / "config/aidc_load_15min_short/route_A/baseline/"
            / "lgbm_direct.yaml"
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

        # quantile 能力拒绝走 catalog 参数注入边界（生产唯一入口）
        ridge = build("ridge")
        probabilistic_spec_from_mapping(ridge.probabilistic)
        with self.assertRaisesRegex(ValueError, "does not declare scalar quantile support"):
            quantile_parameters(ridge.estimator.model_type, {}, 0.5)

        lightgbm = build("lightgbm")
        lightgbm_spec = probabilistic_spec_from_mapping(lightgbm.probabilistic)
        injected = quantile_parameters(lightgbm.estimator.model_type, {}, 0.5)
        self.assertEqual(injected["objective"], "quantile")

        self.assertEqual(lightgbm_spec.mode, "quantile")
        self.assertEqual(lightgbm.probabilistic["mode"], "quantile")
        self.assertNotEqual(base.fingerprint(), lightgbm.fingerprint())


if __name__ == "__main__":
    unittest.main()
