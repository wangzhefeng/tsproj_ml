# -*- coding: utf-8 -*-
"""Quantile objective capability mapping tests。"""

import unittest
import tempfile
from pathlib import Path

from config.config_loader import load_yaml_config
from main import Model
from probabilistic.objectives import (
    inject_quantile_objective,
    supports_quantile_objective,
    validate_quantile_model_support,
)
from probabilistic.spec import ProbabilisticSpec


class QuantileObjectiveMappingTest(unittest.TestCase):
    def test_supported_models_receive_exact_objective_parameters(self):
        cases = {
            "lightgbm": {
                "objective": "quantile",
                "metric": "quantile",
                "alpha": 0.2,
            },
            "xgb": {
                "objective": "reg:quantileerror",
                "quantile_alpha": 0.2,
                "eval_metric": "quantile",
            },
            "catboost": {"loss_function": "Quantile:alpha=0.2"},
            "histgradientboosting": {"loss": "quantile", "quantile": 0.2},
            "qr": {"quantile": 0.2},
        }
        for model_type, expected in cases.items():
            with self.subTest(model_type=model_type):
                original = {"keep": 1}
                actual = inject_quantile_objective(model_type, original, 0.2)
                self.assertEqual(actual, {"keep": 1, **expected})
                self.assertEqual(original, {"keep": 1})
                self.assertTrue(supports_quantile_objective(model_type))

    def test_unsupported_model_and_invalid_level_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "does not support native quantile"):
            inject_quantile_objective("randomforest", {}, 0.5)
        with self.assertRaisesRegex(ValueError, "inside \(0, 1\)"):
            inject_quantile_objective("lightgbm", {}, 1.0)

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

    def test_model_constructor_uses_new_spec_for_capability_and_setting(self):
        root = Path(__file__).resolve().parent.parent
        config_path = (
            root
            / "config/aidc_power_month/route_A/freq_1month/window_length_9/"
            / "ridge_usmd_mean.yaml"
        )
        cfg = load_yaml_config(config_path)
        cfg.probabilistic = {
            "mode": "quantile",
            "quantiles": [0.1, 0.5, 0.9],
            "point_quantile": 0.5,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            cfg.checkpoints_dir = Path(temp_dir) / "models"
            cfg.test_results_dir = Path(temp_dir) / "test"
            cfg.pred_results_dir = Path(temp_dir) / "forecast"
            with self.assertRaisesRegex(ValueError, "does not support native quantile"):
                Model(cfg)

        cfg.model_type = "lightgbm"
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg.checkpoints_dir = Path(temp_dir) / "models"
            cfg.test_results_dir = Path(temp_dir) / "test"
            cfg.pred_results_dir = Path(temp_dir) / "forecast"
            model = Model(cfg)

        self.assertEqual(model.probabilistic_spec.mode, "quantile")
        self.assertEqual(cfg.predict_type, "quantile")
        self.assertTrue(model.setting.endswith("-quantile"))


if __name__ == "__main__":
    unittest.main()
