# -*- coding: utf-8 -*-
"""Canonical-only runtime boundary tests."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import main
import run
from main import build_model


ROOT = Path(__file__).resolve().parents[1]


class CanonicalOnlyRuntimeTest(unittest.TestCase):
    def test_build_model_rejects_non_canonical_runtime_objects(self):
        legacy = SimpleNamespace(pred_method="univariate-single-multistep-direct")

        with self.assertRaisesRegex(TypeError, "only canonical ForecastConfigSpec"):
            build_model(legacy)

    def test_cli_overrides_reject_non_canonical_runtime_objects(self):
        legacy = SimpleNamespace(pred_method="univariate-single-multistep-direct")

        with self.assertRaisesRegex(TypeError, "only canonical ForecastConfigSpec"):
            run._apply_overrides(legacy, SimpleNamespace())

    def test_legacy_runtime_classes_are_not_public(self):
        import model_forecasting.forecaster
        import model_training.trainer

        self.assertFalse(hasattr(main, "Model"))
        self.assertFalse(hasattr(model_forecasting.forecaster, "Forecaster"))
        self.assertFalse(hasattr(model_training.trainer, "Trainer"))

    def test_models_shims_are_removed(self):
        """2026-08-30 卫生切片：models 双 shim 删除，公开实现只在 model_training.trainer/forecaster。"""
        self.assertFalse((ROOT / "models" / "ModelTraining.py").exists())
        self.assertFalse((ROOT / "models" / "ModelForecasting.py").exists())

    def test_legacy_layers_are_removed(self):
        self.assertFalse((ROOT / "config" / "generate_configs.py").exists())
        self.assertFalse((ROOT / "models" / "multistep").exists())
        self.assertFalse((ROOT / "model_forecasting" / "legacy").exists())
        self.assertFalse((ROOT / "config" / "config_sections.py").exists())
        self.assertFalse((ROOT / "config" / "univariate_config.py").exists())
        self.assertFalse((ROOT / "config" / "multivariate_config.py").exists())
        self.assertFalse((ROOT / "data_provider" / "data_loader.py").exists())
        self.assertFalse((ROOT / "features" / "FeatureEngineering.py").exists())
        self.assertFalse((ROOT / "models" / "AuxiliaryForecaster.py").exists())
        self.assertFalse((ROOT / "scripts" / "migrate_forecast_configs.py").exists())
        self.assertFalse(
            (ROOT / "scripts" / "generate_result_invalidation_manifest.py").exists()
        )

    def test_data_provider_package_is_removed(self):
        """架构收敛 P2（D2）：data_provider 整包删除，outlier_handling 迁 data_process。"""
        self.assertFalse((ROOT / "data_provider").exists())
        self.assertTrue((ROOT / "data_process" / "outlier_handling.py").exists())

    def test_features_package_is_removed(self):
        """架构收敛 P4（D4）：features 整包删除，目标变换并入 model_forecasting/transforms。"""
        self.assertFalse((ROOT / "features").exists())
        import ast

        source = (ROOT / "model_forecasting" / "transforms.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }
        self.assertLessEqual(
            {
                "TargetTransformPipeline",
                "PerSeriesTargetTransformPipeline",
                "TargetScaler",
                "CalendarDayTargetNormalizer",
                "resolve_target_scaler_type",
            },
            names,
        )
        self.assertNotIn("from features", source)


if __name__ == "__main__":
    unittest.main()
