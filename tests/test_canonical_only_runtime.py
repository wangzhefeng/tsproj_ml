# -*- coding: utf-8 -*-
"""Canonical-only runtime boundary tests."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import run
from config.config_loader import load_yaml_config
from run import CanonicalModel, build_model


ROOT = Path(__file__).resolve().parents[1]


class CanonicalOnlyRuntimeTest(unittest.TestCase):
    def test_build_model_accepts_active_single_model_spec(self):
        config = load_yaml_config(
            ROOT
            / "config/aidc_load_15min_short/route_A/baseline/st_recursive.yaml"
        )

        self.assertIsInstance(build_model(config), CanonicalModel)

    def test_single_model_consumers_do_not_access_removed_ensemble_field(self):
        violations = []
        for relative in (
            "run.py",
            "scripts/check_model_configs.py",
            "scripts/audit_forecast_configs.py",
        ):
            tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "ensemble":
                    violations.append(f"{relative}:{node.lineno}")
        self.assertEqual(violations, [])

    def test_build_model_rejects_non_canonical_runtime_objects(self):
        legacy = SimpleNamespace(pred_method="univariate-single-multistep-direct")

        with self.assertRaisesRegex(TypeError, "only canonical ForecastConfigSpec"):
            build_model(legacy)

    def test_legacy_runtime_classes_are_not_public(self):
        import model_forecasting.forecaster
        import model_training.trainer

        self.assertFalse(hasattr(run, "Model"))
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

    def test_duplicate_probabilistic_contract_modules_are_removed(self):
        self.assertFalse((ROOT / "probabilistic" / "spec.py").exists())
        self.assertFalse((ROOT / "probabilistic" / "types.py").exists())
        self.assertTrue(
            (ROOT / "forecasting_core" / "probabilistic_spec.py").is_file()
        )
        self.assertTrue((ROOT / "forecasting_core" / "artifacts.py").is_file())
        config_source = (
            ROOT / "forecasting_core" / "specs" / "config.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("_ENSEMBLE_FIELDS", config_source)
        self.assertNotIn("_ENSEMBLE_MEMBER_FIELDS", config_source)

    def test_data_provider_and_legacy_outlier_cleaner_are_removed(self):
        """架构收敛 C5：删除 data_provider 与未接入 canonical 主链的旧清洗器。"""
        self.assertFalse((ROOT / "data_provider").exists())
        self.assertFalse((ROOT / "data_process" / "outlier_handling.py").exists())

    def test_features_package_is_removed(self):
        """架构收敛 P4（D4）：features 整包删除，目标变换并入 model_forecasting/transforms。"""
        self.assertFalse((ROOT / "features").exists())

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
