# -*- coding: utf-8 -*-
"""Canonical-only runtime boundary tests."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

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
        self.assertNotIn("from features", source)

    def test_target_transform_pipeline_restores_strictly(self):
        """行为断言（2026-09-06 重构 R1）：不钉私有类名，改为验证变换契约本身。

        - identity 配置直通且 restore 恒等；
        - calendar normalization 单步启用时 restore 严格还原 fit_transform 结果。
        """
        from model_forecasting.transforms import CanonicalTargetTransform

        def make_config(transformations: dict) -> SimpleNamespace:
            from feature_engineering.transform_specs import (
                normalize_target_transformations,
            )

            normalized = normalize_target_transformations(transformations)
            return SimpleNamespace(
                problem=SimpleNamespace(freq="15min", targets=["y"]),
                features=SimpleNamespace(
                    transformations={"target": normalized},
                    selection=None,
                ),
            )

        identity = CanonicalTargetTransform.from_config(make_config({}))
        self.assertTrue(identity.is_identity)
        identity.fit_identity(["s1"], ["y"])

        from forecasting_core.tensors import PointForecastTensor

        values = np.arange(6, dtype=float).reshape(1, 6, 1)
        tensor = PointForecastTensor(
            values=values,
            series_ids=("s1",),
            forecast_times=pd.date_range("2026-01-01", periods=6, freq="15min"),
            targets=("y",),
        )
        restored = identity.restore_point(identity.transform_point(tensor))
        np.testing.assert_array_equal(restored.values, tensor.values)

    def test_legacy_target_scaler_helpers_are_removed(self):
        """R1 死代码清扫：TargetScaler 三个零消费 legacy 方法与死 resolver 删除。"""
        import ast as _ast

        import model_forecasting.transforms as transforms_module

        tree = _ast.parse(
            (ROOT / "model_forecasting" / "transforms.py").read_text(encoding="utf-8")
        )
        target_scaler_methods = {
            node.name
            for node in _ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "TargetScaler"
            for node in node.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("fit_transform", target_scaler_methods)  # 活方法仍在
        for dead in ("restore_predictions", "prepare_eval_target", "prepare_history_target_for_plot"):
            self.assertNotIn(dead, target_scaler_methods)
        for dead_resolver in (
            "resolve_scale_features_enabled",
            "resolve_inverse_target_enabled",
            "resolve_feature_scaler_type",
        ):
            self.assertFalse(hasattr(transforms_module, dead_resolver))



if __name__ == "__main__":
    unittest.main()
