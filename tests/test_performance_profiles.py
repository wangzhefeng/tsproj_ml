"""Adopted performance evidence is non-semantic and fails closed."""
from __future__ import annotations

import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from config.config_loader import load_yaml_config
from forecasting_core.specs.config import parse_model_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/aidc_load_15min_short/route_A/baseline/cab_direct-pointwise.yaml"
PROFILE_REF = "opt017-catboost-short-a-pointwise-v1"


class PerformanceProfileTest(unittest.TestCase):
    def _profiled_config(self, folds=31):
        payload: Any = load_yaml_config(CONFIG).canonical_payload()
        payload["validation"]["fold_count"] = folds
        payload["validation"]["performance"] = {"profile_ref": PROFILE_REF}
        return parse_model_config(payload, CONFIG)

    def test_adopted_profile_matches_real_shape_and_preserves_fold_boundary(self):
        from model_forecasting.performance_profiles import (
            adopted_catboost_profile, resolve_performance_profile,
        )
        from model_forecasting.resource_planner import build_runtime_workload
        from forecasting_core.runtime_resources import RuntimeResourceBudget

        evidence = adopted_catboost_profile()
        for folds in (5, 31):
            config = self._profiled_config(folds)
            workload = build_runtime_workload(
                config, training_rows=5072, feature_count=29, design_bytes=19476480,
            )
            result = resolve_performance_profile(
                config, workload,
                budget=RuntimeResourceBudget(
                    physical_cores=8, logical_cores=8, total_thread_limit=8,
                    memory_limit_bytes=4 * 1024**3,
                ),
                base_dir=ROOT, feature_schema=evidence["signature"]["feature_schema"],
            )
            self.assertEqual(result["kind"], "adopted_benchmark_profile")
            self.assertEqual(result["controls"]["model_thread_count"], 8)
            self.assertEqual(result["benchmark_fold_count"], 5)
            self.assertEqual(result["actual_fold_count"], folds)
            self.assertEqual(result["benchmark_geometry_match"], folds == 5)
            self.assertFalse(result["formal_benchmark_verified"])
            self.assertEqual(result["signature"], evidence["signature"])

    def _resolve(self, *, config=None, workload_changes=None, budget_changes=None, schema=None, base_dir: str | Path = ROOT):
        from model_forecasting.performance_profiles import adopted_catboost_profile, resolve_performance_profile
        from model_forecasting.resource_planner import build_runtime_workload
        from forecasting_core.runtime_resources import RuntimeResourceBudget

        config = config or self._profiled_config()
        workload = build_runtime_workload(config, training_rows=5072, feature_count=29, design_bytes=19476480)
        budget = RuntimeResourceBudget(physical_cores=8, logical_cores=8, total_thread_limit=8, memory_limit_bytes=4 * 1024**3)
        return resolve_performance_profile(
            config, replace(workload, **(workload_changes or {})),
            budget=replace(budget, **(budget_changes or {})), base_dir=base_dir,
            feature_schema=schema if schema is not None else adopted_catboost_profile()["signature"]["feature_schema"],
        )

    def test_profile_rejects_changed_shape_and_topology(self):
        for field, value in {
            "horizon": 96, "target_count": 2, "quantile_count": 3,
            "training_rows": 5073, "feature_count": 30, "design_bytes": 19476481,
            "series_count": 2, "member_count": 2, "adapter": "native",
            "model_group_count": 2, "physical_fit_task_count": 2,
            "strategy": "recursive", "direct_layout": "horizon_specific",
            "fold_count": 7, "total_fit_task_count": 33,
        }.items():
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "signature mismatch"):
                self._resolve(workload_changes={field: value})

    def test_profile_rejects_semantic_changes_even_with_unchanged_width(self):
        from model_forecasting.performance_profiles import adopted_catboost_profile
        original: Any = self._profiled_config().canonical_payload()
        mutations = [
            ("problem", "training_scope", "global"),
            ("validation", "train_window_steps", 1425),
            ("validation", "forecast_origin", "2026-07-30T14:00:00"),
            ("estimator", "params", {"depth": 7}),
        ]
        for section, key, value in mutations:
            payload: Any = deepcopy(original)
            payload[section][key] = value
            if key == "training_scope":
                # A Global series key is required by the public spec.
                payload["problem"]["series_id_cols"] = ["site"]
                payload["data"]["sources"][0]["series_id_cols"] = ["site"]
                payload["data"]["sources"][0]["columns"].append({"name": "site", "role": "key", "categorical": True})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "signature mismatch"):
                self._resolve(config=parse_model_config(payload, CONFIG))
        schema = adopted_catboost_profile()["signature"]["feature_schema"]
        schema[0] = "different_feature_same_width"
        with self.assertRaisesRegex(ValueError, "feature_schema"):
            self._resolve(schema=schema)

    def test_profile_rejects_data_and_estimator_version_drift(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from model_forecasting.performance_profiles import adopted_catboost_profile

        with TemporaryDirectory() as directory:
            relative = self._profiled_config().data.sources[0].history_path
            assert relative is not None
            path = Path(directory) / relative
            path.parent.mkdir(parents=True)
            path.write_text("time,value\n2026-07-31,999\n")
            with self.assertRaisesRegex(ValueError, "source_sha256"):
                self._resolve(base_dir=directory)
        with patch("model_forecasting.performance_profiles.version", return_value="1.2.11"):
            with self.assertRaisesRegex(ValueError, "estimator_versions"):
                self._resolve()
        defaults = adopted_catboost_profile()["signature"]["estimator_defaults"]
        defaults["depth"] = 7
        with patch("model_forecasting.performance_profiles.CatBoostModel.DEFAULT_PARAMS", defaults):
            with self.assertRaisesRegex(ValueError, "estimator_defaults"):
                self._resolve()

    def test_profile_rejects_unmeasured_budget_and_conflicting_controls(self):
        for changes in (
            {"parent_concurrency": 2}, {"physical_cores": 4},
            {"physical_cores": 16, "logical_cores": 16},
            {"logical_cores": 16}, {"total_thread_limit": 4},
            {"memory_limit_bytes": 451936255}, {"require_determinism": False},
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "profile.*budget"):
                self._resolve(budget_changes=changes)
        for field, value in (("model_thread_count", 4), ("window_parallel_workers", 2), ("multi_output_n_jobs", 2), ("quantile_parallel_workers", 2), ("ensemble_parallel_workers", 1)):
            payload: Any = self._profiled_config().canonical_payload()
            payload["validation"]["performance"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "profile.*conflict"):
                self._resolve(config=parse_model_config(payload, CONFIG))

    def test_explicit_override_is_not_a_benchmark_claim(self):
        payload: Any = self._profiled_config().canonical_payload()
        payload["validation"]["performance"] = {"model_thread_count": 4}
        result = self._resolve(config=parse_model_config(payload, CONFIG), base_dir="/does-not-exist")
        self.assertEqual(result["kind"], "explicit_performance_override")
        self.assertIsNone(result["profile_ref"])
        self.assertNotIn("signature", result)
        payload["validation"].pop("performance")
        result = self._resolve(config=parse_model_config(payload, CONFIG), base_dir="/does-not-exist")
        self.assertEqual(result["kind"], "automatic_default")

    def test_planner_enforces_reference_and_records_distinct_source(self):
        from model_forecasting.performance_profiles import adopted_catboost_profile
        from model_forecasting.resource_planner import build_runtime_workload, plan_runtime_execution
        from forecasting_core.runtime_resources import RuntimeResourceBudget

        config = self._profiled_config()
        workload = build_runtime_workload(config, training_rows=5072, feature_count=29, design_bytes=19476480)
        budget = RuntimeResourceBudget(physical_cores=8, logical_cores=8, total_thread_limit=8, memory_limit_bytes=4 * 1024**3)
        context: dict[str, Any] = {"base_dir": ROOT, "feature_schema": adopted_catboost_profile()["signature"]["feature_schema"]}
        plan = plan_runtime_execution(config, workload, budget=budget, **context)
        self.assertEqual(plan.model_threads, 8)
        self.assertEqual(plan.selected_axis, "serial")
        self.assertEqual(plan.profile_source, f"benchmark_profile:{PROFILE_REF}")
        with self.assertRaisesRegex(ValueError, "signature mismatch"):
            plan_runtime_execution(config, replace(workload, training_rows=5000), budget=budget, **context)
        with self.assertRaisesRegex(ValueError, "requires actual"):
            plan_runtime_execution(config, workload, budget=budget)
        payload: Any = config.canonical_payload()
        payload["validation"]["performance"] = {"model_thread_count": 8}
        override = plan_runtime_execution(parse_model_config(payload, CONFIG), workload, budget=budget)
        self.assertEqual(override.model_threads, 8)
        self.assertEqual(override.profile_source, "validation.performance")
        payload["validation"]["performance"] = {"profile_ref": "unknown-profile"}
        with self.assertRaisesRegex(ValueError, "unknown performance profile_ref"):
            plan_runtime_execution(parse_model_config(payload, CONFIG), workload, budget=budget, **context)

    def test_metadata_rejects_a_plan_replaced_after_profile_resolution(self):
        from model_forecasting.performance_profiles import adopted_catboost_profile, resolve_performance_profile
        from model_forecasting.resource_planner import build_runtime_workload, plan_runtime_execution
        from forecasting_core.runtime_resources import RuntimeResourceBudget

        config = self._profiled_config()
        workload = build_runtime_workload(config, training_rows=5072, feature_count=29, design_bytes=19476480)
        budget = RuntimeResourceBudget(physical_cores=8, logical_cores=8, total_thread_limit=8, memory_limit_bytes=4 * 1024**3)
        context: dict[str, Any] = {"budget": budget, "base_dir": ROOT, "feature_schema": adopted_catboost_profile()["signature"]["feature_schema"]}
        plan = plan_runtime_execution(config, workload, **context)
        result = resolve_performance_profile(config, workload, execution_plan=plan, **context)
        self.assertEqual(result["resolved_execution_plan"], plan.payload())
        changed = replace(plan, model_threads=4, budget_product=4)
        with self.assertRaisesRegex(ValueError, "profile.*execution plan"):
            resolve_performance_profile(config, workload, execution_plan=changed, **context)

    def test_typed_profile_reference_is_nonsemantic(self):
        config = load_yaml_config(CONFIG)
        payload: Any = config.canonical_payload()
        payload["validation"]["performance"] = {"profile_ref": PROFILE_REF}
        profiled = parse_model_config(payload, CONFIG)
        assert profiled.validation.performance is not None
        self.assertEqual(profiled.validation.performance.profile_ref, PROFILE_REF)
        self.assertEqual(profiled.fingerprint(), config.fingerprint())
        for controls in ({}, {"model_thread_count": 8}, {"profile_ref": "different-evidence-id"}):
            comparison = deepcopy(payload)
            comparison["validation"]["performance"] = controls
            self.assertEqual(profiled.fingerprint(), parse_model_config(comparison, CONFIG).fingerprint())
        for invalid in (None, "", "   ", 1, True, {}, []):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                payload["validation"]["performance"]["profile_ref"] = invalid
                parse_model_config(payload, CONFIG)


if __name__ == "__main__":
    unittest.main()
