# -*- coding: utf-8 -*-
"""统一运行资源 workload/budget/execution-plan 合同。"""
from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

from forecasting_core.runtime_resources import RuntimeResourceBudget
from forecasting_core.specs import EstimatorSpec, RuntimePerformanceSpec
from model_forecasting.resource_planner import (
    build_runtime_workload,
    plan_ensemble_resources,
    plan_runtime_execution,
    runtime_budget_for_config,
)
from model_ensemble.loader import parse_ensemble_document
from tests.test_ensemble_runtime import _ensemble_doc
from tests.test_model_thread_runtime import _config


class RuntimeWorkloadTest(unittest.TestCase):
    def _workload(self, config):
        return build_runtime_workload(
            config,
            training_rows=100,
            feature_count=35,
            design_bytes=1024,
            fold_count=8,
        )

    def test_direct_pointwise_counts_one_shared_fit(self):
        config = _config(
            strategy="direct",
            horizon=8,
            direct_layout="single_model_horizon",
        )

        workload = self._workload(config)

        self.assertEqual(workload.model_group_count, 1)
        self.assertEqual(workload.logical_output_count, 1)
        self.assertEqual(workload.scalar_estimator_count, 1)
        self.assertEqual(workload.physical_fit_task_count, 1)

    def test_mimo_counts_logical_outputs_and_ridge_multi_rhs_fit(self):
        lightgbm = self._workload(_config(strategy="mimo", horizon=8))
        ridge = self._workload(
            _config(strategy="mimo", horizon=8, model_type="ridge")
        )

        self.assertEqual(lightgbm.model_group_count, 1)
        self.assertEqual(lightgbm.logical_output_count, 8)
        self.assertEqual(lightgbm.physical_fit_task_count, 8)
        self.assertEqual(ridge.scalar_estimator_count, 8)
        self.assertEqual(ridge.physical_fit_task_count, 1)

    def test_xgboost_shared_multi_quantile_does_not_multiply_physical_fits(self):
        config = _config(
            strategy="mimo",
            horizon=8,
            model_type="xgboost",
            mode="quantile",
        )
        config = replace(
            config,
            validation=replace(
                config.validation,
                performance=RuntimePerformanceSpec.from_mapping(
                    {"multi_output_n_jobs": 8}
                ),
            ),
        )

        workload = self._workload(config)

        self.assertEqual(workload.quantile_count, 3)
        self.assertEqual(workload.scalar_estimator_count, 24)
        self.assertEqual(workload.physical_fit_task_count, 8)
        self.assertEqual(workload.parallel_output_task_count, 1)

        with self.assertRaisesRegex(
            ValueError,
            "shared multi-quantile.*output parallelism",
        ):
            plan_runtime_execution(
                config,
                workload,
                budget=RuntimeResourceBudget(
                    physical_cores=8,
                    logical_cores=8,
                    total_thread_limit=8,
                    memory_limit_bytes=16 * 1024**3,
                    source="test",
                ),
            )

    def test_recmo_chain_and_native_task_counts_match_atomic_boundaries(self):
        recmo = self._workload(
            _config(
                strategy="recmo",
                horizon=8,
                output_chunk_length=2,
            )
        )
        chain_config = _config(strategy="direct", horizon=4, model_type="ridge")
        chain_config = replace(
            chain_config,
            estimator=EstimatorSpec("ridge", "regressor_chain", {}),
        )
        native_config = _config(strategy="mimo", horizon=4, model_type="ridge")
        native_config = replace(
            native_config,
            estimator=EstimatorSpec("ridge", "native", {}),
        )
        chain = self._workload(chain_config)
        native = self._workload(native_config)

        self.assertEqual(recmo.model_group_count, 1)
        self.assertEqual(recmo.logical_output_count, 2)
        self.assertEqual(recmo.parallel_output_task_count, 2)
        self.assertEqual(chain.scalar_estimator_count, 4)
        self.assertEqual(chain.parallel_output_task_count, 4)
        self.assertEqual(native.scalar_estimator_count, 0)
        self.assertEqual(native.physical_fit_task_count, 1)
        self.assertEqual(native.parallel_output_task_count, 1)

    def test_workload_is_immutable(self):
        workload = self._workload(_config(strategy="recursive"))

        with self.assertRaises(FrozenInstanceError):
            workload.horizon = 99

    def test_validation_exposes_typed_performance_contract(self):
        config = _config(
            strategy="direct",
            performance={
                "window_parallel_workers": 1,
                "multi_output_n_jobs": 2,
                "model_thread_count": 1,
            },
        )

        self.assertIsInstance(config.validation.performance, RuntimePerformanceSpec)
        self.assertEqual(config.validation.performance.multi_output_n_jobs, 2)

    def test_removed_dead_performance_fields_raise(self):
        for field, value in (
            ("ensemble_parallel_workers", 2),
            ("step_logging", True),
            ("forecast_log_interval", 10),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "Unknown fields"):
                    _config(strategy="direct", performance={field: value})


class RuntimeExecutionPlanTest(unittest.TestCase):
    @staticmethod
    def _budget(threads: int = 8) -> RuntimeResourceBudget:
        return RuntimeResourceBudget(
            physical_cores=8,
            logical_cores=8,
            total_thread_limit=threads,
            memory_limit_bytes=16 * 1024**3,
            parent_concurrency=1,
            require_determinism=True,
            source="test",
        )

    @staticmethod
    def _workload(config):
        return build_runtime_workload(
            config,
            training_rows=100,
            feature_count=35,
            design_bytes=1024,
            fold_count=8,
        )

    def test_verified_window_model_topology_resolves_within_budget(self):
        config = _config(
            strategy="recursive",
            horizon=8,
            performance={
                "window_parallel_workers": 4,
                "model_thread_count": 2,
            },
        )

        plan = plan_runtime_execution(
            config,
            self._workload(config),
            budget=self._budget(),
        )

        self.assertEqual(plan.selected_axis, "window")
        self.assertEqual(plan.window_workers, 4)
        self.assertEqual(plan.quantile_workers, 1)
        self.assertEqual(plan.output_workers, 1)
        self.assertEqual(plan.model_threads, 2)
        self.assertEqual(plan.budget_product, 8)

    def test_verified_output_topology_resolves_within_budget(self):
        config = _config(
            strategy="direct",
            horizon=8,
            performance={
                "window_parallel_workers": 1,
                "multi_output_n_jobs": 8,
                "model_thread_count": 1,
            },
        )

        plan = plan_runtime_execution(
            config,
            self._workload(config),
            budget=self._budget(),
        )

        self.assertEqual(plan.selected_axis, "output")
        self.assertEqual(plan.output_workers, 8)
        self.assertEqual(plan.model_threads, 1)
        self.assertEqual(plan.budget_product, 8)

    def test_two_explicit_outer_axes_raise_instead_of_silent_suppression(self):
        config = _config(
            strategy="direct",
            horizon=8,
            performance={
                "window_parallel_workers": 2,
                "multi_output_n_jobs": 2,
            },
        )

        with self.assertRaisesRegex(ValueError, "outer parallel axes"):
            plan_runtime_execution(
                config,
                self._workload(config),
                budget=self._budget(),
            )

    def test_explicit_plan_over_budget_raises(self):
        config = _config(
            strategy="recursive",
            horizon=8,
            performance={
                "window_parallel_workers": 4,
                "model_thread_count": 3,
            },
        )

        with self.assertRaisesRegex(ValueError, "exceeds runtime thread budget"):
            plan_runtime_execution(
                config,
                self._workload(config),
                budget=self._budget(),
            )

    def test_parent_concurrency_reduces_available_threads(self):
        budget = RuntimeResourceBudget(
            physical_cores=8,
            logical_cores=8,
            total_thread_limit=8,
            memory_limit_bytes=16 * 1024**3,
            parent_concurrency=2,
            require_determinism=True,
            source="test",
        )
        config = _config(
            strategy="recursive",
            horizon=8,
            performance={"model_thread_count": 4},
        )

        plan = plan_runtime_execution(
            config,
            self._workload(config),
            budget=budget,
        )

        self.assertEqual(plan.available_threads, 4)
        self.assertEqual(plan.model_threads, 4)
        self.assertEqual(plan.budget_product, 4)

    def test_resolved_parent_budget_is_not_annotated_twice(self):
        config = _config(
            strategy="recursive",
            model_type="lightgbm",
            performance={"total_thread_limit": 4},
        )
        workload = self._workload(config)
        resolved_once = runtime_budget_for_config(
            config,
            parent_budget=self._budget(),
        )

        plan_runtime_execution(config, workload, budget=resolved_once)
        resolved_twice = runtime_budget_for_config(
            config,
            parent_budget=resolved_once,
        )

        self.assertEqual(resolved_twice.available_threads, 4)
        self.assertEqual(resolved_twice.source, "test+validation.performance")

    def test_ensemble_uses_same_planner_contract_in_serial_before_opt015(self):
        config = parse_ensemble_document(
            _ensemble_doc("averaging"),
            source_path="ensemble.yaml",
        )
        member_workload = self._workload(_config(strategy="recursive"))
        workload, budget, plan = plan_ensemble_resources(
            config,
            {
                "m_direct": SimpleNamespace(workload=member_workload),
                "m_recursive": SimpleNamespace(workload=member_workload),
            },
            budget=self._budget(),
        )

        self.assertEqual(workload.strategy, "ensemble")
        self.assertEqual(workload.member_count, 2)
        self.assertEqual(plan.selected_axis, "serial")
        self.assertEqual(plan.member_workers, 1)
        self.assertEqual(plan.available_threads, budget.available_threads)


if __name__ == "__main__":
    unittest.main()
