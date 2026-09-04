"""Single source of truth for canonical runtime resource planning."""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from forecasting_core.runtime_resources import (
    RuntimeExecutionPlan,
    RuntimeResourceBudget,
    RuntimeWorkload,
)
from forecasting_core.specs import ForecastConfigSpec, TargetAdapter
from model_training.estimators import supports_native_multi_quantile
from model_training.strategies import target_plan_for_config


_THREAD_PARAM_BY_MODEL = {
    "lgb": "n_jobs",
    "lightgbm": "n_jobs",
    "xgb": "n_jobs",
    "xgboost": "n_jobs",
    "rf": "n_jobs",
    "randomforest": "n_jobs",
    "cat": "thread_count",
    "catboost": "thread_count",
}

_DEFAULT_OUTPUT_WORKERS_BY_MODEL = {
    "lgb": 4,
    "lightgbm": 4,
    "xgb": 4,
    "xgboost": 4,
    "cat": 2,
    "catboost": 2,
    "rf": 2,
    "randomforest": 2,
    "ridge": 4,
    "enet": 4,
    "elasticnet": 4,
    "lasso": 4,
    "qr": 2,
    "quantileregressor": 2,
}


def _positive_performance_int(
    performance: Mapping[str, Any],
    field: str,
) -> int | None:
    value = performance.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"validation.performance.{field} must be positive")
    return value


def _performance(config: ForecastConfigSpec) -> Mapping[str, Any]:
    value = config.validation.performance
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("validation.performance must be a mapping")
    return value


def _quantile_count(config: ForecastConfigSpec) -> int:
    if str(config.probabilistic.get("mode", "point")) != "quantile":
        return 1
    levels = config.probabilistic.get("quantiles", ())
    if isinstance(levels, (str, bytes)) or not isinstance(levels, Sequence):
        raise TypeError("probabilistic.quantiles must be a sequence")
    if not levels:
        raise ValueError("quantile workload requires a nonempty quantile grid")
    return len(levels)


def build_runtime_workload(
    config: ForecastConfigSpec,
    *,
    training_rows: int,
    feature_count: int,
    design_bytes: int,
    fold_count: int | None = None,
    series_count: int = 1,
    member_count: int = 1,
    dynamic_horizons: Sequence[int] = (),
) -> RuntimeWorkload:
    """Resolve the real logical and physical fit topology for one config."""
    if not isinstance(config, ForecastConfigSpec):
        raise TypeError("config must be a ForecastConfigSpec")
    if config.strategy is None:
        raise ValueError("runtime workload requires a strategy")
    plan = target_plan_for_config(config)
    output_width = len(plan.call_coordinates[0])
    logical_outputs = plan.model_count * output_width
    quantile_count = _quantile_count(config)
    adapter = config.estimator.target_adapter
    normalized_model = config.estimator.model_type.strip().lower()

    scalar_estimators = (
        logical_outputs * quantile_count
        if adapter in {TargetAdapter.INDEPENDENT, TargetAdapter.REGRESSOR_CHAIN}
        else 0
    )
    if adapter is TargetAdapter.INDEPENDENT:
        if normalized_model == "ridge" and quantile_count == 1:
            physical_fits = plan.model_count
        elif (
            quantile_count > 1
            and supports_native_multi_quantile(normalized_model)
        ):
            physical_fits = logical_outputs
        else:
            physical_fits = logical_outputs * quantile_count
    elif adapter is TargetAdapter.REGRESSOR_CHAIN:
        physical_fits = logical_outputs * quantile_count
    else:
        physical_fits = plan.model_count * quantile_count
    shared_multi_quantile = (
        quantile_count > 1
        and supports_native_multi_quantile(normalized_model)
    )
    parallel_output_tasks = (
        1
        if shared_multi_quantile
        else (
            logical_outputs
            if adapter is TargetAdapter.INDEPENDENT
            else plan.model_count
        )
    )

    resolved_fold_count = (
        config.validation.backtest.fold_count
        if fold_count is None and config.validation.backtest is not None
        else fold_count
    )
    if resolved_fold_count is None:
        resolved_fold_count = 1
    direct = config.features.transformations.get("direct")
    direct_layout = (
        str(direct.get("layout"))
        if isinstance(direct, Mapping) and direct.get("layout") is not None
        else None
    )
    return RuntimeWorkload(
        strategy=config.strategy.name.value,
        direct_layout=direct_layout,
        horizon=config.problem.horizon,
        chunk_length=config.strategy.output_chunk_length,
        target_count=len(config.problem.targets),
        series_count=series_count,
        quantile_count=quantile_count,
        fold_count=int(resolved_fold_count),
        member_count=member_count,
        model_group_count=plan.model_count,
        logical_output_count=logical_outputs,
        scalar_estimator_count=scalar_estimators,
        physical_fit_task_count=physical_fits,
        parallel_output_task_count=parallel_output_tasks,
        total_fit_task_count=physical_fits * (int(resolved_fold_count) + 1),
        training_rows=training_rows,
        feature_count=feature_count,
        design_bytes=design_bytes,
        adapter=adapter.value,
        estimator=normalized_model,
        dynamic_horizons=tuple(int(value) for value in dynamic_horizons),
    )


def detect_runtime_budget(
    *,
    total_thread_limit: int | None = None,
    memory_limit_bytes: int | None = None,
    parent_concurrency: int = 1,
) -> RuntimeResourceBudget:
    """Detect a conservative stdlib budget; psutil enrichment is optional."""
    logical = os.cpu_count() or 1
    physical = logical
    detected_memory = memory_limit_bytes
    source = "stdlib"
    try:
        import psutil
    except ImportError:
        pass
    else:
        physical = psutil.cpu_count(logical=False) or logical
        logical = psutil.cpu_count(logical=True) or logical
        if detected_memory is None:
            detected_memory = int(psutil.virtual_memory().available)
        source = "psutil"
    if detected_memory is None:
        detected_memory = 1
    return RuntimeResourceBudget(
        physical_cores=int(physical),
        logical_cores=int(logical),
        total_thread_limit=(
            int(total_thread_limit)
            if total_thread_limit is not None
            else int(physical)
        ),
        memory_limit_bytes=int(detected_memory),
        parent_concurrency=parent_concurrency,
        require_determinism=True,
        source=source,
    )


def runtime_budget_for_config(
    config: ForecastConfigSpec,
    *,
    parent_budget: RuntimeResourceBudget | None = None,
) -> RuntimeResourceBudget:
    """Apply optional config limits without widening a parent budget."""
    performance = _performance(config)
    configured_threads = _positive_performance_int(
        performance, "total_thread_limit"
    )
    configured_memory = _positive_performance_int(
        performance, "memory_limit_bytes"
    )
    if parent_budget is None:
        return detect_runtime_budget(
            total_thread_limit=configured_threads,
            memory_limit_bytes=configured_memory,
        )
    has_configured_budget = (
        configured_threads is not None or configured_memory is not None
    )
    performance_marker = "+validation.performance"
    return replace(
        parent_budget,
        total_thread_limit=(
            min(parent_budget.total_thread_limit, configured_threads)
            if configured_threads is not None
            else parent_budget.total_thread_limit
        ),
        memory_limit_bytes=(
            min(parent_budget.memory_limit_bytes, configured_memory)
            if configured_memory is not None
            else parent_budget.memory_limit_bytes
        ),
        source=(
            f"{parent_budget.source}+validation.performance"
            if has_configured_budget
            and not parent_budget.source.endswith(performance_marker)
            else parent_budget.source
        ),
    )


def plan_runtime_execution(
    config: ForecastConfigSpec,
    workload: RuntimeWorkload,
    *,
    budget: RuntimeResourceBudget | None = None,
) -> RuntimeExecutionPlan:
    """Resolve one non-nested execution plan and reject explicit conflicts."""
    if not isinstance(config, ForecastConfigSpec):
        raise TypeError("config must be a ForecastConfigSpec")
    if not isinstance(workload, RuntimeWorkload):
        raise TypeError("workload must be a RuntimeWorkload")
    performance = _performance(config)
    configured_window = _positive_performance_int(
        performance, "window_parallel_workers"
    )
    configured_output = _positive_performance_int(
        performance, "multi_output_n_jobs"
    )
    configured_quantile = _positive_performance_int(
        performance, "quantile_parallel_workers"
    )
    configured_model = _positive_performance_int(
        performance, "model_thread_count"
    )
    budget = runtime_budget_for_config(config, parent_budget=budget)
    available = budget.available_threads
    if workload.design_bytes > budget.memory_limit_bytes:
        raise ValueError(
            "runtime design estimate exceeds memory budget: "
            f"design_bytes={workload.design_bytes}, "
            f"memory_limit_bytes={budget.memory_limit_bytes}"
        )

    explicit_outer = {
        "window": configured_window,
        "quantile": configured_quantile,
        "output": configured_output,
    }
    selected_explicit = [
        name for name, value in explicit_outer.items() if value is not None and value > 1
    ]
    if len(selected_explicit) > 1:
        raise ValueError(
            "validation.performance declares multiple outer parallel axes: "
            f"{selected_explicit}"
        )

    max_window = max(1, workload.fold_count)
    max_output = max(1, workload.parallel_output_task_count)
    native_quantile = (
        workload.quantile_count > 1
        and supports_native_multi_quantile(config.estimator.model_type)
    )
    if native_quantile and configured_output is not None and configured_output > 1:
        raise ValueError(
            "native shared multi-quantile does not allow output parallelism"
        )
    max_quantile = 1 if native_quantile else workload.quantile_count
    fallbacks: list[str] = []

    window_workers = 1
    quantile_workers = 1
    output_workers = 1
    selected_axis = "serial"
    profile_source = "automatic_default"

    if selected_explicit:
        selected_axis = selected_explicit[0]
        profile_source = "validation.performance"
        if selected_axis == "window":
            assert configured_window is not None
            window_workers = min(configured_window, max_window)
            if window_workers != configured_window:
                fallbacks.append("window_workers_clamped_to_task_count")
        elif selected_axis == "quantile":
            assert configured_quantile is not None
            if native_quantile:
                raise ValueError(
                    "native shared multi-quantile does not allow quantile-level parallelism"
                )
            quantile_workers = min(configured_quantile, max_quantile)
            if quantile_workers != configured_quantile:
                fallbacks.append("quantile_workers_clamped_to_task_count")
        else:
            assert configured_output is not None
            output_workers = min(configured_output, max_output)
            if output_workers != configured_output:
                fallbacks.append("output_workers_clamped_to_task_count")
    elif workload.quantile_count > 1 and max_output == 1 and max_quantile > 1:
        selected_axis = "quantile"
        quantile_workers = min(4, max_quantile)
    elif max_output > 1:
        selected_axis = "output"
        output_workers = min(
            _DEFAULT_OUTPUT_WORKERS_BY_MODEL.get(workload.estimator, 1),
            max_output,
        )
        if output_workers == 1:
            selected_axis = "serial"

    normalized_model = config.estimator.model_type.strip().lower()
    thread_param = _THREAD_PARAM_BY_MODEL.get(normalized_model)
    if configured_model is not None and thread_param is None:
        raise ValueError(
            "validation.performance.model_thread_count is unsupported for "
            f"model_type {normalized_model!r}"
        )
    if configured_model is not None:
        model_threads = configured_model
        profile_source = "validation.performance"
    elif selected_axis == "serial" and thread_param is not None:
        model_threads = available
    else:
        model_threads = 1

    outer_workers = max(
        window_workers,
        quantile_workers,
        output_workers,
    )
    budget_product = outer_workers * model_threads
    explicit_resource_request = bool(selected_explicit or configured_model is not None)
    if budget_product > available:
        if explicit_resource_request:
            raise ValueError(
                "resolved execution plan exceeds runtime thread budget: "
                f"product={budget_product}, available={available}"
            )
        if selected_axis != "serial":
            clamped = max(1, available // model_threads)
            if selected_axis == "window":
                window_workers = min(window_workers, clamped)
            elif selected_axis == "quantile":
                quantile_workers = min(quantile_workers, clamped)
            elif selected_axis == "output":
                output_workers = min(output_workers, clamped)
            fallbacks.append("automatic_outer_workers_clamped_to_budget")
        else:
            model_threads = available
            fallbacks.append("automatic_model_threads_clamped_to_budget")
        outer_workers = max(window_workers, quantile_workers, output_workers)
        budget_product = outer_workers * model_threads

    task_count = {
        "window": max_window,
        "quantile": max_quantile,
        "output": max_output,
    }.get(selected_axis, 1)
    return RuntimeExecutionPlan(
        schema_version=1,
        selected_axis=selected_axis,
        config_workers=1,
        member_workers=1,
        window_workers=window_workers,
        quantile_workers=quantile_workers,
        output_workers=output_workers,
        model_threads=model_threads,
        available_threads=available,
        budget_product=budget_product,
        task_count=task_count,
        profile_source=profile_source,
        fallback_reasons=tuple(fallbacks),
        rejection_reasons=(),
    )


def plan_ensemble_resources(
    config: Any,
    runners: Mapping[str, Any],
    *,
    budget: RuntimeResourceBudget | None = None,
) -> tuple[RuntimeWorkload, RuntimeResourceBudget, RuntimeExecutionPlan]:
    """Resolve the serial Ensemble plan used before OPT-015 member parallelism."""
    if not runners:
        raise ValueError("ensemble resource planning requires member runners")
    member_workloads = tuple(runner.workload for runner in runners.values())
    if not all(isinstance(item, RuntimeWorkload) for item in member_workloads):
        raise TypeError("ensemble runners must expose RuntimeWorkload")
    quantile_count = (
        len(tuple(config.probabilistic.get("quantiles", ())))
        if str(config.probabilistic.get("mode", "point")) == "quantile"
        else 1
    )
    physical_per_stage = sum(
        item.physical_fit_task_count for item in member_workloads
    )
    workload = RuntimeWorkload(
        strategy="ensemble",
        direct_layout=None,
        horizon=int(config.problem.horizon),
        chunk_length=None,
        target_count=len(config.problem.targets),
        series_count=member_workloads[0].series_count,
        quantile_count=quantile_count,
        fold_count=int(config.oof.fold_count),
        member_count=len(member_workloads),
        model_group_count=sum(item.model_group_count for item in member_workloads),
        logical_output_count=sum(
            item.logical_output_count for item in member_workloads
        ),
        scalar_estimator_count=sum(
            item.scalar_estimator_count for item in member_workloads
        ),
        physical_fit_task_count=physical_per_stage,
        parallel_output_task_count=sum(
            item.parallel_output_task_count for item in member_workloads
        ),
        total_fit_task_count=physical_per_stage * (int(config.oof.fold_count) + 1),
        training_rows=sum(item.training_rows for item in member_workloads),
        feature_count=sum(item.feature_count for item in member_workloads),
        design_bytes=sum(item.design_bytes for item in member_workloads),
        adapter="member_defined",
        estimator="ensemble",
    )
    resolved_budget = runtime_budget_for_config(
        config,
        parent_budget=budget,
    )
    if workload.design_bytes > resolved_budget.memory_limit_bytes:
        raise ValueError(
            "ensemble design estimate exceeds memory budget: "
            f"design_bytes={workload.design_bytes}, "
            f"memory_limit_bytes={resolved_budget.memory_limit_bytes}"
        )
    execution_plan = RuntimeExecutionPlan(
        schema_version=1,
        selected_axis="serial",
        config_workers=1,
        member_workers=1,
        window_workers=1,
        quantile_workers=1,
        output_workers=1,
        model_threads=1,
        available_threads=resolved_budget.available_threads,
        budget_product=1,
        task_count=len(member_workloads),
        profile_source="ensemble_serial_pre_opt015",
    )
    return workload, resolved_budget, execution_plan


def runtime_estimator_params(
    config: ForecastConfigSpec,
    plan: RuntimeExecutionPlan,
) -> dict[str, Any]:
    """Apply the planner's model-thread decision to estimator parameters."""
    params = dict(config.estimator.params)
    normalized_model = config.estimator.model_type.strip().lower()
    thread_param = _THREAD_PARAM_BY_MODEL.get(normalized_model)
    if thread_param is None:
        return params
    existing = params.get(thread_param)
    if existing is not None and existing != plan.model_threads:
        raise ValueError(
            f"estimator.params.{thread_param} conflicts with runtime execution plan"
        )
    params[thread_param] = plan.model_threads
    return params


__all__ = [
    "build_runtime_workload",
    "detect_runtime_budget",
    "plan_ensemble_resources",
    "plan_runtime_execution",
    "runtime_budget_for_config",
    "runtime_estimator_params",
]
