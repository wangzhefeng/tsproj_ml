"""Immutable contracts for runtime workload and resource planning."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeWorkload:
    """Resolved model work shape, independent of execution policy."""

    strategy: str
    direct_layout: str | None
    horizon: int
    chunk_length: int | None
    target_count: int
    series_count: int
    quantile_count: int
    fold_count: int
    member_count: int
    model_group_count: int
    logical_output_count: int
    scalar_estimator_count: int
    physical_fit_task_count: int
    parallel_output_task_count: int
    total_fit_task_count: int
    training_rows: int
    feature_count: int
    design_bytes: int
    adapter: str
    estimator: str
    dynamic_horizons: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.strategy:
            raise ValueError("strategy must not be blank")
        if not self.adapter:
            raise ValueError("adapter must not be blank")
        if not self.estimator:
            raise ValueError("estimator must not be blank")
        for name in (
            "horizon",
            "target_count",
            "series_count",
            "quantile_count",
            "fold_count",
            "member_count",
            "model_group_count",
            "logical_output_count",
            "physical_fit_task_count",
            "parallel_output_task_count",
            "total_fit_task_count",
            "feature_count",
        ):
            _positive_int(getattr(self, name), name)
        _non_negative_int(self.scalar_estimator_count, "scalar_estimator_count")
        _non_negative_int(self.training_rows, "training_rows")
        _non_negative_int(self.design_bytes, "design_bytes")
        if self.chunk_length is not None:
            _positive_int(self.chunk_length, "chunk_length")
        for horizon in self.dynamic_horizons:
            _positive_int(horizon, "dynamic_horizons entries")

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RuntimeResourceBudget:
    """Machine and parent-scheduler limits available to one runtime."""

    physical_cores: int
    logical_cores: int
    total_thread_limit: int
    memory_limit_bytes: int
    parent_concurrency: int = 1
    require_determinism: bool = True
    source: str = "detected"

    def __post_init__(self) -> None:
        for name in (
            "physical_cores",
            "logical_cores",
            "total_thread_limit",
            "memory_limit_bytes",
            "parent_concurrency",
        ):
            _positive_int(getattr(self, name), name)
        if self.physical_cores > self.logical_cores:
            raise ValueError("physical_cores must not exceed logical_cores")
        if not isinstance(self.require_determinism, bool):
            raise TypeError("require_determinism must be bool")
        if not self.source:
            raise ValueError("source must not be blank")

    @property
    def available_threads(self) -> int:
        machine_limit = min(self.physical_cores, self.total_thread_limit)
        return max(1, machine_limit // self.parent_concurrency)

    def payload(self) -> dict[str, Any]:
        return {**asdict(self), "available_threads": self.available_threads}


@dataclass(frozen=True, slots=True)
class RuntimeExecutionPlan:
    """One deterministic, budget-checked execution topology."""

    schema_version: int
    selected_axis: str
    config_workers: int
    member_workers: int
    window_workers: int
    quantile_workers: int
    output_workers: int
    model_threads: int
    available_threads: int
    budget_product: int
    task_count: int
    profile_source: str
    fallback_reasons: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("RuntimeExecutionPlan schema_version must be 1")
        if self.selected_axis not in {
            "serial",
            "config",
            "ensemble_member",
            "window",
            "quantile",
            "output",
        }:
            raise ValueError(f"unsupported selected_axis: {self.selected_axis!r}")
        for name in (
            "config_workers",
            "member_workers",
            "window_workers",
            "quantile_workers",
            "output_workers",
            "model_threads",
            "available_threads",
            "budget_product",
            "task_count",
        ):
            _positive_int(getattr(self, name), name)
        outer_workers = (
            self.config_workers,
            self.member_workers,
            self.window_workers,
            self.quantile_workers,
            self.output_workers,
        )
        if sum(worker > 1 for worker in outer_workers) > 1:
            raise ValueError("at most one outer parallel axis may exceed one")
        expected_product = self.model_threads
        for worker in outer_workers:
            expected_product *= worker
        if self.budget_product != expected_product:
            raise ValueError("budget_product does not match execution workers")
        if self.budget_product > self.available_threads:
            raise ValueError("execution plan exceeds available_threads")
        if not self.profile_source:
            raise ValueError("profile_source must not be blank")

    def payload(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "RuntimeExecutionPlan",
    "RuntimeResourceBudget",
    "RuntimeWorkload",
]
