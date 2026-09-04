"""Typed runtime-validation and backtest geometry contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from forecasting_core.specs._mapping import (
    FrozenMappingSpec,
    freeze_json_value,
    strict_mapping,
    validate_nested_mappings,
)


@dataclass(frozen=True, slots=True)
class FixedStepBacktestSpec:
    """Rolling-origin geometry measured only in supervised origin steps."""

    history_steps: int
    train_window_steps: int
    fold_count: int
    stride_steps: int

    def __post_init__(self) -> None:
        for field_name in (
            "history_steps",
            "train_window_steps",
            "fold_count",
            "stride_steps",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"validation.{field_name} must be a positive integer")
        if self.train_window_steps >= self.history_steps:
            raise ValueError(
                "validation.train_window_steps must be smaller than history_steps"
            )


@dataclass(frozen=True, slots=True)
class CalendarMonthBacktestSpec:
    """Month-aligned geometry: raw training days and month-spaced folds."""

    train_window_days: int
    fold_count: int
    stride_months: int

    def __post_init__(self) -> None:
        for field_name in ("train_window_days", "fold_count", "stride_months"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"validation.{field_name} must be a positive integer")


BacktestSpec = FixedStepBacktestSpec | CalendarMonthBacktestSpec


@dataclass(frozen=True, slots=True)
class RuntimePerformanceSpec(FrozenMappingSpec):
    """Typed non-semantic execution controls accepted from model YAML."""

    window_parallel_workers: int | None
    multi_output_n_jobs: int | None
    quantile_parallel_workers: int | None
    model_thread_count: int | None
    total_thread_limit: int | None
    memory_limit_bytes: int | None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | "RuntimePerformanceSpec",
        *,
        source: str = "<constructor>",
    ) -> "RuntimePerformanceSpec":
        if isinstance(value, cls):
            return value
        allowed = frozenset(
            {
                "window_parallel_workers",
                "multi_output_n_jobs",
                "quantile_parallel_workers",
                "model_thread_count",
                "total_thread_limit",
                "memory_limit_bytes",
            }
        )
        payload = strict_mapping(
            value,
            path="validation.performance",
            source=source,
            allowed=allowed,
        )
        for field_name, field_value in payload.items():
            if (
                isinstance(field_value, bool)
                or not isinstance(field_value, int)
                or field_value <= 0
            ):
                raise ValueError(
                    f"validation.performance.{field_name} must be a positive integer"
                )
        return cls(
            freeze_json_value(payload, "validation.performance"),
            window_parallel_workers=payload.get("window_parallel_workers"),
            multi_output_n_jobs=payload.get("multi_output_n_jobs"),
            quantile_parallel_workers=payload.get("quantile_parallel_workers"),
            model_thread_count=payload.get("model_thread_count"),
            total_thread_limit=payload.get("total_thread_limit"),
            memory_limit_bytes=payload.get("memory_limit_bytes"),
        )

VALIDATION_FIELDS = frozenset(
    {
        "forecast_origin",
        "schedule_mode",
        "horizon_mode",
        "history_steps",
        "train_window_steps",
        "fold_count",
        "stride_steps",
        "train_window_days",
        "stride_months",
        "training_scope",
        "training",
        "train_outlier",
        "eval_mask",
        "performance",
        "aggregate_weighting",
        "seasonal_naive_lag",
    }
)

_VALIDATION_NESTED_FIELDS: dict[str, frozenset[str]] = {
    "validation.training_scope": frozenset(
        {"incomplete_series_policy", "unknown_series_policy", "series_order"}
    ),
    "validation.training": frozenset(
        {
            "early_stopping_patience",
            "sample_weight",
            "tuning",
            "augmentation",
            "feature_selection",
            "learning_rate",
            "huber_delta",
            "blend_weight_windows",
            "estimator_ensemble",
        }
    ),
    "validation.training.sample_weight": frozenset({"method", "halflife_days"}),
    "validation.training.tuning": frozenset({"method", "metric", "n_splits"}),
    "validation.training.augmentation": frozenset(
        {"method", "ratio", "feature_noise_std", "target_noise_std", "random_state"}
    ),
    "validation.training.feature_selection": frozenset(
        {"method", "max_features", "min_features"}
    ),
    "validation.training.learning_rate": frozenset({"method", "min", "max"}),
    "validation.training.estimator_ensemble": frozenset(
        {"method", "members", "member_specs", "validation_ratio"}
    ),
    "validation.train_outlier": frozenset({"method", "high", "rise", "low", "drop"}),
    "validation.train_outlier.high": frozenset({"threshold", "max_run_points"}),
    "validation.train_outlier.rise": frozenset(
        {"max_run_points", "rebound_min_abs_diff"}
    ),
    "validation.train_outlier.low": frozenset({"threshold", "max_run_points"}),
    "validation.train_outlier.drop": frozenset(
        {"max_run_points", "rebound_min_abs_diff"}
    ),
    "validation.eval_mask": frozenset(
        {"mode", "percentile", "min_value", "max_value"}
    ),
    "validation.performance": frozenset(
        {
            "window_parallel_workers",
            "multi_output_n_jobs",
            "quantile_parallel_workers",
            "model_thread_count",
            "total_thread_limit",
            "memory_limit_bytes",
        }
    ),
}


@dataclass(frozen=True, slots=True)
class RuntimeValidationSpec(FrozenMappingSpec):
    """Strict validation section plus one unambiguous typed backtest geometry."""

    backtest: BacktestSpec | None
    performance: RuntimePerformanceSpec | None

    def semantic_payload(self) -> dict[str, Any]:
        """Validation geometry/semantics without execution-only controls."""
        payload = self.canonical_payload()
        payload.pop("performance", None)
        return payload

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | "RuntimeValidationSpec",
        *,
        source: str = "<constructor>",
        require_geometry: bool = False,
    ) -> "RuntimeValidationSpec":
        if isinstance(value, cls):
            if require_geometry and value.backtest is None:
                raise ValueError(f"validation geometry is required in {source}")
            return value
        payload = strict_mapping(
            value,
            path="validation",
            source=source,
            allowed=VALIDATION_FIELDS,
        )
        validate_nested_mappings(
            payload,
            source=source,
            schemas=_VALIDATION_NESTED_FIELDS,
        )
        schedule_mode = str(payload.get("schedule_mode", "daily")).lower()
        if schedule_mode not in {"daily", "intraday"}:
            raise ValueError("validation.schedule_mode must be daily or intraday")
        horizon_mode = str(payload.get("horizon_mode", "fixed_steps")).lower()
        if horizon_mode not in {"fixed_steps", "calendar_month"}:
            raise ValueError(
                "validation.horizon_mode must be fixed_steps or calendar_month"
            )
        backtest = _parse_backtest_geometry(
            payload,
            horizon_mode=horizon_mode,
            source=source,
            required=require_geometry,
        )
        performance = (
            RuntimePerformanceSpec.from_mapping(
                payload["performance"],
                source=source,
            )
            if "performance" in payload
            else None
        )
        return cls(
            freeze_json_value(payload, "validation"),
            backtest,
            performance,
        )


def _parse_backtest_geometry(
    payload: Mapping[str, Any],
    *,
    horizon_mode: str,
    source: str,
    required: bool,
) -> BacktestSpec | None:
    fixed_fields = frozenset(
        {"history_steps", "train_window_steps", "fold_count", "stride_steps"}
    )
    calendar_fields = frozenset({"train_window_days", "fold_count", "stride_months"})
    keys = set(payload)
    if horizon_mode == "fixed_steps":
        forbidden = sorted(keys & {"train_window_days", "stride_months"})
        if forbidden:
            raise ValueError(
                f"fixed_steps validation forbids calendar fields in {source}: {forbidden}"
            )
        present = keys & fixed_fields
        if not present and not required:
            return None
        missing = sorted(fixed_fields - keys)
        if missing:
            raise ValueError(
                f"fixed_steps validation missing geometry fields in {source}: {missing}"
            )
        return FixedStepBacktestSpec(
            history_steps=payload["history_steps"],
            train_window_steps=payload["train_window_steps"],
            fold_count=payload["fold_count"],
            stride_steps=payload["stride_steps"],
        )

    forbidden = sorted(keys & {"history_steps", "train_window_steps", "stride_steps"})
    if forbidden:
        raise ValueError(
            f"calendar_month validation forbids fixed-step fields in {source}: {forbidden}"
        )
    present = keys & calendar_fields
    if not present and not required:
        return None
    missing = sorted(calendar_fields - keys)
    if missing:
        raise ValueError(
            f"calendar_month validation missing geometry fields in {source}: {missing}"
        )
    return CalendarMonthBacktestSpec(
        train_window_days=payload["train_window_days"],
        fold_count=payload["fold_count"],
        stride_months=payload["stride_months"],
    )


__all__ = [
    "BacktestSpec",
    "CalendarMonthBacktestSpec",
    "FixedStepBacktestSpec",
    "RuntimePerformanceSpec",
    "RuntimeValidationSpec",
    "VALIDATION_FIELDS",
]
