"""Typed runtime-validation and backtest geometry contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any
from forecasting_core.specs.training import SAMPLE_WEIGHT_FIELDS, resolve_sample_weight_spec

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
    train_window_steps: int | None
    fold_count: int
    stride_steps: int
    train_history_steps: int | None = None
    explicit_training_window: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "history_steps",
            "fold_count",
            "stride_steps",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"validation.{field_name} must be a positive integer")
        if self.train_window_steps is None and not self.explicit_training_window:
            raise ValueError("validation.train_window_steps must be a positive integer")
        if self.train_window_steps is not None and (
            isinstance(self.train_window_steps, bool)
            or not isinstance(self.train_window_steps, int)
            or self.train_window_steps <= 0
        ):
            raise ValueError("validation.train_window_steps must be a positive integer")
        if self.train_window_steps is not None and self.train_window_steps >= self.history_steps:
            raise ValueError(
                "validation.train_window_steps must be smaller than history_steps"
            )
        if self.train_history_steps is not None and (
            isinstance(self.train_history_steps, bool)
            or not isinstance(self.train_history_steps, int)
            or self.train_history_steps <= 0
        ):
            raise ValueError("validation.train_history_steps must be a positive integer")


@dataclass(frozen=True, slots=True)
class SlidingWindowBacktestSpec:
    """重叠滑窗几何：与 fixed-step 同字段族，语义为 stride_steps < horizon。

    测试折相互重叠（同一时刻被多个折预测），折级评分照常逐折进行，
    产物侧不拼接总图。stride_steps >= horizon 时与 fixed_steps 无差异，
    在窗口构造期 RAISE。
    """

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
class ExpandingWindowBacktestSpec:
    """扩展窗几何：每折训练集为全部合格历史候选（不截断 train_window_steps）。

    无固定训练窗口长度，final fit 的窗口语义因此无定义——暂限
    backtest-only（与 train_history_steps 同一先例）。
    """

    history_steps: int
    fold_count: int
    stride_steps: int

    def __post_init__(self) -> None:
        for field_name in ("history_steps", "fold_count", "stride_steps"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"validation.{field_name} must be a positive integer")


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


BacktestSpec = (
    FixedStepBacktestSpec
    | SlidingWindowBacktestSpec
    | ExpandingWindowBacktestSpec
    | CalendarMonthBacktestSpec
)


@dataclass(frozen=True, slots=True)
class RuntimePerformanceSpec(FrozenMappingSpec):
    """Typed non-semantic execution controls accepted from model YAML."""

    window_parallel_workers: int | None
    multi_output_n_jobs: int | None
    quantile_parallel_workers: int | None
    ensemble_parallel_workers: int | None
    model_thread_count: int | None
    total_thread_limit: int | None
    memory_limit_bytes: int | None
    profile_ref: str | None = None

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
                "ensemble_parallel_workers",
                "model_thread_count",
                "total_thread_limit",
                "memory_limit_bytes",
            }
            | {"profile_ref"}
        )
        payload = strict_mapping(
            value,
            path="validation.performance",
            source=source,
            allowed=allowed,
        )
        for field_name, field_value in payload.items():
            if field_name == "profile_ref":
                if not isinstance(field_value, str) or not field_value.strip():
                    raise ValueError("validation.performance.profile_ref must be nonempty text")
                continue
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
            ensemble_parallel_workers=payload.get("ensemble_parallel_workers"),
            model_thread_count=payload.get("model_thread_count"),
            total_thread_limit=payload.get("total_thread_limit"),
            memory_limit_bytes=payload.get("memory_limit_bytes"),
            profile_ref=payload.get("profile_ref"),
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

        "eval_mask",
        "performance",
        "aggregate_weighting",
        "seasonal_naive_lag",
        "train_history_steps",
        "refit_every",
        "forecast_window",
        "training_window",
    }
)

_VALIDATION_NESTED_FIELDS: dict[str, frozenset[str]] = {
    "validation.training_scope": frozenset(
        {"incomplete_series_policy", "unknown_series_policy", "series_order"}
    ),
    "validation.forecast_window": frozenset({"start", "gap_steps"}),
    "validation.training_window": frozenset({"kind", "history_steps", "start_time"}),
    "validation.training": frozenset(
        {
            "origin_sampling",
            "sample_weight",
        }
    ),
    "validation.training.sample_weight": SAMPLE_WEIGHT_FIELDS,
    "validation.training.origin_sampling": frozenset(
        {"stride_steps", "time_of_day", "max_origins", "anchor_time"}
    ),

    "validation.eval_mask": frozenset(
        {"mode", "percentile", "min_value", "max_value"}
    ),
    "validation.performance": frozenset(
        {
            "window_parallel_workers",
            "multi_output_n_jobs",
            "quantile_parallel_workers",
            "ensemble_parallel_workers",
            "model_thread_count",
            "total_thread_limit",
            "memory_limit_bytes",
        }
        | {"profile_ref"}
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
        if payload.get("refit_every", 1) == 1:
            payload.pop("refit_every", None)
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
        # eval_mask.mode 解析期白名单（2026-10-05 前置：此前拼错 mode 要跑完
        # 整条回测、在评分时才由 build_eval_mask 报 ValueError）。
        # 与消费方 build_eval_mask_payload 同口径：不做大小写折叠。
        eval_mask = payload.get("eval_mask")
        if eval_mask is not None:
            mask_mode = str(eval_mask.get("mode", "percentile"))
            if mask_mode not in {"percentile", "absolute", "combined"}:
                raise ValueError(
                    "validation.eval_mask.mode must be one of "
                    "['percentile', 'absolute', 'combined']; "
                    f"got {mask_mode!r} in {source}"
                )
        schedule_mode = str(payload.get("schedule_mode", "daily")).lower()
        if schedule_mode not in {"daily", "intraday"}:
            raise ValueError("validation.schedule_mode must be daily or intraday")
        horizon_mode = str(payload.get("horizon_mode", "fixed_steps")).lower()
        if horizon_mode not in {"fixed_steps", "sliding_window", "expanding_window", "calendar_month"}:
            raise ValueError(
                "validation.horizon_mode must be fixed_steps, sliding_window, "
                "expanding_window or calendar_month"
            )
        if horizon_mode != "fixed_steps" and (
            "training_window" in payload or "forecast_window" in payload
        ):
            raise ValueError(
                "forecast_window/training_window require fixed_steps horizon_mode"
            )
        training = payload.get("training", {})
        if not isinstance(training, Mapping):
            raise TypeError("validation.training must be a mapping")
        if "sample_weight" in training:
            resolve_sample_weight_spec(training["sample_weight"])
        sampling = training.get("origin_sampling")
        if sampling is not None:
            for field, minimum in (("stride_steps", 1), ("max_origins", 2)):
                if field in sampling:
                    number = sampling[field]
                    if isinstance(number, bool) or not isinstance(number, int) or number < minimum:
                        raise ValueError(f"origin_sampling.{field} must be an integer >= {minimum}")
            if "time_of_day" in sampling:
                clock = sampling["time_of_day"]
                if not isinstance(clock, str) or re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", clock) is None:
                    raise ValueError("origin_sampling.time_of_day must be HH:MM")
                if "stride_steps" in sampling:
                    raise ValueError("origin_sampling time_of_day and stride_steps are mutually exclusive")
        backtest = _parse_backtest_geometry(
            payload,
            horizon_mode=horizon_mode,
            source=source,
            required=require_geometry,
        )
        refit_every = payload.get("refit_every", 1)
        if isinstance(refit_every, bool) or not isinstance(refit_every, int) or refit_every < 0:
            raise ValueError("validation.refit_every must be a non-negative integer")
        if refit_every != 1:
            if backtest is None or horizon_mode == "calendar_month":
                raise ValueError("non-default refit_every requires rolling geometry, not calendar-month")
            if payload.get("train_history_steps") is not None:
                raise ValueError("non-default refit_every does not support train_history_steps")
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
    if horizon_mode in {"fixed_steps", "sliding_window", "expanding_window"}:
        forbidden = sorted(keys & {"train_window_days", "stride_months"})
        if forbidden:
            raise ValueError(
                f"{horizon_mode} validation forbids calendar fields in {source}: {forbidden}"
            )
        if horizon_mode == "fixed_steps" and "training_window" in payload:
            if payload["training_window"] is None:
                raise ValueError("training_window must be a mapping")
            if keys & {"train_history_steps", "train_window_steps"}:
                raise ValueError("training_window replaces train_history_steps/train_window_steps")
            fixed_fields = fixed_fields - {"train_window_steps"}
        if horizon_mode == "expanding_window":
            forbidden = sorted(keys & {"train_window_steps", "train_history_steps"})
            if forbidden:
                raise ValueError(
                    f"expanding_window validation forbids fields in {source}: {forbidden}"
                )
            required_fields = frozenset({"history_steps", "fold_count", "stride_steps"})
        else:
            required_fields = fixed_fields
        if horizon_mode == "sliding_window" and "train_history_steps" in keys:
            raise ValueError(
                f"sliding_window validation forbids train_history_steps in {source}"
            )
        present = keys & fixed_fields
        if not present and not required and "train_history_steps" not in keys:
            return None
        missing = sorted(required_fields - keys)
        if missing:
            raise ValueError(
                f"{horizon_mode} validation missing geometry fields in {source}: {missing}"
            )
        if "train_history_steps" in payload and payload["train_history_steps"] is None:
            raise ValueError("validation.train_history_steps must be a positive integer")
        if horizon_mode == "sliding_window":
            return SlidingWindowBacktestSpec(
                history_steps=payload["history_steps"],
                train_window_steps=payload["train_window_steps"],
                fold_count=payload["fold_count"],
                stride_steps=payload["stride_steps"],
            )
        if horizon_mode == "expanding_window":
            return ExpandingWindowBacktestSpec(
                history_steps=payload["history_steps"],
                fold_count=payload["fold_count"],
                stride_steps=payload["stride_steps"],
            )
        return FixedStepBacktestSpec(
            history_steps=payload["history_steps"],
            train_window_steps=payload.get("train_window_steps"),
            fold_count=payload["fold_count"],
            stride_steps=payload["stride_steps"],
            train_history_steps=payload.get("train_history_steps"),
            explicit_training_window="training_window" in payload,
        )

    forbidden = sorted(keys & {"history_steps", "train_window_steps", "stride_steps", "train_history_steps"})
    if forbidden:
        raise ValueError(
            f"calendar_month validation forbids rolling-mode fields in {source}: {forbidden}"
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
    "ExpandingWindowBacktestSpec",
    "FixedStepBacktestSpec",
    "RuntimePerformanceSpec",
    "RuntimeValidationSpec",
    "SlidingWindowBacktestSpec",
    "VALIDATION_FIELDS",
]
