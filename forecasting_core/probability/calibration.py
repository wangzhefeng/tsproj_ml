"""CQR/残差校准的配置与保存状态合同，不执行校准。"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from forecasting_core.probability._validation import (
    _is_close,
    _strict_bool,
    _strict_int,
    _strict_number,
)
from typing import Any, Tuple
import math
import pandas as pd


_SUPPORTED_CALIBRATION_METHODS = {"cqr"}


_SUPPORTED_CALIBRATION_GROUPINGS = {"pooled"}


def validate_cqr_params(alpha: float, min_scores: int) -> Tuple[float, int]:
    """校验 CQR 的误覆盖率和最小 score 数。"""
    alpha_value = _strict_number(alpha, "alpha")
    min_score_count = _strict_int(min_scores, "min_scores")
    if not math.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise ValueError("alpha must be finite and inside (0, 1)")
    if min_score_count <= 0:
        raise ValueError("min_scores must be > 0")
    return alpha_value, min_score_count


@dataclass(frozen=True)
class CalibrationSpec:
    """一个 prediction interval 的 CQR 校准配置。"""

    method: str
    interval_name: str
    target_coverage: float
    calibration_windows: int
    min_windows: int
    min_scores: int
    label_availability_delay_steps: int = 0
    allow_interval_shrink: bool = False
    grouping: str = "pooled"

    def __post_init__(self) -> None:
        method = str(self.method).lower()
        grouping = str(self.grouping).lower()
        coverage = _strict_number(self.target_coverage, "target_coverage")
        calibration_windows = _strict_int(self.calibration_windows, "calibration_windows")
        min_windows = _strict_int(self.min_windows, "min_windows")
        min_scores = _strict_int(self.min_scores, "min_scores")
        delay_steps = _strict_int(self.label_availability_delay_steps, "label_availability_delay_steps")
        if method not in _SUPPORTED_CALIBRATION_METHODS:
            raise ValueError(f"Unsupported calibration method={method}")
        if grouping not in _SUPPORTED_CALIBRATION_GROUPINGS:
            raise ValueError(f"Unsupported calibration grouping={grouping}")
        if not math.isfinite(coverage) or not 0.0 < coverage < 1.0:
            raise ValueError("target_coverage must be finite and inside (0, 1)")
        if calibration_windows <= 0:
            raise ValueError("calibration_windows must be > 0")
        if min_windows <= 0:
            raise ValueError("min_windows must be > 0")
        if min_windows > calibration_windows:
            raise ValueError("min_windows must be <= calibration_windows")
        if min_scores <= 0:
            raise ValueError("min_scores must be > 0")
        if delay_steps < 0:
            raise ValueError("label_availability_delay_steps must be >= 0")
        interval_name = str(self.interval_name).strip()
        if not interval_name:
            raise ValueError("calibration interval must not be empty")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "interval_name", interval_name)
        object.__setattr__(self, "target_coverage", coverage)
        object.__setattr__(self, "calibration_windows", calibration_windows)
        object.__setattr__(self, "min_windows", min_windows)
        object.__setattr__(self, "min_scores", min_scores)
        object.__setattr__(self, "label_availability_delay_steps", delay_steps)
        object.__setattr__(self, "allow_interval_shrink", _strict_bool(self.allow_interval_shrink, "allow_interval_shrink"))
        object.__setattr__(self, "grouping", grouping)


def validate_cqr_state(state: Mapping[str, Any] | None, spec: CalibrationSpec) -> None:
    """保存和部署共同校验冻结校准事实；不重建或重新校准。"""
    if not isinstance(state, Mapping):
        raise ValueError("CQR bundle requires calibration state")
    if state.get("method") != "cqr" or state.get("interval") != spec.interval_name:
        raise ValueError("CQR calibration state method/interval does not match spec")
    coverage = _strict_number(state.get("target_coverage"), "target_coverage")
    if not math.isfinite(coverage) or not _is_close(coverage, spec.target_coverage):
        raise ValueError("CQR calibration state target_coverage does not match spec")
    origin = state.get("forecast_origin")
    if not isinstance(origin, str) or bool(pd.isna(pd.Timestamp(origin))):
        raise ValueError("CQR calibration state requires finite forecast_origin")
    status = state.get("status")
    if status not in {"applied", "insufficient_windows", "insufficient_scores"}:
        raise ValueError("invalid CQR calibration state status")
    counts = {}
    for name in ("selected_windows", "selected_scores"):
        counts[name] = _strict_int(state.get(name), name)
        if counts[name] < 0:
            raise ValueError(f"CQR {name} must be non-negative")
    if counts["selected_windows"] > spec.calibration_windows:
        raise ValueError("CQR selected_windows exceeds calibration_windows")
    if status == "applied":
        correction = _strict_number(state.get("correction"), "correction")
        if not math.isfinite(correction):
            raise ValueError("CQR correction must be finite")
        if correction < 0 and not spec.allow_interval_shrink:
            raise ValueError("CQR negative correction requires allow_interval_shrink")
        if counts["selected_windows"] < spec.min_windows or counts["selected_scores"] < spec.min_scores:
            raise ValueError("applied CQR calibration state does not meet sample thresholds")
    elif state.get("correction") is not None:
        raise ValueError("unavailable CQR calibration state forbids correction")


@dataclass(frozen=True, slots=True)
class ResidualCalibrationSpec:
    target_coverage: float
    calibration_windows: int = 30
    min_windows: int = 5
    min_scores: int = 5
    label_availability_delay_steps: int = 0
    method: str = "absolute_residual"
    grouping: str = "series_target_horizon"

    def __post_init__(self):
        if self.method != "absolute_residual" or self.grouping != "series_target_horizon":
            raise ValueError("residual calibration requires absolute_residual/series_target_horizon")
        if isinstance(self.target_coverage, bool) or not isinstance(self.target_coverage, (int, float)):
            raise ValueError("target_coverage must be numeric")
        if not math.isfinite(self.target_coverage) or not 0 < self.target_coverage < 1:
            raise ValueError("target_coverage must be finite and inside (0, 1)")
        for name in ("calibration_windows", "min_windows", "min_scores", "label_availability_delay_steps"):
            value = getattr(self, name)
            minimum = 0 if name == "label_availability_delay_steps" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"invalid residual calibration {name}")
        if self.min_windows > self.calibration_windows:
            raise ValueError("min_windows must not exceed calibration_windows")

    @classmethod
    def from_mapping(cls, value: Mapping) -> ResidualCalibrationSpec:
        allowed = {"method", "grouping", "target_coverage", "calibration_windows", "min_windows", "min_scores", "label_availability_delay_steps"}
        if not isinstance(value, Mapping) or set(value) - allowed or "target_coverage" not in value:
            raise ValueError("invalid absolute_residual calibration fields")
        return cls(**dict(value))


def validate_residual_state(state: dict, *, series_ids: tuple, targets: tuple, shape: tuple, coverage: float, freq: str | None = None) -> None:
    if not isinstance(state, dict) or state.get("method") != "absolute_residual" or state.get("grouping") != "series_target_horizon":
        raise ValueError("invalid absolute_residual calibration state")
    if (tuple(state.get("series_ids", ())) != series_ids or tuple(state.get("targets", ())) != targets
            or tuple(state.get("shape", ())) != shape or state.get("target_coverage") != coverage):
        raise ValueError("residual state axes/coverage do not match prediction")
    if not isinstance(state.get("origin"), str) or pd.isna(pd.Timestamp(state["origin"])):
        raise ValueError("residual state requires a finite calibration origin")
    if not isinstance(state.get("freq"), str):
        raise ValueError("residual state requires a frequency")
    offset = pd.tseries.frequencies.to_offset(state["freq"])
    if offset.nanos <= 0 or (freq is not None and offset != pd.tseries.frequencies.to_offset(freq)):
        raise ValueError("residual state frequency does not match prediction")
    validate_interval_values(math.prod(shape), state.get("radii", ()), state.get("statuses", ()), coverage)


def validate_interval_values(size: int, radii, statuses, coverage: float) -> None:
    if not math.isfinite(coverage) or not 0 < coverage < 1:
        raise ValueError("target_coverage must be inside (0, 1)")
    if len(radii) != size or len(statuses) != size:
        raise ValueError("point interval state must match N*H*K axes")
    allowed = {"applied", "insufficient_windows", "insufficient_scores", "insufficient_rank"}
    for radius, status in zip(radii, statuses):
        if status not in allowed or (radius is None) != (status != "applied"):
            raise ValueError("point interval availability/status mismatch")
        if radius is not None and (isinstance(radius, bool) or not isinstance(radius, (int, float))
                                   or not math.isfinite(radius) or radius < 0):
            raise ValueError("applied residual radius must be finite and nonnegative")
