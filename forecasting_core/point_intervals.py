"""点模型残差区间合同；与模型 quantile 完全分离。"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from forecasting_core.tensors import PointForecastTensor


def pi_column_names(target_coverage: float) -> tuple[str, str]:
    """点模型与 CQR 共用的预测区间列名合同。"""
    coverage = float(target_coverage)
    alpha = 1.0 - coverage
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and inside (0, 1)")
    percent = coverage * 100.0
    token = (str(int(round(percent))) if np.isclose(percent, round(percent), atol=1e-10)
             else f"{percent:.6f}".rstrip("0").rstrip(".").replace(".", "p"))
    return f"predict_pi{token}_lower", f"predict_pi{token}_upper"


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


@dataclass(frozen=True, slots=True)
class PointIntervalForecast:
    point: PointForecastTensor
    radii: tuple[float | None, ...]
    target_coverage: float
    statuses: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.point, PointForecastTensor):
            raise TypeError("point must be PointForecastTensor")
        radii, statuses = tuple(self.radii), tuple(self.statuses)
        validate_interval_values(self.point.values.size, radii, statuses, self.target_coverage)
        object.__setattr__(self, "radii", radii)
        object.__setattr__(self, "statuses", statuses)
        if not np.isfinite(self.lower[self.available]).all() or not np.isfinite(self.upper[self.available]).all():
            raise ValueError("applied point interval bounds must be finite")

    @property
    def radius(self) -> np.ndarray:
        return np.asarray([np.nan if value is None else value for value in self.radii], dtype=float).reshape(self.point.shape)

    @property
    def available(self) -> np.ndarray:
        return np.asarray([value is not None for value in self.radii]).reshape(self.point.shape)

    @property
    def lower(self) -> np.ndarray:
        return self.point.values - self.radius

    @property
    def upper(self) -> np.ndarray:
        return self.point.values + self.radius
