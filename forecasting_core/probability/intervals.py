"""点模型与分位模型的预测区间结果合同。"""
from __future__ import annotations

from dataclasses import dataclass
from forecasting_core.probability.calibration import validate_interval_values
from forecasting_core.probability.grid import validate_interval_quantiles
from forecasting_core.tensors.point import PointForecastTensor
from typing import Any
import math
import numpy as np


def _as_finite_1d(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape={array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


@dataclass
class PredictionIntervalForecast:
    """独立 prediction interval；边界不冒充分位数。"""

    name: str
    lower: np.ndarray
    upper: np.ndarray
    target_coverage: float
    method: str
    base_quantiles: tuple[float, float]

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        self.method = str(self.method).strip().lower()
        if not self.name:
            raise ValueError("prediction interval name must not be empty")
        if not self.method:
            raise ValueError("prediction interval method must not be empty")
        self.lower = _as_finite_1d(self.lower, "interval lower")
        self.upper = _as_finite_1d(self.upper, "interval upper")
        if len(self.lower) != len(self.upper):
            raise ValueError(
                "prediction interval length mismatch: "
                f"lower={len(self.lower)}, upper={len(self.upper)}"
            )
        if np.any(self.lower > self.upper):
            raise ValueError("prediction interval requires lower <= upper at every point")
        coverage = float(self.target_coverage)
        if not math.isfinite(coverage) or not 0.0 < coverage < 1.0:
            raise ValueError("target_coverage must be finite and inside (0, 1)")
        if len(self.base_quantiles) != 2:
            raise ValueError("base_quantiles must contain exactly two levels")
        lower_q, upper_q = validate_interval_quantiles(
            self.base_quantiles[0],
            self.base_quantiles[1],
            self.base_quantiles,
        )
        self.target_coverage = coverage
        self.base_quantiles = (lower_q, upper_q)


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
