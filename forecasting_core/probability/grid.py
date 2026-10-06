"""分位网格及列名编码合同。"""
from __future__ import annotations

from dataclasses import dataclass
from forecasting_core.probability._validation import _is_close, _strict_number
from typing import Iterable, Tuple
import math


def _quantile_column_token(level: float) -> str:
    percent = float(level) * 100.0
    token = format(percent, ".15g")
    return token.replace(".", "p")


@dataclass(frozen=True)
class QuantileGrid:
    """数值 quantile grid；CSV 列名只是可逆序列化表示。"""

    levels: tuple[float, ...]
    point_level: float = 0.5

    def __post_init__(self) -> None:
        levels = validate_quantile_grid(self.levels, self.point_level)
        column_names = [f"predict_q{_quantile_column_token(level)}" for level in levels]
        if len(set(column_names)) != len(column_names):
            raise ValueError(
                "quantile column codec collision; levels are too close to serialize safely"
            )
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "point_level", float(self.point_level))

    def index_of(self, level: float) -> int:
        value = float(level)
        for index, candidate in enumerate(self.levels):
            if math.isclose(candidate, value, rel_tol=0.0, abs_tol=1e-12):
                return index
        raise ValueError(f"quantile level={value:g} is not present in the grid")

    @property
    def point_index(self) -> int:
        return self.index_of(self.point_level)

    def column_name(self, level: float) -> str:
        canonical_level = self.levels[self.index_of(level)]
        return f"predict_q{_quantile_column_token(canonical_level)}"


def validate_quantile_grid(
    quantiles: Iterable[float],
    point_quantile: float = 0.5,
) -> Tuple[float, ...]:
    """将合法分位数网格归一化为严格递增的 float tuple。"""
    levels = tuple(_strict_number(level, "quantiles") for level in quantiles)
    if not levels:
        raise ValueError("quantiles must not be empty")
    if any(not math.isfinite(level) or not 0.0 < level < 1.0 for level in levels):
        raise ValueError("quantiles must be finite and inside (0, 1)")
    if len(set(levels)) != len(levels):
        raise ValueError("quantiles must be unique")
    if any(left >= right for left, right in zip(levels, levels[1:])):
        raise ValueError("quantiles must be strictly increasing")
    point = _strict_number(point_quantile, "point_quantile")
    if not any(_is_close(level, point) for level in levels):
        raise ValueError(f"point_quantile={point:g} must be present in quantiles")
    return levels


def validate_interval_quantiles(
    lower_quantile: float,
    upper_quantile: float,
    quantiles: Iterable[float],
) -> Tuple[float, float]:
    """校验区间边界引用已配置且严格有序的 quantile。"""
    lower = _strict_number(lower_quantile, "lower_quantile")
    upper = _strict_number(upper_quantile, "upper_quantile")
    levels = tuple(float(level) for level in quantiles)
    if lower >= upper:
        raise ValueError("lower_quantile must be < upper_quantile")
    for name, value in (("lower_quantile", lower), ("upper_quantile", upper)):
        if not any(_is_close(level, value) for level in levels):
            raise ValueError(f"{name}={value:g} must be present in quantiles")
    return lower, upper


def _quantile_token(level: float) -> str:
    percent = float(level) * 100.0
    if _is_close(percent, round(percent)):
        return str(int(round(percent)))
    return f"{percent:.12f}".rstrip("0").rstrip(".").replace(".", "p")
