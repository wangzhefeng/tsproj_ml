# -*- coding: utf-8 -*-
"""概率预测配置值的严格语义校验。"""

import math
from typing import Any, Iterable, Tuple


def validate_quantile_grid(
    quantiles: Iterable[float],
    point_quantile: float = 0.5,
) -> Tuple[float, ...]:
    """将合法分位数网格归一化为严格递增的 float tuple。"""
    levels = tuple(float(level) for level in quantiles)
    if not levels:
        raise ValueError("quantiles must not be empty")
    if any(not math.isfinite(level) or not 0.0 < level < 1.0 for level in levels):
        raise ValueError("quantiles must be finite and inside (0, 1)")
    if len(set(levels)) != len(levels):
        raise ValueError("quantiles must be unique")
    if any(left >= right for left, right in zip(levels, levels[1:])):
        raise ValueError("quantiles must be strictly increasing")
    point = float(point_quantile)
    if not any(math.isclose(level, point, rel_tol=0.0, abs_tol=1e-12) for level in levels):
        raise ValueError(f"point_quantile={point:g} must be present in quantiles")
    return levels


def validate_interval_quantiles(
    lower_quantile: float,
    upper_quantile: float,
    quantiles: Iterable[float],
) -> Tuple[float, float]:
    """校验区间边界引用已配置且严格有序的 quantile。"""
    lower = float(lower_quantile)
    upper = float(upper_quantile)
    levels = tuple(float(level) for level in quantiles)
    if lower >= upper:
        raise ValueError("lower_quantile must be < upper_quantile")
    for name, value in (("lower_quantile", lower), ("upper_quantile", upper)):
        if not any(math.isclose(level, value, rel_tol=0.0, abs_tol=1e-12) for level in levels):
            raise ValueError(f"{name}={value:g} must be present in quantiles")
    return lower, upper


def validate_cqr_params(alpha: float, min_scores: int) -> Tuple[float, int]:
    """校验 legacy CQR 的误覆盖率和最小 score 数。"""
    alpha_value = float(alpha)
    min_score_count = int(min_scores)
    if not math.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise ValueError("alpha must be finite and inside (0, 1)")
    if min_score_count <= 0:
        raise ValueError("min_scores must be > 0")
    return alpha_value, min_score_count


def validate_probabilistic_args(args: Any) -> Tuple[float, ...]:
    """在主流程开始前校验 legacy 概率预测配置。"""
    mode = str(getattr(args, "predict_type", "point") or "point").lower()
    if mode not in {"point", "quantile"}:
        raise ValueError(f"Unsupported predict_type={mode}; expected point or quantile")

    conformal_enabled = bool(getattr(args, "enable_conformal_calibration", False))
    if mode == "point":
        if conformal_enabled:
            raise ValueError("enable_conformal_calibration requires predict_type=quantile")
        return ()

    levels = validate_quantile_grid(
        getattr(args, "quantiles", ()),
        point_quantile=0.5,
    )
    if conformal_enabled:
        validate_interval_quantiles(levels[0], levels[-1], levels)
        validate_cqr_params(
            alpha=getattr(args, "conformal_alpha", 0.1),
            min_scores=getattr(args, "conformal_min_scores", 30),
        )
        calibration_windows = int(getattr(args, "conformal_calibration_windows", 5))
        if calibration_windows <= 0:
            raise ValueError("conformal_calibration_windows must be > 0")
    return levels
