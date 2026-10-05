# -*- coding: utf-8 -*-
"""概率预测 proper scores、区间质量和校准指标。"""

import math
from typing import Dict, Optional, Tuple

import numpy as np


def _paired_finite_arrays(*named_values) -> Tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(values, dtype=float).reshape(-1) for _, values in named_values)
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        details = ", ".join(
            f"{name}={len(array)}"
            for (name, _), array in zip(named_values, arrays)
        )
        raise ValueError(f"probabilistic metric length mismatch: {details}")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("probabilistic metrics require finite inputs")
    return arrays


def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
) -> np.ndarray:
    """返回逐点 pinball loss。"""
    q = float(quantile)
    if not math.isfinite(q) or not 0.0 < q < 1.0:
        raise ValueError("quantile must be finite and inside (0, 1)")
    y, prediction = _paired_finite_arrays(("y_true", y_true), ("y_pred", y_pred))
    error = y - prediction
    return np.maximum(q * error, (q - 1.0) * error)


def wilson_interval(
    successes: int,
    total: int,
    z_value: float = 1.959963984540054,
) -> Tuple[float, float]:
    """二项覆盖率的 Wilson score interval。"""
    n = int(total)
    count = int(successes)
    if n <= 0:
        raise ValueError("total must be > 0")
    if not 0 <= count <= n:
        raise ValueError("successes must be between 0 and total")
    z = float(z_value)
    proportion = count / n
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2.0 * n)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / n
            + z * z / (4.0 * n * n)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def interval_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_coverage: float,
) -> Dict[str, float]:
    """计算 coverage、width、Winkler、coverage gap 及 Wilson CI。"""
    coverage_target = float(target_coverage)
    if not math.isfinite(coverage_target) or not 0.0 < coverage_target < 1.0:
        raise ValueError("target_coverage must be finite and inside (0, 1)")
    y, lower_values, upper_values = _paired_finite_arrays(
        ("y_true", y_true),
        ("lower", lower),
        ("upper", upper),
    )
    if len(y) == 0:
        raise ValueError("interval metrics require at least one point")
    if np.any(lower_values > upper_values):
        raise ValueError("interval metrics require lower <= upper at every point")

    covered = (y >= lower_values) & (y <= upper_values)
    widths = upper_values - lower_values
    alpha = 1.0 - coverage_target
    winkler = widths.copy()
    below = y < lower_values
    above = y > upper_values
    winkler[below] += (2.0 / alpha) * (lower_values[below] - y[below])
    winkler[above] += (2.0 / alpha) * (y[above] - upper_values[above])

    n_points = len(y)
    n_covered = int(np.sum(covered))
    empirical_coverage = n_covered / n_points
    ci_lower, ci_upper = wilson_interval(n_covered, n_points)
    mean_width = float(np.mean(widths))
    coverage_gap = empirical_coverage - coverage_target
    return {
        "coverage": float(empirical_coverage),
        "width": mean_width,
        "winkler": float(np.mean(winkler)),
        "coverage_gap": float(coverage_gap),
        "calibration_error": float(abs(coverage_gap)),
        "n_points": n_points,
        "coverage_ci_lower": float(ci_lower),
        "coverage_ci_upper": float(ci_upper),
    }


def seasonal_insample_scales(history: np.ndarray, lag: int) -> Tuple[float, float]:
    """MAE/RMSE 的 in-sample 季节差分缩放因子（MASE/RMSSE 分母）。

    ``history``: ``(N, T)`` 原始域历史（逐 series 沿时间轴差分，不跨 series 边界）；
    ``lag``: 季节滞后阶数（与 seasonal-naive 基线同一 lag）。差分有效点为零或
    尺度近似零时返回 ``(nan, nan)``，由调用方按 NaN 传播；历史长度 <= lag
    属于显式误用，RAISE。
    """
    values = np.asarray(history, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"insample history must be 2D (N, T), got shape {values.shape}")
    lag = int(lag)
    if lag <= 0:
        raise ValueError(f"seasonal lag must be a positive integer, got {lag}")
    if values.shape[1] <= lag:
        raise ValueError(
            f"insample history length {values.shape[1]} must exceed seasonal lag {lag}"
        )
    diffs = values[:, lag:] - values[:, :-lag]
    finite = diffs[np.isfinite(diffs)]
    if finite.size == 0:
        return float("nan"), float("nan")
    mae_scale = float(np.mean(np.abs(finite)))
    rmse_scale = float(np.sqrt(np.mean(np.square(finite))))
    eps = np.finfo(float).eps
    return (
        mae_scale if mae_scale > eps else float("nan"),
        rmse_scale if rmse_scale > eps else float("nan"),
    )


def crps_from_pinball(
    quantile_levels: Tuple[float, ...],
    pinball_means: Tuple[float, ...],
) -> float:
    """CRPS 的分位梯形积分近似：``2 ∫ pinball_q dq``。

    离散分位网格上的标准近似（levels 升序、至少 2 个）；网格越密越接近真 CRPS。
    """
    levels = tuple(float(level) for level in quantile_levels)
    pinballs = tuple(float(value) for value in pinball_means)
    if len(levels) != len(pinballs):
        raise ValueError("quantile_levels and pinball_means must have equal length")
    if len(levels) < 2:
        raise ValueError("CRPS trapezoid requires at least two quantile levels")
    if any(left >= right for left, right in zip(levels, levels[1:])):
        raise ValueError("quantile_levels must be strictly increasing")
    if any(not math.isfinite(value) for value in pinballs):
        raise ValueError("pinball_means must be finite")
    return float(2.0 * np.trapezoid(np.asarray(pinballs), x=np.asarray(levels)))


def crossing_metrics(
    raw_quantiles: np.ndarray,
    quantile_levels: Tuple[float, ...],
    processed_quantiles: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """统计 raw crossing 及后处理实际改动比例。"""
    raw = np.asarray(raw_quantiles, dtype=float)
    levels = tuple(float(level) for level in quantile_levels)
    if raw.ndim != 2 or raw.shape[1] != len(levels):
        raise ValueError(
            "raw_quantiles must have shape (n_points, n_quantiles) matching quantile_levels"
        )
    if len(levels) < 2:
        raise ValueError("crossing metrics require at least two quantiles")
    if any(left >= right for left, right in zip(levels, levels[1:])):
        raise ValueError("quantile_levels must be strictly increasing")
    if not np.isfinite(raw).all():
        raise ValueError("raw_quantiles must contain only finite values")

    violations = raw[:, :-1] - raw[:, 1:]
    crossed = violations > 0.0
    result = {
        "row_crossing_rate": float(np.mean(np.any(crossed, axis=1))),
        "adjacent_crossing_rate": float(np.mean(crossed)),
        "crossing_magnitude": float(np.mean(np.maximum(violations, 0.0))),
        "repair_changed_ratio": 0.0,
        "q50_changed_ratio": 0.0,
    }
    if processed_quantiles is None:
        return result

    processed = np.asarray(processed_quantiles, dtype=float)
    if processed.shape != raw.shape:
        raise ValueError(
            f"processed_quantiles shape mismatch: raw={raw.shape}, processed={processed.shape}"
        )
    changed = ~np.isclose(processed, raw, rtol=0.0, atol=1e-12)
    median_matches = [
        index
        for index, level in enumerate(levels)
        if math.isclose(level, 0.5, rel_tol=0.0, abs_tol=1e-12)
    ]
    if len(median_matches) != 1:
        raise ValueError("crossing metrics require exactly one q50 level")
    result["repair_changed_ratio"] = float(np.mean(changed))
    result["q50_changed_ratio"] = float(np.mean(changed[:, median_matches[0]]))
    return result
