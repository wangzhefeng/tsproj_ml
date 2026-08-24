# -*- coding: utf-8 -*-
"""Conformalized Quantile Regression 的严格数学内核。"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from probabilistic.spec import validate_cqr_params


@dataclass(frozen=True)
class CalibrationRecord:
    """一条带完整时序可得性信息的 CQR 校准记录。"""

    forecast_origin: pd.Timestamp
    target_time: pd.Timestamp
    label_available_at: pd.Timestamp
    horizon_step: int
    window: int
    interval_name: str
    lower: float
    upper: float
    y_true: float
    score: float
    stage: str


@dataclass(frozen=True)
class CalibrationResult:
    """一次 as-of CQR 校准的结果和审计状态。"""

    lower: Optional[np.ndarray]
    upper: Optional[np.ndarray]
    correction: Optional[float]
    selected_records: Tuple[CalibrationRecord, ...]
    selected_windows: int
    selected_scores: int
    status: str
    reason: str


def select_calibration_records(
    records: Sequence[CalibrationRecord],
    forecast_origin: pd.Timestamp,
    n_windows: int,
) -> List[CalibrationRecord]:
    """按 as-of 语义选择最新的完整历史窗口，并按 target time 去重。"""
    window_limit = int(n_windows)
    if window_limit <= 0:
        raise ValueError("n_windows must be > 0")
    current_origin = pd.Timestamp(forecast_origin)
    if bool(pd.isna(current_origin)):
        raise ValueError("forecast_origin must be a finite timestamp")
    grouped = {}
    for record in records:
        key = (int(record.window), pd.Timestamp(record.forecast_origin))
        grouped.setdefault(key, []).append(record)

    eligible = []
    for (_, origin), window_records in grouped.items():
        latest_label = max(pd.Timestamp(record.label_available_at) for record in window_records)
        if bool(pd.isna(origin)) or bool(pd.isna(latest_label)):
            raise ValueError("calibration records require finite timestamps")
        if origin.value < current_origin.value and latest_label.value <= current_origin.value:
            eligible.append((origin, window_records))
    eligible.sort(key=lambda item: item[0], reverse=True)

    selected: List[CalibrationRecord] = []
    seen_targets = set()
    for _, window_records in eligible[:window_limit]:
        for record in sorted(window_records, key=lambda item: pd.Timestamp(item.target_time)):
            target_time = pd.Timestamp(record.target_time)
            if target_time in seen_targets:
                continue
            selected.append(record)
            seen_targets.add(target_time)
    return selected


def calibrate_with_records(
    q_low: np.ndarray,
    q_high: np.ndarray,
    records: Sequence[CalibrationRecord],
    forecast_origin: pd.Timestamp,
    n_windows: int,
    min_windows: int,
    min_scores: int,
    alpha: float,
    allow_interval_shrink: bool = False,
) -> CalibrationResult:
    """用满足 as-of 约束的完整历史窗口执行 CQR 双门槛校准。"""
    required_windows = int(min_windows)
    if required_windows <= 0:
        raise ValueError("min_windows must be > 0")
    validate_cqr_params(alpha, min_scores)
    selected = select_calibration_records(records, forecast_origin, n_windows)
    selected_window_count = len(
        {(record.window, pd.Timestamp(record.forecast_origin)) for record in selected}
    )
    finite_scores = np.asarray(
        [record.score for record in selected if np.isfinite(record.score)],
        dtype=float,
    )
    if selected_window_count < required_windows:
        return CalibrationResult(
            lower=None,
            upper=None,
            correction=None,
            selected_records=tuple(selected),
            selected_windows=selected_window_count,
            selected_scores=len(finite_scores),
            status="insufficient_windows",
            reason=f"selected {selected_window_count} windows < min_windows={required_windows}",
        )
    if len(finite_scores) < int(min_scores):
        return CalibrationResult(
            lower=None,
            upper=None,
            correction=None,
            selected_records=tuple(selected),
            selected_windows=selected_window_count,
            selected_scores=len(finite_scores),
            status="insufficient_scores",
            reason=f"selected {len(finite_scores)} scores < min_scores={int(min_scores)}",
        )

    lower, upper, correction = calibrate_quantile_band(
        q_low,
        q_high,
        finite_scores,
        alpha=alpha,
        min_scores=min_scores,
        allow_interval_shrink=allow_interval_shrink,
    )
    return CalibrationResult(
        lower=lower,
        upper=upper,
        correction=correction,
        selected_records=tuple(selected),
        selected_windows=selected_window_count,
        selected_scores=len(finite_scores),
        status="applied",
        reason="",
    )

def _as_1d_finite_array(values, name: str, allow_nonfinite: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not allow_nonfinite and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def compute_nonconformity_scores(
    y_true: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
) -> np.ndarray:
    """逐点计算 CQR score；区间内的 score 可以为负。"""
    y = _as_1d_finite_array(y_true, "y_true")
    lower = _as_1d_finite_array(q_low, "q_low")
    upper = _as_1d_finite_array(q_high, "q_high")
    if not (len(y) == len(lower) == len(upper)):
        raise ValueError(
            "CQR score length mismatch: "
            f"y_true={len(y)}, q_low={len(lower)}, q_high={len(upper)}"
        )
    if np.any(lower > upper):
        raise ValueError("CQR score requires q_low <= q_high at every point")
    return np.maximum(lower - y, y - upper)


def calibrate_quantile_band(
    q_low: np.ndarray,
    q_high: np.ndarray,
    scores: np.ndarray,
    alpha: float,
    min_scores: int = 30,
    allow_interval_shrink: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """用历史 CQR score 对基础区间做对称校准。"""
    alpha_value, min_score_count = validate_cqr_params(alpha, min_scores)
    lower = np.asarray(q_low, dtype=float).reshape(-1)
    upper = np.asarray(q_high, dtype=float).reshape(-1)
    if len(lower) != len(upper):
        raise ValueError(
            f"CQR calibration length mismatch: q_low={len(lower)}, q_high={len(upper)}"
        )
    score_values = np.asarray(scores, dtype=float).reshape(-1)
    score_values = score_values[np.isfinite(score_values)]
    if len(score_values) < min_score_count:
        return None, None, None

    n_scores = len(score_values)
    quantile_level = min(
        1.0,
        np.ceil((1.0 - alpha_value) * (n_scores + 1)) / n_scores,
    )
    correction = float(np.quantile(score_values, quantile_level, method="higher"))
    if not allow_interval_shrink:
        correction = max(0.0, correction)
    return lower - correction, upper + correction, correction


def attach_cqr_interval_columns(
    frame: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    target_coverage: float,
) -> pd.DataFrame:
    """追加独立 CQR prediction interval 列，不覆盖模型 quantile。"""
    coverage = float(target_coverage)
    validate_cqr_params(alpha=1.0 - coverage, min_scores=1)
    lower_values = _as_1d_finite_array(lower, "lower")
    upper_values = _as_1d_finite_array(upper, "upper")
    if not (len(frame) == len(lower_values) == len(upper_values)):
        raise ValueError(
            "CQR interval output length mismatch: "
            f"frame={len(frame)}, lower={len(lower_values)}, upper={len(upper_values)}"
        )
    if np.any(lower_values > upper_values):
        raise ValueError("CQR interval output requires lower <= upper at every point")

    coverage_percent = coverage * 100.0
    if np.isclose(coverage_percent, round(coverage_percent), atol=1e-10):
        token = str(int(round(coverage_percent)))
    else:
        token = f"{coverage_percent:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    result = frame.copy()
    result[f"predict_pi{token}_lower"] = lower_values
    result[f"predict_pi{token}_upper"] = upper_values
    return result
