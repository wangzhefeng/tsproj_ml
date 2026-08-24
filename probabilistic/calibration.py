# -*- coding: utf-8 -*-
"""Conformalized Quantile Regression 的严格数学内核。"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from probabilistic.spec import validate_cqr_params

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
