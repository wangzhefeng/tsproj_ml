# -*- coding: utf-8 -*-

# ***************************************************
# * File        : conformal.py
# * Description : Conformalized Quantile Regression (CQR) 后处理校准。
# *               在滑窗测试 nonconformity scores 上取分位，对 q_low/q_high
# *               做对称膨胀，保证边际覆盖率 ≥ 1−α。纯后处理，不重训模型。
# * Reference   : Romano, Patterson, Candès (2019), "Conformalized Quantile Regression"
# * ***************************************************

from typing import Optional, Tuple

import numpy as np


def compute_nonconformity_scores(
    y_true: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
) -> np.ndarray:
    """
    CQR nonconformity score：逐点取 q_low 欠估量与 q_high 过估量的较大者。

    score >= 0；score = 0 表示 y ∈ [q_low, q_high]（区间已覆盖该点）。

    Args:
        y_true: 真实值，1D 数组。
        q_low:  预测下界（低分位数），1D 数组。
        q_high: 预测上界（高分位数），1D 数组。

    Returns:
        1D nonconformity scores，长度 = min(len(y_true), len(q_low), len(q_high))。
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    q_low = np.asarray(q_low, dtype=float).flatten()
    q_high = np.asarray(q_high, dtype=float).flatten()
    n = min(len(y_true), len(q_low), len(q_high))
    return np.maximum(q_low[:n] - y_true[:n], y_true[:n] - q_high[:n])


def calibrate_quantile_band(
    q_low: np.ndarray,
    q_high: np.ndarray,
    scores: np.ndarray,
    alpha: float,
    min_scores: int = 30,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    用 nonconformity scores 对分位数带做对称膨胀。

    E_α 采用 "plus one" 修正分位：level = ⌈(1−α)(n+1)⌉ / n，
    这是 split conformal 在有限样本下的严格覆盖保证。

    Args:
        q_low:  待校准的下界预测。
        q_high: 待校准的上界预测。
        scores: 滑窗测试阶段累积的 nonconformity scores。
        alpha:  目标误覆盖率（1−α = 目标覆盖率，如 0.1 → 90% 覆盖）。
        min_scores: 校准集最少 score 数；不足则返回 (None, None, None)。

    Returns:
        (calibrated_q_low, calibrated_q_high, E_alpha)；
        样本不足时返回 (None, None, None)，调用方应跳过校准。
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) < min_scores:
        return None, None, None

    n = len(scores)
    quantile_level = min(1.0, np.ceil((1.0 - alpha) * (n + 1)) / n)
    E_alpha = float(np.quantile(scores, quantile_level, method="higher"))

    q_low_cal = np.asarray(q_low, dtype=float) - E_alpha
    q_high_cal = np.asarray(q_high, dtype=float) + E_alpha
    return q_low_cal, q_high_cal, E_alpha
