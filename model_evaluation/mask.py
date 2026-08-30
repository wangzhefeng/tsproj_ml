# -*- coding: utf-8 -*-
"""评估掩码构造（percentile/absolute/combined 三模式）。

MAPE / Accuracy 与概率指标（pinball/区间）共用本模块，保证同一业务口径
（低负荷时段不计入评估）。实现自 `model_testing/backtest.py` 迁入（2026-08-30
evaluation 模块化；最初自 utils/eval_mask.py 迁入，逻辑逐字保真）。
"""

from typing import Any, Dict

import numpy as np


def build_eval_mask(
    values: Any,
    *,
    mode: str = "percentile",
    percentile: float = 5.0,
    min_value: float | None = None,
    max_value: float | None = None,
) -> Dict[str, Any]:
    """构造有效点掩码。

    三种模式：
    - ``percentile``：``valid = y >= P{percentile} of positive values``（默认，向后兼容）
    - ``absolute``：``valid = y >= min_value``（需设置 ``min_value``；适合正常带窄、
      异常明显偏低的数据，能精准砍掉断点而不误伤正常低谷）
    - ``combined``：``valid = y >= max(P{percentile}, min_value)``

    Args:
        values: 一维序列（真值或历史值）。
        mode: ``percentile`` | ``absolute`` | ``combined``。
        percentile: percentile/combined 模式下正样本的下分位（%）。
        min_value: 绝对下限；``None`` 表示不启用绝对过滤。
        max_value: 绝对上限（与 mode 正交）；``None`` 表示不启用上限过滤，
            ``y > max_value`` 视为异常并被排除。

    Returns:
        dict(threshold, upper_threshold, valid_mask, valid_points, excluded_points,
        excluded_ratio)，``threshold`` 为下限，``upper_threshold`` 为上限。
    """
    values = np.asarray(values).reshape(-1)
    positive_values = values[values > 0]

    pct_threshold = (
        float(np.percentile(positive_values, percentile))
        if mode in ("percentile", "combined") and positive_values.size > 0
        else np.nan
    )
    abs_threshold = float(min_value) if min_value is not None else np.nan

    if mode == "percentile":
        threshold = pct_threshold
    elif mode == "absolute":
        threshold = abs_threshold
    elif mode == "combined":
        finite = [t for t in (pct_threshold, abs_threshold) if not np.isnan(t)]
        threshold = max(finite) if finite else np.nan
    else:
        raise ValueError(
            f"Unknown eval_mask mode: {mode!r} (expected percentile|absolute|combined)"
        )

    if np.isnan(threshold):
        # percentile 模式下无正值 → 全部排除；absolute 未设 min_value → 退化为「全部正值有效」
        if mode == "absolute" and min_value is None:
            valid_mask = values > 0
        else:
            valid_mask = np.zeros(len(values), dtype=bool)
    else:
        valid_mask = values >= threshold

    # 上限掩码：max_value 与 mode 正交，y > max_value 视为异常
    upper_threshold = float(max_value) if max_value is not None else np.nan
    if max_value is not None:
        valid_mask = valid_mask & (values <= upper_threshold)

    valid_points = int(valid_mask.sum())
    excluded_points = int(len(values) - valid_points)
    excluded_ratio = float(excluded_points / len(values)) if len(values) > 0 else np.nan

    return {
        "threshold": threshold,
        "upper_threshold": upper_threshold,
        "valid_mask": valid_mask,
        "valid_points": valid_points,
        "excluded_points": excluded_points,
        "excluded_ratio": excluded_ratio,
    }


__all__ = ["build_eval_mask"]
