# -*- coding: utf-8 -*-

"""
时间频率工具。
"""

from pathlib import Path

import numpy as np
from pandas.tseries.frequencies import to_offset


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

# 月频（非固定时长）频率集合：pandas 月频 offset 无 .nanos，需单独处理
_MONTHLY_FREQS = {"1ME", "1MS", "ME", "MS"}


def is_monthly_freq(freq: str) -> bool:
    """判断是否为月频（非固定时长）频率。"""
    return str(freq) in _MONTHLY_FREQS


def resolve_freq_step_minutes(freq: str) -> float:
    """
    解析 pandas 频率字符串对应的步长（分钟）。

    要求频率必须是固定时长，并且不超过 1 天。
    月频（1ME/1MS）是非固定时长 offset，用 30 天近似，仅供 future_time 推算。
    """
    if is_monthly_freq(freq):
        return 30 * 24 * 60  # 近似 30 天 = 43200 分钟

    try:
        offset = to_offset(freq)
    except ValueError as exc:
        raise ValueError(f"Invalid pandas frequency: {freq}") from exc

    try:
        step_nanos = offset.nanos
    except (ValueError, TypeError, NotImplementedError) as exc:
        raise ValueError(
            f"Frequency '{freq}' is not a fixed-duration offset. "
            "Please use fixed frequencies such as '5min', '15min', '30min', '1h', or '1D'."
        ) from exc

    step_minutes = step_nanos / (60 * 1_000_000_000)
    if step_minutes <= 0:
        raise ValueError(f"Frequency '{freq}' must be positive.")
    if step_minutes > 24 * 60:
        raise ValueError(f"Frequency '{freq}' must not exceed 1 day for this project.")

    return int(step_minutes)


def resolve_samples_per_day(freq: str) -> int:
    """
    根据 pandas 频率字符串计算一天内的样本数。

    当前项目要求一天能被频率整除，例如：
    - 5min -> 288
    - 15min -> 96
    - 1h -> 24
    - 1D -> 1
    月频（1ME/1MS）特判返回 1（每个月 1 个点，类比日频 n_per_day=1）。
    """
    if is_monthly_freq(freq):
        return 1

    day_minutes = 24 * 60
    step_minutes = resolve_freq_step_minutes(freq)
    quotient = day_minutes / step_minutes

    if int(round(quotient)) != quotient:
        raise ValueError(
            f"Frequency '{freq}' does not evenly divide one day. "
            "Please use a fixed frequency that evenly partitions 24 hours."
        )

    return int(quotient)


def compute_time_decay_weights(
    n_samples: int,
    n_per_day: int,
    halflife_days: float,
) -> "np.ndarray":
    """
    指数时间衰减样本权重:越近期的样本权重越大,用于抑制概念漂移导致的远期噪声。

    权重定义为 ``w_i = exp(-λ * age_i)``,其中 ``λ = ln(2) / (halflife_days * n_per_day)``,
    ``age`` 以"距序列末尾的步数"度量(末尾样本 age=0,权重最大;最老样本 age=n_samples-1)。
    最终权重归一化到均值 1,避免改变有效学习率量级。

    Args:
        n_samples: 训练样本行数。
        n_per_day: 每天样本数(由 ``resolve_samples_per_day`` 得到)。
        halflife_days: 半衰期(天);权重衰减到一半所需的样本年龄。

    Returns:
        一维权重数组,长度为 ``n_samples``,均值≈1。
    """
    import numpy as np

    if n_samples <= 0:
        return np.asarray([], dtype=float)
    halflife_steps = max(1.0, float(halflife_days) * float(n_per_day))
    lam = np.log(2) / halflife_steps
    # 末尾样本 age=0(最新),最老样本 age=n_samples-1
    ages = np.arange(n_samples - 1, -1, -1, dtype=float)
    weights = np.exp(-lam * ages)
    mean_w = weights.mean()
    if mean_w > 0:
        weights = weights / mean_w  # 归一化到均值1,保持学习率量级
    return weights
