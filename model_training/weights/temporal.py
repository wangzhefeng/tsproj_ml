"""显式训练时间权重：纯算法实现（指数衰减族）。

合同边界（docstring 即合同）：训练样本权重改变拟合损失中对样本的侧重，
不以评估掩码或数据删除替代加权；权重只由 (origins, 锚点, spec) 决定，
与模型、特征无关。锚点时间必须不早于任一 origin——拒绝未来监督原点。
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from forecasting_core.specs.training import resolve_sample_weight_spec

def temporal_sample_weight(
    origins,
    anchor_time,
    spec: Mapping | None,
) -> np.ndarray | None:
    """按 origin 年龄生成指数衰减权重。

    权重公式：``w = exp2(-(age - age_min) / halflife_days)``，age 为
    锚点与样本原点的时间差（天）。先平移到最新样本（保证至少一个
    权重为 1，避免长历史下整批下溢），再按 normalization 归一。
    anchor 约束可得性上界；合法锚点的常数平移不改变相对权重。

    Args:
        origins: 每个训练样本的监督原点时间（可为 DatetimeIndex 或可转
            pd.Timestamp 序列）。
        anchor_time: 年龄锚点。spec 为 None 时该参数被忽略；否则必须
            不早于任一 origin（未来监督原点 RAISE）。
        spec: 配置字典；None 表示未声明加权，直接返回 None。

    Returns:
        长度等于 origins 的权重数组；spec 为 None 时返回 None。
    """
    if spec is None:
        return None
    half, _anchor_mode, normalization = resolve_sample_weight_spec(spec)
    times = pd.DatetimeIndex(origins)
    end = pd.Timestamp(anchor_time)
    after_anchor = bool(np.any(np.asarray(times > end)))
    if times.empty or times.hasnans or pd.isna(end) or after_anchor:
        raise ValueError("sample_weight origins must be nonempty, valid and not after anchor")
    ages = np.asarray((end - times).total_seconds(), dtype=float) / 86400.0
    weights = np.exp2(-(ages - ages.min()) / half)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("sample_weight underflow or invalid weights")
    if normalization == "mean":
        return weights / weights.mean()
    if normalization == "sum":
        return weights / weights.sum()
    return weights  # none：保留原始尺度（最新样本=1）
