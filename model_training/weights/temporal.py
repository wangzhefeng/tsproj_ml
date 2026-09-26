"""显式训练时间权重：纯算法实现（指数衰减族）。

合同边界（docstring 即合同）：训练样本权重改变拟合损失中对样本的侧重，
不以评估掩码或数据删除替代加权；权重只由 (origins, 锚点, spec) 决定，
与模型、特征无关。锚点时间必须不早于任一 origin——拒绝未来监督原点。
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

# 归一化模式：mean=均值 1（默认，样本总量语义不变）；sum=总和 1
#（每样本期望贡献固定）；none=不归一（保留 exp2 原始尺度，最老样本=1）。
_NORMALIZATION_MODES = frozenset({"mean", "sum", "none"})

# 年龄锚点：cutoff=训练历史截止（标签可见性边界，回测几何口径）；
# latest_origin=最新监督原点（样本网格口径）。两者在「origins 含跨多折
# 拉通的长窗」时结果不同；默认 cutoff 与回测几何一致。
_ANCHOR_MODES = frozenset({"cutoff", "latest_origin"})


def _resolve_spec(spec: Mapping) -> tuple[float, str, str]:
    """校验 sample_weight spec 并返回 (halflife_days, anchor, normalization)。"""
    if not isinstance(spec, Mapping) or set(spec) - {
        "method", "halflife_days", "anchor", "normalization",
    }:
        raise ValueError(
            "sample_weight requires method/halflife_days with optional "
            "anchor/normalization only"
        )
    half = spec["halflife_days"]
    if spec.get("method", "exponential") != "exponential" or isinstance(half, bool) or not isinstance(half, (int, float)):
        raise ValueError("sample_weight requires exponential method and numeric halflife_days")
    if not np.isfinite(half) or half <= 0:
        raise ValueError("sample_weight halflife_days must be finite and positive")
    anchor = str(spec.get("anchor", "cutoff"))
    if anchor not in _ANCHOR_MODES:
        raise ValueError(
            f"sample_weight anchor must be one of {sorted(_ANCHOR_MODES)}; got {anchor!r}"
        )
    normalization = str(spec.get("normalization", "mean"))
    if normalization not in _NORMALIZATION_MODES:
        raise ValueError(
            "sample_weight normalization must be one of "
            f"{sorted(_NORMALIZATION_MODES)}; got {normalization!r}"
        )
    return float(half), anchor, normalization


def temporal_sample_weight(
    origins,
    anchor_time,
    spec: Mapping | None,
) -> np.ndarray | None:
    """按 origin 年龄生成指数衰减权重（半衰期以天计，2 的幂次衰减）。

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
    half, anchor_mode, normalization = _resolve_spec(spec)
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
    return weights  # none：保留原始尺度（最老样本=1）
