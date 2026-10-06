"""历史统计纯数值内核：调用方负责历史可见性、配置校验和特征命名。

保持 pandas 统计定义；单窗未定义统计报错，不持有序列、origin 或拟合状态。
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd

from feature_engineering.kernels.spectral import signal_entropy


def rolling_statistics(
    history: pd.Series,
    window: int,
    stats: Sequence[str],
) -> dict[str, pd.Series]:
    results: dict[str, pd.Series] = {}
    rolling = history.rolling(window, min_periods=1)
    for stat in stats:
        if stat == "std":
            # 平移后计算避免大直流分量、小波动导致增量方差消减。
            results[stat] = (
                (history - history.iloc[0]).rolling(window, min_periods=1).std().fillna(0.0)
            )
            continue
        if stat == "entropy":
            # pandas 的返回标注包含 DataFrame；此处输入明确为 Series。
            results[stat] = cast(pd.Series, rolling.apply(signal_entropy, raw=True)).fillna(0.0)
            continue
        if stat in {"max_diff", "min_diff"}:
            if window < 2:
                results[stat] = pd.Series(0.0, index=history.index)
            else:
                diffs = history.diff()
                method = "max" if stat == "max_diff" else "min"
                results[stat] = getattr(
                    diffs.rolling(window - 1, min_periods=1), method
                )().fillna(0.0)
            continue
        if stat not in {"mean", "std", "min", "max", "median", "skew", "kurt"}:
            raise ValueError(f"unsupported history statistic: {stat!r}")
        results[stat] = getattr(rolling, stat)().fillna(0.0)
        if stat == "kurt":
            # pandas窗口峰度对恒定窗返回-3；保持单窗Series.kurt的0语义。
            results[stat] = results[stat].mask(rolling.max().eq(rolling.min()), 0.0)
    return results


def expanding_statistics(
    history: pd.Series,
    stats: Sequence[str],
) -> dict[str, pd.Series]:
    """按当前历史下界一次计算全部前缀，不逐原点重扫历史。"""
    results: dict[str, pd.Series] = {}
    for stat in stats:
        if stat == "std":
            values = (history - history.iloc[0]).expanding().std()
        elif stat in {"max_diff", "min_diff"}:
            method = "max" if stat == "max_diff" else "min"
            values = getattr(history.diff().expanding(), method)()
        elif stat == "entropy":
            values = cast(pd.Series, history.expanding().apply(signal_entropy, raw=True))
        elif stat in {"mean", "min", "max", "median", "skew", "kurt"}:
            values = getattr(history.expanding(), stat)()
        else:
            raise ValueError(f"unsupported history statistic: {stat!r}")
        results[stat] = values.fillna(0.0)
        if stat == "kurt":
            results[stat] = results[stat].mask(history.cummax().eq(history.cummin()), 0.0)
    return results


def history_statistic(values: pd.Series, stat: str) -> float:
    if stat == "entropy":
        # 香农熵（p = |y|/sum|y|）；非常量窗内分布越均匀熵越高。
        return signal_entropy(values.to_numpy())
    if stat in {"max_diff", "min_diff"}:
        # 爬坡统计必须存在至少一对真实相邻观测，不以0替代未定义值。
        if len(values) < 2:
            raise ValueError(f"{stat} requires at least 2 history samples")
        diffs = values.diff().dropna()
        return float(diffs.max() if stat == "max_diff" else diffs.min())
    supported = {"mean", "std", "min", "max", "median", "skew", "kurt"}
    if stat not in supported:
        raise ValueError(f"unsupported history statistic: {stat!r}")
    result = getattr(values, stat)()
    if pd.isna(result):
        raise ValueError(f"history statistic {stat!r} is undefined for {len(values)} samples")
    return float(result)


def time_since_event(values: pd.Series, event: str) -> float:
    if event == "peak":
        mask = (values.shift(1) < values) & (values > values.shift(-1))
    elif event == "trough":
        mask = (values.shift(1) > values) & (values < values.shift(-1))
    else:
        raise ValueError(f"unsupported time-since event: {event!r}")
    indices = np.flatnonzero(mask.to_numpy())
    position = len(values) - 1
    prior = indices[indices < position]
    return float(position - prior[-1] if len(prior) else position)
