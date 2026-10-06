# -*- coding: utf-8 -*-
"""同槽/近期状态/残差基线的纯函数内核。

全部严格 as-of：任何越界（越过 origin 或越过历史起点）直接 RAISE，
不做静默截断或缩窗。周期按整数步数（5min 每日周期 = 288），
不做任何粒度转换。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

_SLOT_STATS = ("mean", "std")
_RECENT_STATS = ("level", "mean", "std", "diff", "slope")
_BASELINE_KEYS = ("column", "period", "days")
_SAME_SLOT_KEYS = ("columns", "period", "days", "stats")
_RECENT_STATE_KEYS = ("columns", "windows", "stats")



def _require_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _int_sequence(value: Any, name: str, *, minimum: int = 1) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of integers")
    normalized = tuple(_require_int(item, f"{name} entries", minimum=minimum) for item in value)
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(sorted(normalized))


def _stat_sequence(value: Any, name: str, allowed: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    if any(not isinstance(item, str) for item in value):
        raise TypeError(f"{name} entries must be strings")
    normalized = tuple(value)
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    unknown = sorted(set(normalized) - set(allowed))
    if unknown:
        raise ValueError(f"unsupported {name}: {unknown}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(sorted(normalized))


def _column_sequence(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    if any(not isinstance(item, str) for item in value):
        raise TypeError(f"{name} entries must be strings")
    normalized = tuple(value)
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if any(not item.strip() for item in normalized):
        raise ValueError(f"{name} entries must be non-blank")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} entries must be unique")
    return normalized


def normalize_seasonal_baseline_spec(value: Any) -> dict[str, Any]:
    """`transformations.seasonal_baseline` 严格归一化。"""
    if not isinstance(value, Mapping):
        raise TypeError("seasonal_baseline must be a mapping")
    unknown = set(value) - set(_BASELINE_KEYS)
    if unknown:
        raise ValueError(f"unknown seasonal_baseline keys: {sorted(unknown)}")
    column = value.get("column")
    if not isinstance(column, str) or not column.strip():
        raise ValueError("seasonal_baseline.column must be a non-blank string")
    return {
        "column": column,
        "period": _require_int(value.get("period"), "seasonal_baseline.period"),
        "days": _require_int(value.get("days"), "seasonal_baseline.days"),
    }


def normalize_same_slot_spec(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("advanced.same_slot must be a mapping")
    unknown = set(value) - set(_SAME_SLOT_KEYS)
    if unknown:
        raise ValueError(f"unknown advanced.same_slot keys: {sorted(unknown)}")
    return {
        "columns": _column_sequence(value.get("columns"), "same_slot.columns"),
        "period": _require_int(value.get("period"), "same_slot.period"),
        "days": _int_sequence(value.get("days"), "same_slot.days"),
        "stats": _stat_sequence(value.get("stats", ("mean",)), "same_slot.stats", _SLOT_STATS),
    }


def normalize_recent_state_spec(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("advanced.recent_state must be a mapping")
    unknown = set(value) - set(_RECENT_STATE_KEYS)
    if unknown:
        raise ValueError(f"unknown advanced.recent_state keys: {sorted(unknown)}")
    return {
        "columns": _column_sequence(value.get("columns"), "recent_state.columns"),
        "windows": _int_sequence(value.get("windows"), "recent_state.windows", minimum=2),
        "stats": _stat_sequence(
            value.get("stats", ("level",)), "recent_state.stats", _RECENT_STATS
        ),
    }


def _regular_history(values: pd.Series) -> tuple[pd.DatetimeIndex, np.ndarray]:
    times = values.index
    if not isinstance(times, pd.DatetimeIndex):
        raise TypeError("seasonal kernels require a DatetimeIndex history")
    if times.hasnans or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError("seasonal kernels require unique ordered history timestamps")
    array = values.to_numpy(dtype=float)
    if len(times) < 2 or not np.all(np.diff(times.asi8) == times.asi8[1] - times.asi8[0]):
        raise ValueError("seasonal kernels require a complete regular history grid")
    if not np.isfinite(array).all():
        raise ValueError("seasonal kernels require finite history values")
    return times, array


def _locate(times: pd.DatetimeIndex, moment: pd.Timestamp, *, as_of: pd.Timestamp) -> int:
    moment = pd.Timestamp(moment)
    as_of = pd.Timestamp(as_of)
    if moment > as_of:
        raise ValueError(f"seasonal slot moment {moment} is after the as-of origin {as_of}")
    position = int(np.searchsorted(times.asi8, moment.value, side="right")) - 1
    if position < 0 or times[position] != moment:
        raise ValueError(f"seasonal slot moment {moment} is not on the regular history grid")
    return position


def same_slot_stats(
    values: pd.Series,
    *,
    anchor: pd.Timestamp,
    origin: pd.Timestamp,
    period: int,
    days: Sequence[int],
) -> dict[int, dict[str, float]]:
    """锚点时刻的过去 k 天同槽统计；anchor 可晚于 origin（如监督训练行）。

    anchor 必须落在规则网格上；对每个 k 取 anchor - k×period，该时刻必须
    在历史范围内且不晚于 origin，否则 RAISE（严格 as-of，不静默降 k）。
    """
    times, array = _regular_history(values.loc[:origin])
    origin_position = _locate(times, origin, as_of=origin)
    delta = pd.Timestamp(anchor).value - times.asi8[0]
    frequency_ns = times.asi8[1] - times.asi8[0]
    if delta % frequency_ns:
        raise ValueError("same_slot anchor must be on the history grid")
    anchor_position = int(delta // frequency_ns)
    period = _require_int(period, "same_slot.period")
    days = _int_sequence(days, "same_slot.days")
    results: dict[int, dict[str, float]] = {}
    for k in days:
        positions = anchor_position - np.arange(1, k + 1) * period
        if (positions < 0).any() or (positions > origin_position).any():
            raise ValueError("same_slot requires every slot inside as-of visible history")
        slot_values = array[positions]
        stats: dict[str, float] = {}
        if len(slot_values) >= 1:
            stats["mean"] = float(np.mean(slot_values))
            stats["std"] = (
                float(np.std(slot_values, ddof=1)) if len(slot_values) >= 2 else 0.0
            )
        results[int(k)] = stats
    return results


def recent_state_stats(
    values: pd.Series,
    *,
    origin: pd.Timestamp,
    windows: Sequence[int],
) -> dict[int, dict[str, float]]:
    """原点前含原点的 window 个点，斜率单位为每步变化量。"""
    times, array = _regular_history(values.loc[:origin])
    origin_position = _locate(times, origin, as_of=origin)
    windows = _int_sequence(windows, "recent_state.windows", minimum=2)
    results: dict[int, dict[str, float]] = {}
    for window in windows:
        start = origin_position - int(window) + 1
        if start < 0:
            raise ValueError(
                f"recent_state window {window} exceeds visible history "
                f"(needs {int(window)} points up to the origin)"
            )
        segment = array[start : origin_position + 1]
        if len(segment) != int(window):
            raise ValueError("recent_state history grid is not regular")
        stats = {
            "level": float(segment[-1]),
            "mean": float(np.mean(segment)),
            "diff": float(segment[-1] - segment[0]),
            "slope": float((segment[-1] - segment[0]) / (len(segment) - 1)),
        }
        stats["std"] = (
            float(np.std(segment, ddof=1)) if len(segment) >= 2 else 0.0
        )
        results[int(window)] = stats
    return results


def seasonal_baseline_values(
    values: pd.Series,
    *,
    origin: pd.Timestamp,
    horizon: int,
    period: int,
    days: int,
) -> np.ndarray:
    """监督原点后 horizon 个标签时刻的 k 天同槽均值基线。

    标签时刻 = origin + (1..horizon) 步；其 k 天同槽 = 标签时刻 - m×period
    (m=1..k)，全部必须 ≤ origin 且在历史网格内，否则 RAISE。
    """
    times, array = _regular_history(values.loc[:origin])
    origin_position = _locate(times, origin, as_of=origin)
    horizon = _require_int(horizon, "seasonal_baseline.horizon")
    period = _require_int(period, "seasonal_baseline.period")
    days = _require_int(days, "seasonal_baseline.days")
    baseline = np.empty(int(horizon), dtype=float)
    for step in range(1, int(horizon) + 1):
        label_position = origin_position + step
        slot_values = []
        for m in range(1, int(days) + 1):
            position = label_position - m * int(period)
            if position < 0 or position > origin_position:
                raise ValueError(
                    f"seasonal baseline label step {step} day {m} slot is outside the "
                    "as-of visible history"
                )
            slot_values.append(float(array[position]))
        baseline[step - 1] = float(np.mean(slot_values))
    return baseline
