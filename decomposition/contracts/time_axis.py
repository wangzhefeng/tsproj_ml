"""分解专用规则时间轴与有限目标验证。"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd


def to_datetime_index(values: Any) -> pd.DatetimeIndex:
    times = pd.DatetimeIndex(pd.to_datetime(values))
    if times.hasnans:
        raise ValueError("Target decomposition times must not contain NaT.")
    return times


def validate_regular_times(times: pd.DatetimeIndex) -> int:
    if len(times) < 2:
        raise ValueError("Target decomposition requires at least two observations.")
    if times.hasnans:
        raise ValueError("Target decomposition times must not contain NaT.")
    diffs = np.diff(times.asi8)
    if np.any(diffs <= 0):
        raise ValueError("Target decomposition requires strictly increasing timestamps.")
    step = int(np.median(diffs))
    if np.any(diffs != step):
        raise ValueError("Target decomposition requires a regular time index.")
    return step


def elapsed_days(times: pd.DatetimeIndex, origin_ns: int) -> np.ndarray:
    return (times.asi8 - origin_ns) / 86400.0 / 1e9


def read_history(history: pd.DataFrame, time_col: str, target_col: str):
    if time_col not in history or target_col not in history:
        raise ValueError(f"Target decomposition requires columns: {time_col}, {target_col}.")
    times = to_datetime_index(history[time_col])
    validate_regular_times(times)
    y = pd.to_numeric(history[target_col], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Target decomposition does not accept NaN or infinite target values.")
    return times, y
