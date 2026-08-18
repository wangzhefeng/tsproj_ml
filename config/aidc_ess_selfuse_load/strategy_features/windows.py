"""5 分钟自然日槽位、调度周期映射与时间戳校验。"""

from dataclasses import dataclass
from typing import cast

import pandas as pd


SLOT_MINUTES = 5
SLOTS_PER_DAY = 24 * 60 // SLOT_MINUTES
DISPATCH_START_HOUR = 22


@dataclass(frozen=True)
class HistoryTimestampAudit:
    """历史时间戳完整性审计结果，不因局部缺失阻断处理。"""

    incomplete_days: tuple[pd.Timestamp, ...]
    missing_slots_by_day: dict[pd.Timestamp, tuple[int, ...]]
    has_duplicates: bool
    has_off_grid_timestamps: bool


def _as_datetime_index(timestamps) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(timestamps)
    if index.hasnans:
        raise ValueError("timestamps must not contain NaT")
    return index


def _off_grid_mask(index: pd.DatetimeIndex):
    return (
        (index.minute % SLOT_MINUTES != 0)
        | (index.second != 0)
        | (index.microsecond != 0)
        | (index.nanosecond != 0)
    )


def calendar_day_slot(timestamps) -> pd.Index:
    """返回自然日 5 分钟槽位：00:00 为 0，23:55 为 287。"""
    index = _as_datetime_index(timestamps)
    return pd.Index((index.hour * 60 + index.minute) // SLOT_MINUTES, dtype="int64")


def dispatch_cycle_start(timestamps) -> pd.DatetimeIndex:
    """把时间戳映射到所属的 [22:00, 次日 21:55] 调度周期起点。"""
    index = _as_datetime_index(timestamps)
    same_day_start = index.normalize() + pd.Timedelta(hours=DISPATCH_START_HOUR)
    return pd.DatetimeIndex(
        same_day_start.where(index >= same_day_start, same_day_start - pd.Timedelta(days=1))
    )


def dispatch_cycle_slot(timestamps) -> pd.Index:
    """返回调度周期槽位：周期起点 22:00 为 0，次日 21:55 为 287。"""
    index = _as_datetime_index(timestamps)
    elapsed = index - dispatch_cycle_start(index)
    return pd.Index(elapsed.total_seconds() // (SLOT_MINUTES * 60), dtype="int64")


def validate_future_timestamps(timestamps) -> None:
    """未来数据必须是完整、唯一、递增且严格对齐的自然日 5 分钟序列。"""
    index = _as_datetime_index(timestamps)
    if len(index) == 0:
        raise ValueError("future timestamps must not be empty")
    if index.has_duplicates:
        raise ValueError("future timestamps contain duplicates")
    if not index.is_monotonic_increasing:
        raise ValueError("future timestamps must be increasing")
    if _off_grid_mask(index).any():
        raise ValueError("future timestamps must align to the 5-minute grid")

    for day in index.normalize().unique():
        expected = pd.date_range(day, periods=SLOTS_PER_DAY, freq="5min")
        actual = index[index.normalize() == day]
        if not actual.equals(expected):
            raise ValueError(f"future day {day.date()} is incomplete")


def audit_history_timestamps(timestamps) -> HistoryTimestampAudit:
    """审计历史自然日完整性；缺失、重复或偏离网格均只记录不抛错。"""
    index = _as_datetime_index(timestamps)
    missing_slots_by_day: dict[pd.Timestamp, tuple[int, ...]] = {}

    if len(index) == 0:
        calendar_days = pd.DatetimeIndex([])
    else:
        first_day = cast(pd.Timestamp, index.min()).normalize()
        last_day = cast(pd.Timestamp, index.max()).normalize()
        calendar_days = pd.date_range(first_day, last_day, freq="1D")

    for day in calendar_days:
        day_timestamp = pd.Timestamp(day)
        actual_slots = set(calendar_day_slot(index[index.normalize() == day]).tolist())
        missing_slots = tuple(sorted(set(range(SLOTS_PER_DAY)) - actual_slots))
        if missing_slots:
            missing_slots_by_day[day_timestamp] = missing_slots

    return HistoryTimestampAudit(
        incomplete_days=tuple(sorted(missing_slots_by_day)),
        missing_slots_by_day=missing_slots_by_day,
        has_duplicates=index.has_duplicates,
        has_off_grid_timestamps=bool(_off_grid_mask(index).any()),
    )
