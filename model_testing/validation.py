"""Shared rolling-origin time contract for backtests and OOF folds.

Extracted in E1 from `model_forecasting/runtime.py` (`_label_start`, `_label_end`,
`_holdout_training_indices`, `_rolling_backtest_windows`). The functions here
are deliberately decoupled from any runtime type: they take explicit time
geometry (origin offset / horizon steps) so the ensemble OOF splitter (E3) can
reuse the exact same semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

import pandas as pd


def label_start(origin: pd.Timestamp, offset: pd.tseries.frequencies.BaseOffset) -> pd.Timestamp:
    """First timestamp a forecast made at ``origin`` predicts."""
    return origin + offset


def label_end(
    origin: pd.Timestamp,
    offset: pd.tseries.frequencies.BaseOffset,
    horizon: int,
) -> pd.Timestamp:
    """Last timestamp a forecast made at ``origin`` predicts."""
    return origin + horizon * offset


def is_label_safe(
    origin: pd.Timestamp,
    offset: pd.tseries.frequencies.BaseOffset,
    horizon: int,
    holdout_label_start: pd.Timestamp,
    *,
    gap_steps: int = 0,
) -> bool:
    """Require label_end strictly before the holdout's embargo boundary.

    A positive gap excludes grid steps before the first holdout label; it
    never shifts the forecast itself. Calendar offsets stay calendar offsets.
    """
    if isinstance(gap_steps, bool) or not isinstance(gap_steps, int) or gap_steps < 0:
        raise ValueError("gap_steps must be a non-negative integer")
    cutoff = holdout_label_start - gap_steps * offset if gap_steps else holdout_label_start
    return label_end(origin, offset, horizon) < cutoff


@dataclass(frozen=True, slots=True)
class TimeGeometry:
    """Explicit origin-step geometry shared by backtests and OOF folds."""

    offset: pd.tseries.frequencies.BaseOffset
    horizon: int

    def label_start(self, origin: pd.Timestamp) -> pd.Timestamp:
        return label_start(origin, self.offset)

    def label_end(self, origin: pd.Timestamp) -> pd.Timestamp:
        return label_end(origin, self.offset, self.horizon)


class OriginTimeline(Protocol):
    """Read-only time geometry required for selecting supervised folds."""

    @property
    def geometry(self) -> TimeGeometry: ...

    @property
    def supervised_origins(self) -> tuple[pd.Timestamp, ...]: ...


@dataclass(frozen=True, slots=True)
class CalendarMonthFold:
    """One complete calendar-month holdout with a fixed preceding day window."""

    window: int
    origin_index: int
    origin: pd.Timestamp
    train_indices: tuple[int, ...]
    forecast_times: pd.DatetimeIndex
    horizon: int
    metadata: dict[str, Any]


def calendar_month_folds(
    timestamps: Iterable[pd.Timestamp],
    *,
    train_window_days: int,
    fold_count: int,
    stride_months: int,
) -> tuple[CalendarMonthFold, ...]:
    """Build complete month-aligned folds in chronological order.

    Each fold predicts one whole month and trains on exactly the preceding
    ``train_window_days`` daily rows. The newest complete month anchors the
    sequence; preceding folds are spaced by ``stride_months`` calendar months.
    """
    if (
        isinstance(train_window_days, bool)
        or not isinstance(train_window_days, int)
        or train_window_days <= 0
    ):
        raise ValueError("train_window_days must be a positive integer")
    for field_name, value in (
        ("fold_count", fold_count),
        ("stride_months", stride_months),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")

    times = pd.DatetimeIndex(pd.to_datetime(tuple(timestamps)))
    if times.empty:
        return ()
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("calendar-month folds require ordered unique timestamps")
    normalized = times.normalize()
    if not times.equals(normalized):
        raise ValueError("calendar-month folds require normalized daily timestamps")
    expected = pd.date_range(times[0], times[-1], freq="1D")
    if not times.equals(expected):
        raise ValueError("calendar-month folds require a complete regular 1D index")

    candidates: list[tuple[pd.Timestamp, int, int]] = []
    current = times[-1].to_period("M")
    while len(candidates) < fold_count * stride_months:
        month_start = current.to_timestamp()
        month_end = (current + 1).to_timestamp()
        test_start = int(times.searchsorted(month_start, side="left"))
        test_end = int(times.searchsorted(month_end, side="left"))
        horizon = int(month_start.days_in_month)
        complete = (
            test_start < len(times)
            and times[test_start] == month_start
            and test_end - test_start == horizon
        )
        if complete and test_start >= train_window_days:
            candidates.append((month_start, test_start, test_end))
        if month_start <= times[0]:
            break
        current -= 1

    ordered = list(reversed(candidates[::stride_months][:fold_count]))
    folds = []
    for window, (month_start, test_start, test_end) in enumerate(
        ordered, start=1
    ):
        train_indices = tuple(range(test_start - train_window_days, test_start))
        forecast_times = times[test_start:test_end]
        origin_index = test_start - 1
        origin = pd.Timestamp(times[origin_index])
        folds.append(
            CalendarMonthFold(
                window=window,
                origin_index=origin_index,
                origin=origin,
                train_indices=train_indices,
                forecast_times=forecast_times,
                horizon=len(forecast_times),
                metadata={
                    "window": window,
                    "origin": origin.isoformat(),
                    "calendar_month": month_start.strftime("%Y-%m"),
                    "label_start": forecast_times[0].isoformat(),
                    "label_end": forecast_times[-1].isoformat(),
                    "training_sample_count": len(train_indices),
                },
            )
        )
    return tuple(folds)


@dataclass(frozen=True, slots=True)
class RollingOriginFold:
    """One rolling-origin split: origins before ``origin`` train the holdout."""

    window: int
    origin_index: int
    origin: pd.Timestamp
    train_indices: tuple[int, ...]
    metadata: dict[str, Any]


def _positive_int(source: dict[str, Any], field: str, default: int) -> int:
    value = source.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"validation.{field} must be a positive integer")
    return value


def scheduled_origin_indices(
    origins: tuple[pd.Timestamp, ...],
    geometry: TimeGeometry,
    schedule_origin: pd.Timestamp,
    stride_steps: int,
) -> tuple[int, ...]:
    """Select the formal schedule grid, newest first, from complete origins."""
    if isinstance(stride_steps, bool) or not isinstance(stride_steps, int) or stride_steps <= 0:
        raise ValueError("stride_steps must be a positive integer")
    if not origins:
        return ()
    positions = {pd.Timestamp(origin): index for index, origin in enumerate(origins)}
    if len(positions) != len(origins) or tuple(sorted(positions)) != origins:
        raise ValueError("scheduled origins must be ordered and unique")
    current = pd.Timestamp(schedule_origin)
    stride = stride_steps * geometry.offset
    selected = []
    while current >= origins[0]:
        if current in positions:
            selected.append(positions[current])
        previous = current - stride
        if previous >= current:
            raise ValueError("schedule stride must move backwards")
        current = previous
    return tuple(selected)


def rolling_origin_folds(
    origins: tuple[pd.Timestamp, ...],
    geometry: TimeGeometry,
    *,
    history_steps: int | None,
    train_window_steps: int,
    fold_count: int,
    stride_steps: int,
    schedule_origin: pd.Timestamp | None = None,
) -> tuple[RollingOriginFold, ...]:
    """Rolling-origin folds with an explicit supervised-origin-step contract.

    The candidate set is the last ``history_steps`` origins; holdouts are the
    last ``fold_count`` candidates spaced ``stride_steps`` apart, chronologically
    ordered; each holdout trains on the last ``train_window_steps`` candidates
    whose labels end strictly before the holdout label start.
    """
    if history_steps is None:
        history_steps = len(origins)
    history_steps = min(len(origins), history_steps)
    history_start = len(origins) - history_steps
    candidates = (
        tuple(range(len(origins) - 1, history_start - 1, -stride_steps))
        if schedule_origin is None
        else tuple(index for index in scheduled_origin_indices(origins, geometry, schedule_origin, stride_steps)
                   if index >= history_start)
    )[:fold_count]
    if not candidates:
        raise ValueError("no complete supervised origin matches the forecast schedule")
    folds = []
    for window, origin_index in enumerate(reversed(candidates), start=1):
        holdout_origin = origins[origin_index]
        holdout_label_start = geometry.label_start(holdout_origin)
        train_indices = tuple(
            index
            for index in range(history_start, origin_index)
            if is_label_safe(origins[index], geometry.offset, geometry.horizon, holdout_label_start)
        )[-train_window_steps:]
        if not train_indices:
            raise ValueError(
                "canonical rolling backtest requires at least one non-overlapping "
                f"training sample for window={window}"
            )
        training_label_end_max = max(
            geometry.label_end(origins[index]) for index in train_indices
        )
        folds.append(
            RollingOriginFold(
                window=window,
                origin_index=origin_index,
                origin=holdout_origin,
                train_indices=train_indices,
                metadata={
                    "window": window,
                    "origin": holdout_origin.isoformat(),
                    "label_start": holdout_label_start.isoformat(),
                    "label_end": geometry.label_end(holdout_origin).isoformat(),
                    "training_sample_count": len(train_indices),
                    "training_label_end_max": training_label_end_max.isoformat(),
                    "excluded_overlapping_samples": origin_index - len(train_indices),
                },
            )
        )
    return tuple(folds)


def validate_no_overlap(
    origins: tuple[pd.Timestamp, ...],
    train_indices: Iterable[int],
    holdout_origin: pd.Timestamp,
    geometry: TimeGeometry,
) -> dict[str, Any]:
    """Strict `training_label_end_max < validation_label_start` audit.

    Mirrors the pre-extraction `_holdout_training_indices` behaviour and
    metadata contract.
    """
    holdout_label_start = geometry.label_start(holdout_origin)
    train_indices = tuple(train_indices)
    if not train_indices:
        raise ValueError(
            "canonical holdout requires at least one training sample with "
            "label_end < holdout_label_start"
        )
    training_label_end_max = max(
        geometry.label_end(origins[index]) for index in train_indices
    )
    if not all(
        is_label_safe(origins[index], geometry.offset, geometry.horizon, holdout_label_start)
        for index in train_indices
    ):
        raise ValueError("training labels must not overlap holdout labels")
    return {
        "origin": holdout_origin.isoformat(),
        "label_start": holdout_label_start.isoformat(),
        "label_end": geometry.label_end(holdout_origin).isoformat(),
        "training_sample_count": len(train_indices),
        "training_label_end_max": training_label_end_max.isoformat(),
        "excluded_overlapping_samples": len(origins) - 1 - len(train_indices),
    }
