"""Shared rolling-origin time contract for backtests and OOF folds.

Extracted in E1 from `model_forecasting/runtime.py` (`_label_start`, `_label_end`,
`_holdout_training_indices`, `_rolling_backtest_windows`). The functions here
are deliberately decoupled from any runtime type: they take explicit time
geometry (origin offset / horizon steps) so the ensemble OOF splitter (E3) can
reuse the exact same semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

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
) -> bool:
    """True when a training sample never overlaps the holdout labels."""
    return label_end(origin, offset, horizon) < holdout_label_start


@dataclass(frozen=True, slots=True)
class TimeGeometry:
    """Explicit origin-step geometry shared by backtests and OOF folds."""

    offset: pd.tseries.frequencies.BaseOffset
    horizon: int

    def label_start(self, origin: pd.Timestamp) -> pd.Timestamp:
        return label_start(origin, self.offset)

    def label_end(self, origin: pd.Timestamp) -> pd.Timestamp:
        return label_end(origin, self.offset, self.horizon)


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


def rolling_origin_folds(
    origins: tuple[pd.Timestamp, ...],
    geometry: TimeGeometry,
    *,
    history_length: int | None,
    window_length: int,
    max_windows: int,
    stride: int,
) -> tuple[RollingOriginFold, ...]:
    """Rolling-origin folds with the canonical non-overlap contract.

    Contract (identical to the pre-extraction `_rolling_backtest_windows`):
    the candidate set is the last ``history_length`` origins; holdouts are the
    last ``max_windows`` candidates spaced ``stride`` apart, chronologically
    ordered; each holdout trains on the last ``window_length`` candidates whose
    labels end strictly before the holdout label start.
    """
    if history_length is None:
        history_length = len(origins)
    history_length = min(len(origins), history_length)
    history_start = len(origins) - history_length
    candidates = tuple(
        range(len(origins) - 1, history_start - 1, -stride)
    )[:max_windows]
    folds = []
    for window, origin_index in enumerate(reversed(candidates), start=1):
        holdout_origin = origins[origin_index]
        holdout_label_start = geometry.label_start(holdout_origin)
        train_indices = tuple(
            index
            for index in range(history_start, origin_index)
            if geometry.label_end(origins[index]) < holdout_label_start
        )[-window_length:]
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
    if training_label_end_max >= holdout_label_start:
        raise ValueError("training labels must not overlap holdout labels")
    return {
        "origin": holdout_origin.isoformat(),
        "label_start": holdout_label_start.isoformat(),
        "label_end": geometry.label_end(holdout_origin).isoformat(),
        "training_sample_count": len(train_indices),
        "training_label_end_max": training_label_end_max.isoformat(),
        "excluded_overlapping_samples": len(origins) - 1 - len(train_indices),
    }
