# -*- coding: utf-8 -*-
"""Global panel 的主键、切窗与逐序列执行契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Sequence

import pandas as pd


IncompleteSeriesPolicy = Literal["raise", "drop"]


@dataclass(frozen=True)
class PanelSeriesSlice:
    series_id: Any
    history: pd.DataFrame
    future: pd.DataFrame


def ordered_series_ids(frame: pd.DataFrame, series_id_col: str) -> tuple[Any, ...]:
    if series_id_col not in frame.columns:
        raise ValueError(f"Panel frame missing series ID column '{series_id_col}'.")
    if frame[series_id_col].isna().any():
        raise ValueError(f"Panel series ID column '{series_id_col}' contains missing values.")
    return tuple(pd.unique(frame[series_id_col]))


def validate_panel_keys(
    frame: pd.DataFrame,
    series_id_col: str,
    *,
    label: str,
    expected_steps: int | None = None,
) -> tuple[Any, ...]:
    required = {series_id_col, "time"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} panel frame missing columns: {missing}.")
    if frame[list(required)].isna().any(axis=None):
        raise ValueError(f"{label} panel keys contain missing values.")
    duplicates = frame.duplicated([series_id_col, "time"], keep=False)
    if duplicates.any():
        duplicate_keys = (
            frame.loc[duplicates, [series_id_col, "time"]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(f"{label} panel keys are not unique: {duplicate_keys}.")
    ids = ordered_series_ids(frame, series_id_col)
    if expected_steps is not None:
        counts = frame.groupby(series_id_col, sort=False, observed=True).size()
        invalid = counts[counts != int(expected_steps)]
        if not invalid.empty:
            raise ValueError(
                f"{label} panel requires exactly {expected_steps} rows per series; "
                f"got {invalid.to_dict()}."
            )
    return ids


def materialize_panel_history(
    source: pd.DataFrame,
    times: Sequence[Any],
    *,
    series_id_col: str,
    source_time_col: str,
    target_col: str,
    numeric_columns: Iterable[str] = (),
    categorical_columns: Iterable[str] = (),
    incomplete_policy: IncompleteSeriesPolicy = "raise",
) -> pd.DataFrame:
    """按输入 series 顺序构造完整 `(series_id, time)` 历史面板。"""
    if incomplete_policy not in {"raise", "drop"}:
        raise ValueError(f"Unsupported panel incomplete policy '{incomplete_policy}'.")
    required = {series_id_col, source_time_col, target_col}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"Panel source missing columns: {missing}.")
    source = source.copy()
    source[source_time_col] = pd.to_datetime(source[source_time_col])
    ids = ordered_series_ids(source, series_id_col)
    duplicate = source.duplicated([series_id_col, source_time_col], keep=False)
    if duplicate.any():
        raise ValueError("Panel source contains duplicate (series_id, time) keys.")

    frames: list[pd.DataFrame] = []
    time_index = pd.DatetimeIndex(pd.to_datetime(times))
    keep_numeric = [
        column
        for column in numeric_columns
        if column in source.columns and column not in required
    ]
    keep_categorical = [
        column
        for column in categorical_columns
        if column in source.columns and column not in required
    ]
    for series_id in ids:
        series = source[source[series_id_col] == series_id].set_index(source_time_col)
        aligned = series.reindex(time_index)
        aligned.index.name = "time"
        aligned[series_id_col] = series_id
        aligned["y"] = pd.to_numeric(aligned[target_col], errors="coerce")
        for column in keep_numeric:
            aligned[column] = pd.to_numeric(aligned[column], errors="coerce")
        for column in keep_categorical:
            aligned[column] = aligned[column].astype("category")
        columns = ["time", series_id_col, "y", *keep_numeric, *keep_categorical]
        aligned = aligned.reset_index()[columns]
        if aligned["y"].isna().any():
            if incomplete_policy == "drop":
                continue
            missing_count = int(aligned["y"].isna().sum())
            raise ValueError(
                f"Panel series '{series_id}' is incomplete in the configured history "
                f"window: {missing_count} target rows are missing."
            )
        frames.append(aligned)
    if not frames:
        raise ValueError("Panel history has no complete series.")
    result = pd.concat(frames, ignore_index=True)
    validate_panel_keys(result, series_id_col, label="History")
    return result


def materialize_panel_future(
    series_ids: Sequence[Any],
    times: Sequence[Any],
    *,
    series_id_col: str,
) -> pd.DataFrame:
    frames = [
        pd.DataFrame(
            {
                series_id_col: series_id,
                "time": pd.DatetimeIndex(pd.to_datetime(times)),
            }
        )
        for series_id in series_ids
    ]
    if not frames:
        raise ValueError("Panel future requires at least one series ID.")
    result = pd.concat(frames, ignore_index=True)
    validate_panel_keys(
        result,
        series_id_col,
        label="Future",
        expected_steps=len(times),
    )
    return result


def split_panel_window(
    frame: pd.DataFrame,
    *,
    series_id_col: str,
    window: int,
    horizon: int,
    window_len: int,
    incomplete_policy: IncompleteSeriesPolicy = "raise",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对每个 series 独立应用相同的相对滑窗位置。"""
    if incomplete_policy not in {"raise", "drop"}:
        raise ValueError(f"Unsupported panel incomplete policy '{incomplete_policy}'.")
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for series_id in ordered_series_ids(frame, series_id_col):
        series = (
            frame[frame[series_id_col] == series_id]
            .sort_values("time")
            .reset_index(drop=True)
        )
        test_end = len(series) - horizon * (int(window) - 1)
        test_start = test_end - int(horizon)
        train_end = test_start
        train_start = max(0, train_end - (int(window_len) - int(horizon)))
        valid = train_start < train_end and test_start >= 0 and test_end <= len(series)
        if not valid or test_end - test_start != int(horizon):
            if incomplete_policy == "drop":
                continue
            raise ValueError(
                f"Panel series '{series_id}' cannot satisfy window={window}, "
                f"horizon={horizon}, window_len={window_len}."
            )
        train_parts.append(series.iloc[train_start:train_end].copy())
        test_parts.append(series.iloc[test_start:test_end].copy())
    if not train_parts:
        raise ValueError("Panel window has no complete series.")
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    validate_panel_keys(train, series_id_col, label="Panel train")
    validate_panel_keys(test, series_id_col, label="Panel test", expected_steps=horizon)
    return train, test


def iter_panel_slices(
    history: pd.DataFrame,
    future: pd.DataFrame,
    *,
    series_id_col: str,
    horizon: int,
) -> tuple[PanelSeriesSlice, ...]:
    history_ids = validate_panel_keys(history, series_id_col, label="History")
    future_ids = validate_panel_keys(
        future,
        series_id_col,
        label="Future",
        expected_steps=horizon,
    )
    if future_ids != history_ids:
        raise ValueError(
            "Panel history/future series order mismatch: "
            f"history={history_ids}, future={future_ids}."
        )
    return tuple(
        PanelSeriesSlice(
            series_id=series_id,
            history=(
                history[history[series_id_col] == series_id]
                .sort_values("time")
                .reset_index(drop=True)
            ),
            future=(
                future[future[series_id_col] == series_id]
                .sort_values("time")
                .reset_index(drop=True)
            ),
        )
        for series_id in history_ids
    )


def execute_panel(
    history: pd.DataFrame,
    future: pd.DataFrame,
    *,
    series_id_col: str,
    horizon: int,
    execute_one: Callable[[PanelSeriesSlice], pd.DataFrame],
) -> pd.DataFrame:
    """逐 series 隔离执行；任一失败即整批失败。"""
    outputs: list[pd.DataFrame] = []
    for series_slice in iter_panel_slices(
        history,
        future,
        series_id_col=series_id_col,
        horizon=horizon,
    ):
        output = execute_one(series_slice).copy()
        if len(output) != int(horizon):
            raise ValueError(
                f"Panel series '{series_slice.series_id}' forecast length mismatch: "
                f"expected {horizon}, got {len(output)}."
            )
        if series_id_col not in output.columns:
            output[series_id_col] = series_slice.series_id
        if not (output[series_id_col] == series_slice.series_id).all():
            raise ValueError(
                f"Panel series '{series_slice.series_id}' output identity mismatch."
            )
        outputs.append(output)
    result = pd.concat(outputs, ignore_index=True)
    validate_panel_keys(
        result,
        series_id_col,
        label="Forecast output",
        expected_steps=horizon,
    )
    return result
