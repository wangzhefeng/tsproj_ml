# -*- coding: utf-8 -*-
"""外生数据的角色切分、as-of选择和时间覆盖契约。"""
from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd


def _normalize_frame(df: pd.DataFrame, ts_col: str, label: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError(f"{label} is empty.")
    if not ts_col or ts_col not in df.columns:
        raise ValueError(f"{label} missing timestamp column '{ts_col}'.")
    frame = df.copy()
    frame[ts_col] = pd.to_datetime(frame[ts_col], errors="coerce")
    if frame[ts_col].isna().any():
        raise ValueError(f"{label} contains invalid timestamp(s).")
    return frame.sort_values(ts_col).reset_index(drop=True)


def validate_exact_coverage(
    df: pd.DataFrame,
    expected_times: Iterable,
    ts_col: str,
    label: str,
) -> None:
    """要求数据逐点、唯一地覆盖期望时间轴。"""
    frame = _normalize_frame(df, ts_col, label)
    duplicates = frame[ts_col].duplicated(keep=False)
    if duplicates.any():
        preview = ", ".join(str(ts) for ts in frame.loc[duplicates, ts_col].head(10))
        raise ValueError(f"{label} contains duplicate target timestamp(s): {preview}")

    expected = pd.DatetimeIndex(pd.to_datetime(list(expected_times)))
    actual = pd.DatetimeIndex(frame[ts_col])
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    if len(missing):
        preview = ", ".join(str(ts) for ts in missing[:10])
        raise ValueError(f"{label} missing {len(missing)} target timestamp(s): {preview}")
    if len(extra):
        preview = ", ".join(str(ts) for ts in extra[:10])
        raise ValueError(f"{label} contains {len(extra)} unexpected timestamp(s): {preview}")


def validate_daily_coverage(
    df: pd.DataFrame,
    expected_times: Iterable,
    ts_col: str,
    value_columns: Iterable[str],
    label: str,
) -> None:
    """验证日历表逐日唯一覆盖，类别值不得缺失。"""
    frame = _normalize_frame(df, ts_col, label)
    frame[ts_col] = frame[ts_col].dt.normalize()
    if frame[ts_col].duplicated().any():
        raise ValueError(f"{label} contains duplicate calendar day(s).")
    expected_days = pd.DatetimeIndex(pd.to_datetime(list(expected_times))).normalize().unique()
    available_days = pd.DatetimeIndex(frame[ts_col])
    missing = expected_days.difference(available_days)
    if len(missing):
        preview = ", ".join(str(ts.date()) for ts in missing[:10])
        raise ValueError(f"{label} missing {len(missing)} calendar day(s): {preview}")
    columns = list(value_columns)
    absent = [column for column in columns if column not in frame.columns]
    if absent:
        raise ValueError(f"{label} missing value column(s): {absent}")
    required = frame[frame[ts_col].isin(expected_days)]
    if required[columns].isna().to_numpy().any():
        raise ValueError(f"{label} contains missing calendar value(s).")


def split_role_frames(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    ts_col: str,
    forecast_start,
    label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """严格分离history/future角色，拒绝重叠和边界缺口。"""
    history = _normalize_frame(history_df, ts_col, f"{label} history")
    future = _normalize_frame(future_df, ts_col, f"{label} future")
    if history[ts_col].duplicated().any():
        raise ValueError(f"{label} history contains duplicate timestamp(s).")
    if future[ts_col].duplicated().any():
        raise ValueError(f"{label} future contains duplicate timestamp(s).")

    start = pd.Timestamp(forecast_start)
    overlap = pd.DatetimeIndex(history[ts_col]).intersection(pd.DatetimeIndex(future[ts_col]))
    if len(overlap):
        preview = ", ".join(str(ts) for ts in overlap[:10])
        raise ValueError(f"{label} history/future overlap at {preview}")
    if history[ts_col].max() >= start:
        raise ValueError(
            f"{label} history must end before forecast start {start}; "
            f"got {history[ts_col].max()}."
        )
    if future[ts_col].min() != start:
        raise ValueError(
            f"{label} future must start exactly at forecast start {start}; "
            f"got {future[ts_col].min()}."
        )
    return history, future


def select_asof_rows(
    df: pd.DataFrame,
    expected_times: Iterable,
    forecast_origin,
    ts_col: str,
    available_at_col: str,
    label: str,
) -> pd.DataFrame:
    """按预测原点选择每个目标时刻当时可得的最新版本。"""
    frame = _normalize_frame(df, ts_col, label)
    if available_at_col not in frame.columns:
        raise ValueError(f"{label} missing available-at column '{available_at_col}'.")
    frame[available_at_col] = pd.to_datetime(frame[available_at_col], errors="coerce")
    if frame[available_at_col].isna().any():
        raise ValueError(f"{label} contains invalid available-at timestamp(s).")

    expected = pd.DatetimeIndex(pd.to_datetime(list(expected_times)))
    origin = pd.Timestamp(forecast_origin)
    candidates = frame[
        frame[ts_col].isin(expected)
        & (frame[available_at_col] <= origin)
    ].copy()
    if candidates.empty:
        raise ValueError(f"{label} has no rows available at forecast origin {origin}.")

    selected = (
        candidates.sort_values([ts_col, available_at_col])
        .drop_duplicates(subset=ts_col, keep="last")
        .set_index(ts_col)
        .reindex(expected)
        .reset_index(names=ts_col)
    )
    missing_mask = selected.drop(columns=[ts_col]).isna().all(axis=1)
    if missing_mask.any():
        missing = selected.loc[missing_mask, ts_col].tolist()
        preview = ", ".join(str(ts) for ts in missing[:10])
        raise ValueError(
            f"{label} missing {len(missing)} target timestamp(s) available at "
            f"forecast origin {origin}: {preview}"
        )
    validate_exact_coverage(selected, expected, ts_col, label)
    return selected
