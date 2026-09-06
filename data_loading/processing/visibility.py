"""严格 as-of 版本选择与历史/监督标签访问边界。"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from forecasting_core.specs.data import ColumnRole, DataSourceSpec

from data_loading.processing.validation import resolve_available_at_col, source_primary_key
from data_loading.processing.alignment import select_requested_series, require_history_per_series, order_temporal, role_frame

if TYPE_CHECKING:
    from data_loading.information.information_set import InformationSetRequest


def select_as_of_vintage(
    source: DataSourceSpec,
    frame: pd.DataFrame,
    request: InformationSetRequest,
    *,
    include_target_labels: bool = False,
) -> pd.DataFrame:
    available_at_col = resolve_available_at_col(source)
    if available_at_col is None:
        raise ValueError(f"source {source.name!r} has no available_at semantics")
    eligible_mask = frame[available_at_col] <= request.forecast_origin
    if include_target_labels:
        if source.time_col is None:
            raise ValueError(
                f"target label source {source.name!r} requires time_col"
            )
        eligible_mask |= frame[source.time_col].isin(
            request.forecast_times.tolist()
        )
    eligible = frame.loc[eligible_mask].copy()
    primary_key = source_primary_key(source)
    if eligible.empty:
        return eligible
    max_available = eligible.groupby(primary_key, dropna=False)[available_at_col].transform("max")
    latest = eligible.loc[eligible[available_at_col] == max_available].copy()
    if latest.duplicated(primary_key, keep=False).any():
        raise ValueError(f"source {source.name!r} has duplicate latest as-of vintages")
    return latest


def history_frame(
    source: DataSourceSpec,
    frame: pd.DataFrame,
    request: InformationSetRequest,
    *,
    role: ColumnRole,
) -> pd.DataFrame:
    if source.time_col is None:
        raise ValueError(f"history source {source.name!r} requires time_col at runtime")
    if (
        role is ColumnRole.TARGET
        and request.target_access == "supervised_labels"
    ):
        allowed_time = (frame[source.time_col] <= request.forecast_origin) | frame[source.time_col].isin(
            request.forecast_times
        )
    else:
        allowed_time = frame[source.time_col] <= request.forecast_origin
    selected = frame.loc[allowed_time].copy()
    selected = select_requested_series(source, selected, request, strict_extra=False)
    require_history_per_series(source, selected, request)
    selected = order_temporal(source, selected, request)
    return role_frame(source, selected, role)
