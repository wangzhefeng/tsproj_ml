"""序列身份、请求覆盖、稳定排序与角色投影。"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd

from forecasting_core.specs.data import ColumnRole, DataSourceSpec

from data_loading.processing.validation import resolve_available_at_col

if TYPE_CHECKING:
    from data_loading.information.information_set import InformationSetRequest


def known_future_frame(
    source: DataSourceSpec,
    frame: pd.DataFrame,
    request: InformationSetRequest,
) -> pd.DataFrame:
    if source.time_col is None:
        raise ValueError(f"known_future source {source.name!r} requires time_col at runtime")
    horizon_rows = frame.loc[frame[source.time_col].isin(request.forecast_times)].copy()
    selected = select_requested_series(source, horizon_rows, request, strict_extra=True)
    expected = expected_temporal_keys(source, request)
    actual = actual_temporal_keys(source, selected)
    if actual != expected or len(selected) != len(expected):
        missing = expected - actual
        extra = actual - expected
        raise ValueError(
            f"known_future source {source.name!r} must exactly cover the request; "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    selected = order_temporal(source, selected, request)
    return role_frame(source, selected, ColumnRole.KNOWN_FUTURE)


def static_frame(
    source: DataSourceSpec,
    frame: pd.DataFrame,
    request: InformationSetRequest,
) -> pd.DataFrame:
    if source.series_id_cols and not request.series_ids:
        identities = tuple(dict.fromkeys(frame_identities(source, frame)))
        if len(identities) != 1:
            raise ValueError(
                f"static source {source.name!r} requires one keyed row for a local request"
            )
        selected = frame.copy()
        expected_ids = identities
    else:
        selected = select_requested_series(source, frame, request, strict_extra=False)
        expected_ids = request_identities(source, request)
    actual_ids = frame_identities(source, selected)
    if len(selected) != len(expected_ids) or set(actual_ids) != set(expected_ids):
        raise ValueError(f"static source {source.name!r} requires exactly one row per requested series")
    if request.series_ids:
        selected = order_static(source, selected, request)
    return role_frame(source, selected, ColumnRole.STATIC)


def filter_identity(
    source: DataSourceSpec,
    frame: pd.DataFrame,
    identity: Any,
) -> pd.DataFrame:
    if not source.series_id_cols:
        return frame
    if len(source.series_id_cols) == 1:
        return frame.loc[frame[source.series_id_cols[0]] == identity]
    mask = np.ones(len(frame), dtype=bool)
    for column, value in zip(source.series_id_cols, identity):
        mask &= frame[column].to_numpy() == value
    return frame.loc[mask]


def select_requested_series(
    source: DataSourceSpec,
    frame: pd.DataFrame,
    request: InformationSetRequest,
    *,
    strict_extra: bool,
) -> pd.DataFrame:
    expected_ids = request_identities(source, request)
    if not source.series_id_cols:
        return frame.copy()
    identities = frame_identities(source, frame)
    expected_set = set(expected_ids)
    if strict_extra and any(identity not in expected_set for identity in identities):
        raise ValueError(f"source {source.name!r} contains unrequested series in the requested horizon")
    mask = [identity in expected_set for identity in identities]
    return frame.loc[mask].copy()


def request_identities(
    source: DataSourceSpec,
    request: InformationSetRequest,
) -> tuple[Any, ...]:
    width = len(source.series_id_cols)
    if width == 0:
        if request.series_ids:
            raise ValueError(f"local source {source.name!r} cannot serve a multi-series request")
        return ((),)
    if not request.series_ids:
        raise ValueError(f"series source {source.name!r} requires request.series_ids")
    normalized = []
    for value in request.series_ids:
        if width == 1:
            if isinstance(value, tuple):
                if len(value) != 1:
                    raise ValueError(f"series identity for source {source.name!r} must have width 1")
                normalized.append(value[0])
            else:
                normalized.append(value)
        else:
            if not isinstance(value, tuple) or len(value) != width:
                raise ValueError(
                    f"series identity for source {source.name!r} must have width {width}"
                )
            normalized.append(value)
    return tuple(normalized)


def frame_identities(source: DataSourceSpec, frame: pd.DataFrame) -> list[Any]:
    if not source.series_id_cols:
        return [()] * len(frame)
    if len(source.series_id_cols) == 1:
        return frame[source.series_id_cols[0]].tolist()
    return list(frame.loc[:, list(source.series_id_cols)].itertuples(index=False, name=None))


def require_history_per_series(
    source: DataSourceSpec,
    frame: pd.DataFrame,
    request: InformationSetRequest,
) -> None:
    expected = set(request_identities(source, request))
    actual = set(frame_identities(source, frame))
    missing = expected - actual
    if missing:
        raise ValueError(f"history source {source.name!r} is missing requested series")


def expected_temporal_keys(
    source: DataSourceSpec,
    request: InformationSetRequest,
) -> set[tuple[Any, ...]]:
    identities = request_identities(source, request)
    if not source.series_id_cols:
        return {(time,) for time in request.forecast_times}
    if len(source.series_id_cols) == 1:
        return {(identity, time) for identity in identities for time in request.forecast_times}
    return {
        (*identity, time)
        for identity in identities
        for time in request.forecast_times
    }


def actual_temporal_keys(
    source: DataSourceSpec,
    frame: pd.DataFrame,
) -> set[tuple[Any, ...]]:
    columns = [*source.series_id_cols, source.time_col]
    return set(frame.loc[:, columns].itertuples(index=False, name=None))


def order_temporal(
    source: DataSourceSpec,
    frame: pd.DataFrame,
    request: InformationSetRequest,
) -> pd.DataFrame:
    if not source.series_id_cols:
        return frame.sort_values(source.time_col, kind="stable").reset_index(drop=True)
    order = {identity: index for index, identity in enumerate(request_identities(source, request))}
    ordered = frame.copy()
    ordered["__series_order"] = [order[identity] for identity in frame_identities(source, ordered)]
    ordered = ordered.sort_values(["__series_order", source.time_col], kind="stable")
    return ordered.drop(columns="__series_order").reset_index(drop=True)


def order_static(
    source: DataSourceSpec,
    frame: pd.DataFrame,
    request: InformationSetRequest,
) -> pd.DataFrame:
    order = {identity: index for index, identity in enumerate(request_identities(source, request))}
    ordered = frame.copy()
    ordered["__series_order"] = [order[identity] for identity in frame_identities(source, ordered)]
    return ordered.sort_values("__series_order", kind="stable").drop(columns="__series_order").reset_index(drop=True)


def role_frame(
    source: DataSourceSpec,
    frame: pd.DataFrame,
    role: ColumnRole,
) -> pd.DataFrame:
    columns = list(source.series_id_cols)
    if source.time_col is not None:
        columns.append(source.time_col)
    available_at_col = resolve_available_at_col(source)
    if available_at_col is not None:
        columns.append(available_at_col)
    columns.extend(column.name for column in source.columns if column.role is role)
    return frame.loc[:, list(dict.fromkeys(columns))].reset_index(drop=True)
