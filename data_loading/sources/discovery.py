"""目标 history 的序列/时间覆盖事实；不发布模型输入值或决定训练政策。"""
from dataclasses import dataclass
from typing import Any

import pandas as pd

from data_loading.processing.alignment import filter_identity, frame_identities
from data_loading.sources.source_io import SourceFrames
from data_loading.processing.validation import validate_frame
from forecasting_core.specs.data import ColumnRole, DataSpec


@dataclass(frozen=True, slots=True)
class TargetHistoryCoverage:
    source_name: str
    series_ids: tuple[Any, ...]
    times: pd.DatetimeIndex
    times_by_series: dict[Any, pd.DatetimeIndex]


def target_history_coverage(data_spec: DataSpec, frames: SourceFrames) -> tuple[TargetHistoryCoverage, ...]:
    """报告完整 history 资产覆盖，不受预测原点截断；不含任何 target 值。"""
    coverage = []
    for source in data_spec.sources:
        if not any(column.role is ColumnRole.TARGET for column in source.columns):
            continue
        if source.source_type != "file" or source.history_path is None:
            raise ValueError(
                "global generated target sources require explicit "
                "validation.training_scope.series_order"
            )
        frame = validate_frame(source, frames.read_path(source.history_path), generated=False, path_version="history")
        identities = tuple(dict.fromkeys(frame_identities(source, frame)))
        times = pd.DatetimeIndex(pd.to_datetime(frame[source.time_col]).unique()).sort_values()
        times_by_series = {
            identity: pd.DatetimeIndex(pd.to_datetime(filter_identity(source, frame, identity)[source.time_col])).sort_values()
            for identity in identities
        }
        coverage.append(TargetHistoryCoverage(source.name, identities, times, times_by_series))
    if not coverage:
        raise ValueError("canonical runtime requires target history")
    return tuple(coverage)


def latest_target_time(data_spec: DataSpec, frames: SourceFrames) -> pd.Timestamp:
    latest: list[pd.Timestamp] = []
    for source in data_spec.sources:
        if not any(column.role is ColumnRole.TARGET for column in source.columns):
            continue
        if source.source_type != "file" or source.history_path is None:
            raise ValueError("target history discovery requires file history_path")
        frame = validate_frame(
            source,
            frames.read_path(source.history_path),
            generated=False,
            path_version="history",
        )
        if source.time_col is None or frame.empty:
            raise ValueError(f"target source {source.name!r} has no timestamps")
        latest.append(pd.Timestamp(frame[source.time_col].max()))
    if not latest:
        raise ValueError("DataSpec has no target source")
    if len(set(latest)) != 1:
        raise ValueError("target sources must share one latest timestamp")
    return latest[0]
