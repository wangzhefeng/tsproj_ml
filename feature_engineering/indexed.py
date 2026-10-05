"""规则 source-time 历史设计的因子化执行。

不适用的节点显式返回 None，由已有信息集编译器处理（不降级数据语义）。
lag 依赖目标时刻/原点；rolling 依赖原点；expanding 同时依赖历史下界。
训练标签单独构造，不作为预测特征的可见性授权。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from forecasting_core.design import IndexedDesign
from forecasting_core.specs import AvailabilityPolicy, ColumnRole
from feature_engineering.history_statistics import expanding_statistics, rolling_statistics


def indexed_history_eligible(compiler, call_steps: Sequence[int]) -> bool:
    config = compiler.config
    window = config.validation.get("forecast_window", {})
    if window.get("start") == "next_day" or window.get("gap_steps", 0):
        return False
    if config.problem.training_scope != "local":
        return False
    if compiler.datetime_categorical:
        return False
    transformations = config.features.transformations
    if set(transformations) - {"direct", "advanced", "feature_scaling", "target"}:
        return False
    advanced = transformations.get("advanced", {})
    if set(advanced) - {"rolling", "expanding", "difference"}:
        return False
    for source in config.data.sources:
        if (source.source_type != "file" or source.availability is not AvailabilityPolicy.SOURCE_TIME
                or source.series_id_cols or source.inference_columns
                or any(c.categorical or c.role not in {ColumnRole.TARGET, ColumnRole.OBSERVED_PAST}
                       for c in source.columns)):
            return False
    direct = transformations.get("direct", {})
    aligned = compiler.resolved_strategy.consumes_previous or direct.get("align_to_target") is not False
    # Oracle/provider-dependent training needs the original dependency graph.
    if aligned and any(lag < max(call_steps) for mapping in (
            config.features.target_lags, config.features.observed_past_lags)
            for lags in mapping.values() for lag in lags):
        return False
    return True


def compile_indexed_history(compiler, information_set, origins: pd.DatetimeIndex,
                            call_steps: Sequence[int], *, registry=None):
    """Return compact calls, labels and schema, or None for unsupported layout."""
    if not indexed_history_eligible(compiler, call_steps) or len(origins) == 0:
        return None
    config = compiler.config
    offset = pd.tseries.frequencies.to_offset(config.problem.freq)
    if not origins.is_monotonic_increasing or origins.has_duplicates:
        return None
    frames_by_role = {ColumnRole.TARGET: information_set.target_history,
                      ColumnRole.OBSERVED_PAST: information_set.observed_past}
    columns = {}
    times = None
    for source in config.data.sources:
        for column in source.columns:
            frame = frames_by_role[column.role][source.name]
            index = pd.DatetimeIndex(frame[source.time_col])
            if times is None:
                times = index
            if not index.equals(times) or not index.equals(pd.date_range(index[0], periods=len(index), freq=offset)):
                return None
            if registry is None:
                values = pd.to_numeric(frame[column.name], errors="raise").to_numpy(dtype=float, copy=True)
            else:
                snapshot_times, snapshot_values = registry.numeric_history(source, column.name)
                begin = snapshot_times.get_indexer(index[:1])[0]
                if begin < 0 or not snapshot_times[begin:begin + len(index)].equals(index):
                    raise ValueError("numeric snapshot does not match visible history")
                values = snapshot_values[begin:begin + len(index)]
            if not np.isfinite(values).all():
                raise ValueError(f"history column {column.name!r} must be finite")
            values.flags.writeable = False
            columns[column.name] = values
    positions = times.get_indexer(origins)
    if np.any(positions < 0):
        raise ValueError("supervised origins must belong to the regular history grid")
    start, stop = int(positions[0]), int(positions[-1]) + 1
    rows = positions - start
    sparse = len(positions) != stop - start
    if stop + config.problem.horizon > len(times):
        raise ValueError("supervised labels exceed the available history grid")

    def freeze(values):
        array = np.asarray(values, dtype=float)
        array.flags.writeable = False
        return array

    origin_columns = {}
    advanced = config.features.transformations.get("advanced", {})
    rolling = advanced.get("rolling", {})
    for column in rolling.get("columns", ()):
        series = pd.Series(columns[column], index=times)
        for window in rolling["windows"]:
            values = rolling_statistics(series, window, rolling["stats"])
            for stat, array in values.items():
                origin_columns[f"{column}_rolling_{stat}_{window}"] = freeze(array.to_numpy())
    expanding = advanced.get("expanding", {})
    for column in expanding.get("columns", ()):
        values = expanding_statistics(pd.Series(columns[column], index=times), expanding["stats"])
        for stat, array in values.items():
            origin_columns[f"{column}_expanding_{stat}"] = freeze(array.to_numpy())
    difference = advanced.get("difference", {})
    for column in difference.get("columns", ()):
        for period in difference["periods"]:
            origin_columns[f"{column}_diff_{period}"] = freeze(pd.Series(columns[column]).diff(period).to_numpy())
    datetime = {f"dt_{name}": freeze([compiler._DATETIME_FEATURES[name](ts) for ts in times])
                for name in config.features.datetime_features}
    direct = config.features.transformations.get("direct", {})
    aligned = compiler.resolved_strategy.consumes_previous or direct.get("align_to_target") is not False
    calls = []
    schema = None
    for step in call_steps:
        descriptors = {}
        for mapping in (config.features.target_lags, config.features.observed_past_lags):
            for column, lags in mapping.items():
                for lag in lags:
                    shift = (step if aligned else 0) - lag
                    if shift > 0:
                        raise ValueError("indexed history cannot access future values")
                    descriptors[f"{column}__lag_{lag}"] = (columns[column], shift)
        descriptors.update({name: (array, step) for name, array in datetime.items()})
        horizon_values = {"horizon_step": np.full(len(times), step, dtype=float)}
        if direct:
            compiler._compile_direct_transformations(horizon_values, direct, vectorized=True)
        descriptors.update({name: (freeze(array), 0) for name, array in horizon_values.items() if name != "horizon_step"})
        descriptors.update({name: (array, 0) for name, array in origin_columns.items()})
        if not descriptors:
            return None
        schema = tuple(descriptors)
        design = IndexedDesign(tuple(descriptors.values()), start=start, stop=stop)
        calls.append(design[rows] if sparse else design)
    if len(config.problem.targets) == 1:
        windows = np.lib.stride_tricks.sliding_window_view(columns[config.problem.targets[0]], config.problem.horizon)
        targets = windows[start + 1:stop + 1, :, None]
        if sparse:
            targets = targets[rows]
        return tuple(calls), targets, schema
    targets = np.empty((len(origins), config.problem.horizon, len(config.problem.targets)), dtype=float)
    for k, name in enumerate(config.problem.targets):
        windows = np.lib.stride_tricks.sliding_window_view(columns[name], config.problem.horizon)
        targets[:, :, k] = windows[positions + 1]
    return tuple(calls), targets, schema
