"""回测窗口构造：fixed-step rolling 折与显式原始历史（train_history_steps）窗口。

自 pipeline.supervised_design 下沉（2026-09-27 边界审计）：窗口几何归属本包；
折载体直接复用 geometry.RollingOriginFold，不再维护同构的 _BacktestWindow。
raw 历史路径所需的 minimum_history 由调用方显式传入（其定义涉及特征/策略
设计知识，留在 pipeline.supervised_design，本包不 import pipeline）。
"""

from __future__ import annotations

import pandas as pd

from data_loading import SourceRegistry
from forecasting_core.specs import (
    ExpandingWindowBacktestSpec,
    FixedStepBacktestSpec,
    ForecastConfigSpec,
    SlidingWindowBacktestSpec,
)
from model_testing.contracts import geometry as backtest_geometry
from model_testing.contracts.geometry import RollingOriginFold

RollingBacktestSpec = (
    FixedStepBacktestSpec | SlidingWindowBacktestSpec | ExpandingWindowBacktestSpec
)


def rolling_backtest_windows(
    supervised_origins: tuple[pd.Timestamp, ...],
    *,
    offset: pd.tseries.frequencies.BaseOffset,
    horizon: int,
    backtest: RollingBacktestSpec,
    schedule_origin: pd.Timestamp | None = None,
) -> tuple[RollingOriginFold, ...]:
    """rolling 系回测折（fixed/sliding/expanding）的 spec 校验包装。

    sliding 要求 stride_steps < horizon（重叠语义），否则与 fixed_steps
    无差异，直接 RAISE 指引换模式；expanding 无 train_window_steps，
    训练集取全部合格候选。
    """
    if not isinstance(
        backtest,
        (FixedStepBacktestSpec, SlidingWindowBacktestSpec, ExpandingWindowBacktestSpec),
    ):
        raise TypeError("rolling backtest requires a rolling-mode backtest spec")
    if isinstance(backtest, SlidingWindowBacktestSpec) and backtest.stride_steps >= horizon:
        raise ValueError(
            "sliding_window requires stride_steps < horizon; "
            "use horizon_mode=fixed_steps for non-overlapping folds"
        )
    geometry = backtest_geometry.TimeGeometry(offset=offset, horizon=horizon)
    train_window_steps = (
        None
        if isinstance(backtest, ExpandingWindowBacktestSpec)
        else backtest.train_window_steps
    )
    return backtest_geometry.rolling_origin_folds(
        supervised_origins,
        geometry,
        history_steps=backtest.history_steps,
        train_window_steps=train_window_steps,
        fold_count=backtest.fold_count,
        stride_steps=backtest.stride_steps,
        schedule_origin=schedule_origin,
    )


def raw_history_backtest_windows(
    *,
    config: ForecastConfigSpec,
    registry: SourceRegistry,
    offset: pd.tseries.frequencies.BaseOffset,
    origin: pd.Timestamp,
    minimum_history: int,
) -> tuple[RollingOriginFold, ...]:
    """只按时间覆盖调度；返回各折有界 runner 内的局部训练索引。

    与 rolling_backtest_windows 的差异：train_indices 为折内有界上下文的
    局部索引（0..train_window_steps-1），metadata 附 raw_history_start/end
    与 train_history_steps。
    """
    spec = config.validation.backtest
    if not isinstance(spec, FixedStepBacktestSpec) or spec.train_history_steps is None:
        raise ValueError("raw history windows require train_history_steps")
    coverage = registry.target_history_coverage()
    times = coverage[0].times
    if any(not item.times.equals(times) for item in coverage[1:]):
        raise ValueError("target sources must share the same history grid")
    times = times[times <= origin]
    if times.empty or times[-1] != origin or not times.equals(pd.date_range(times[0], origin, freq=config.problem.freq)):
        raise ValueError("raw history backtest requires a complete regular history grid")
    available = tuple(times[minimum_history - 1:len(times) - config.problem.horizon])[-spec.history_steps:]
    windows = rolling_backtest_windows(
        available,
        offset=offset,
        horizon=config.problem.horizon,
        backtest=spec,
        schedule_origin=origin if config.validation.get("schedule_mode") == "intraday" else None,
    )
    if len(windows) != spec.fold_count:
        raise ValueError("train_history_steps cannot provide requested fold_count")
    result = []
    for window in windows:
        start = window.origin - (spec.train_history_steps - 1) * offset
        if start < times[0] or len(window.train_indices) != spec.train_window_steps:
            raise ValueError("train_history_steps cannot provide the complete requested fold history")
        result.append(RollingOriginFold(
            window=window.window, origin_index=window.origin_index, origin=window.origin,
            train_indices=tuple(range(spec.train_window_steps)),
            metadata={**window.metadata, "raw_history_start": start.isoformat(),
                      "raw_history_end": window.origin.isoformat(),
                      "train_history_steps": spec.train_history_steps},
        ))
    return tuple(result)


__all__ = ["raw_history_backtest_windows", "rolling_backtest_windows"]
