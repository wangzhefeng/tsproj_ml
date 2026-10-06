"""回测公开原语（2026-08-29 架构收敛 P3/D3：自 pipeline.runner 公开化迁入，实现逐字保真）。

- `seasonal_naive_tensor`：seasonal-naive 基线张量（滞后阶数可配，默认一个自然日）；
- `actual_tensor`：holdout 实际值张量；
- `positive_validation_int`：validation 段正整数解析（回测原语的共享校验）。

`pipeline.runner` 保留同名私有别名转发，行为零变化；
评估掩码 `build_eval_mask` 已迁入 `model_evaluation/mask.py`（2026-08-30 evaluation 模块化）；
`resolve_origin` 已迁入 `forecasting_core/temporal/origin.py`（2026-09-27：部署路径通用原语，非回测专属）。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthBegin, MonthEnd

from forecasting_core.tensors.point import PointForecastTensor


def positive_validation_int(
    validation: Mapping[str, Any],
    field: str,
    default: int,
) -> int:
    """validation 段正整数字段解析（非法直接 RAISE，不静默钳位）。"""
    value = validation.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"validation.{field} must be a positive integer")
    return value


def resolve_seasonal_naive_lag(
    builder,
    forecast_times: pd.DatetimeIndex,
    origin: pd.Timestamp,
) -> int:
    """seasonal-naive 滞后阶数解析（2026-10-05 自 seasonal_naive_tensor 抽取）。

    配置 ``validation.seasonal_naive_lag`` 优先；未配置时默认一个自然日步数
    （且不小于 horizon；forecast_window 场景按 lead 向上取整到日步数倍数）。
    回测评分（MASE/RMSSE 缩放）与 naive 基线共用本函数，保证同一 lag 口径。
    """
    validation = builder.config.validation
    configured = validation.get("seasonal_naive_lag")
    if configured is None:
        if isinstance(builder.offset, (MonthBegin, MonthEnd)):
            configured = 1
        else:
            step = pd.Timedelta(builder.offset)
            one_day_steps = max(
                1,
                int(round(pd.Timedelta(days=1) / step)),
            )
            configured = max(
                one_day_steps,
                int(builder.config.problem.horizon),
            )
            if builder.config.validation.get("forecast_window") is not None:
                lead = int((forecast_times[-1] - origin) / step)
                configured = max(configured, ((lead + one_day_steps - 1) // one_day_steps) * one_day_steps)
    return positive_validation_int(
        {"seasonal_naive_lag": configured},
        "seasonal_naive_lag",
        1,
    )


def seasonal_naive_tensor(
    builder,
    origin: pd.Timestamp,
    forecast_times: pd.DatetimeIndex,
    *,
    history: PointForecastTensor | None = None,
) -> PointForecastTensor:
    """seasonal-naive 基线：历史 target 按滞后阶数回看（默认一个自然日步数）。

    ``history``（2026-10-05）：调用方已实体化的 as-of 目标历史可传入复用，
    避免同一折内重复 materialize（回测评分接缝 MASE 缩放与 naive 共用一次取数）。
    """
    lag = resolve_seasonal_naive_lag(builder, forecast_times, origin)
    if history is None:
        history = builder.target_history(origin)
    naive_times = pd.DatetimeIndex(
        [pd.Timestamp(value) - lag * builder.offset for value in forecast_times]
    )
    positions = history.forecast_times.get_indexer(naive_times)
    if np.any(positions < 0):
        raise ValueError(
            "seasonal naive requires complete target history at the configured lag"
        )
    return PointForecastTensor(
        values=history.values[:, positions, :],
        series_ids=history.series_ids,
        forecast_times=forecast_times,
        targets=history.targets,
    )


def actual_tensor(
    config,
    values: np.ndarray,
    forecast_times: pd.DatetimeIndex,
    series_ids: tuple,
) -> PointForecastTensor:
    """holdout 实际值整理为 canonical `(N,H,K)` 张量。"""
    return PointForecastTensor(
        values=values.reshape(
            len(series_ids),
            config.problem.horizon,
            len(config.problem.targets),
        ),
        series_ids=series_ids,
        forecast_times=forecast_times,
        targets=config.problem.targets,
    )


__all__ = [
    "actual_tensor",
    "positive_validation_int",
    "resolve_seasonal_naive_lag",
    "seasonal_naive_tensor",
]
