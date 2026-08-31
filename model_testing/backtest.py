# -*- coding: utf-8 -*-
"""回测公开原语（2026-08-29 架构收敛 P3/D3：自 model_forecasting.runtime 公开化迁入，实现逐字保真）。

- `seasonal_naive_tensor`：seasonal-naive 基线张量（滞后阶数可配，默认一个自然日）；
- `actual_tensor`：holdout 实际值张量；
- `resolve_origin`：预测原点解析（ensemble runtime 经此公开 API 使用，修复 E-B 私有导入）；
- `positive_validation_int`：validation 段正整数解析（回测原语的共享校验）。

`model_forecasting.runtime` 保留同名私有别名转发，行为零变化；
评估掩码 `build_eval_mask` 已迁入 `model_evaluation/mask.py`（2026-08-30 evaluation 模块化）。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthBegin, MonthEnd

from forecasting_core.tensors import PointForecastTensor


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


def seasonal_naive_tensor(
    builder,
    origin: pd.Timestamp,
    forecast_times: pd.DatetimeIndex,
) -> PointForecastTensor:
    """seasonal-naive 基线：历史 target 按滞后阶数回看（默认一个自然日步数）。"""
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
    lag = positive_validation_int(
        {"seasonal_naive_lag": configured},
        "seasonal_naive_lag",
        1,
    )
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


def resolve_origin(registry, raw_origin) -> pd.Timestamp:
    """预测原点解析：None = 数据最后已知时刻，否则严格 Timestamp。"""
    if raw_origin is None:
        return registry.latest_target_time()
    return pd.Timestamp(raw_origin)

__all__ = [
    "actual_tensor",
    "positive_validation_int",
    "resolve_origin",
    "seasonal_naive_tensor",
]
