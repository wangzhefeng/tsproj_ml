# -*- coding: utf-8 -*-
"""canonical 张量到 long DataFrame 的纯转换（R5b 自 model_forecasting/results.py 迁出）。

只依赖 forecasting_core 合同类型与 numpy/pandas，无 IO、无绘图——
供 model_testing 评分体与 model_forecasting/model_ensemble 结果写盘共用，
不构成 model_testing -> model_forecasting 的包环。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting_core.tensors import PointForecastTensor, require_matching_point_axes
from forecasting_core.artifacts import MarginalForecastDistribution, QuantileGrid

_CANONICAL_KEY_COLUMNS = ["series_id", "time", "target"]
_BACKTEST_KEY_COLUMNS = [*_CANONICAL_KEY_COLUMNS, "window"]


def point_tensor_to_long(tensor: PointForecastTensor) -> pd.DataFrame:
    if not isinstance(tensor, PointForecastTensor):
        raise TypeError("tensor must be a PointForecastTensor")
    rows = []
    for series_index, series_id in enumerate(tensor.series_ids):
        for time_index, timestamp in enumerate(tensor.forecast_times):
            for target_index, target in enumerate(tensor.targets):
                rows.append(
                    {
                        "series_id": series_id,
                        "time": pd.Timestamp(timestamp),
                        "target": target,
                        "predict_value": float(
                            tensor.values[series_index, time_index, target_index]
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.duplicated(_CANONICAL_KEY_COLUMNS).any():
        raise ValueError("canonical point result keys must be unique")
    return frame


def distribution_to_long(distribution: MarginalForecastDistribution) -> pd.DataFrame:
    if not isinstance(distribution, MarginalForecastDistribution):
        raise TypeError("distribution must be a MarginalForecastDistribution")
    frame = point_tensor_to_long(distribution.point)
    grid = QuantileGrid(
        distribution.quantiles.levels,
        point_level=distribution.quantiles.point_level,
    )
    for level_index, level in enumerate(distribution.quantiles.levels):
        frame[grid.column_name(level)] = distribution.quantiles.values[
            ..., level_index
        ].reshape(-1)
    # 部署期 CQR 区间（2026-09-02 接线）：bundle 携带 calibration_state=applied
    # 时由 deployment._attach_bundle_prediction_intervals 写入 metadata，
    # 此处落成 predict_pi<coverage>_lower/upper 列进 prediction.csv。
    for interval in (distribution.metadata.get("prediction_intervals") or {}).values():
        lower_col, upper_col = interval["columns"]
        lower = np.asarray(interval["lower"], dtype=float).reshape(-1)
        upper = np.asarray(interval["upper"], dtype=float).reshape(-1)
        if len(lower) != len(frame) or len(upper) != len(frame):
            raise ValueError(
                f"prediction interval {lower_col!r}/{upper_col!r} length "
                f"({len(lower)}/{len(upper)}) != frame rows {len(frame)}"
            )
        frame[lower_col] = lower
        frame[upper_col] = upper
    return frame


def _point_forecast(
    value: PointForecastTensor | MarginalForecastDistribution,
) -> PointForecastTensor:
    if isinstance(value, PointForecastTensor):
        return value
    if isinstance(value, MarginalForecastDistribution):
        return value.point
    raise TypeError(
        "forecast must be PointForecastTensor or MarginalForecastDistribution"
    )


def backtest_tensors_to_long(
    actual: PointForecastTensor,
    prediction: PointForecastTensor | MarginalForecastDistribution,
    *,
    window: int,
    plot_valid: np.ndarray | None = None,
) -> pd.DataFrame:
    point = _point_forecast(prediction)
    require_matching_point_axes(actual, point)
    actual_frame = point_tensor_to_long(actual).rename(
        columns={"predict_value": "actual_value"}
    )
    prediction_frame = (
        point_tensor_to_long(prediction)
        if isinstance(prediction, PointForecastTensor)
        else distribution_to_long(prediction)
    )
    frame = actual_frame.merge(
        prediction_frame,
        on=_CANONICAL_KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    frame["window"] = int(window)
    if plot_valid is None:
        valid = np.isfinite(frame["actual_value"].to_numpy(dtype=float)) & np.isfinite(
            frame["predict_value"].to_numpy(dtype=float)
        )
    else:
        valid = np.asarray(plot_valid, dtype=bool).reshape(-1)
        if len(valid) != len(frame):
            raise ValueError("plot_valid must match canonical long row count")
    frame["plot_valid"] = valid
    if frame.duplicated(_BACKTEST_KEY_COLUMNS).any():
        raise ValueError("canonical backtest result keys must be unique")
    quantile_columns = [column for column in frame if column.startswith("predict_q")]
    return frame[
        [
            "series_id",
            "time",
            "target",
            "actual_value",
            "predict_value",
            *quantile_columns,
            "window",
            "plot_valid",
        ]
    ]
