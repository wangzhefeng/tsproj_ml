"""Canonical long-format forecast results and legacy read-only conversion."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

from forecasting_core.tensors import PointForecastTensor
from forecasting_core.artifacts import MarginalForecastDistribution
from model_evaluation.point import EVALUATION_AGGREGATION

# 回测写盘/绘图已迁 model_testing/reporting.py（2026-09-06 R5b）；预测侧复用其共享绘图工具。
from model_testing.reporting import (  # noqa: F401  (预测图复用 + 兼容 re-export)
    _format_time_axis,
    _plot_timeseries,
    _safe_plot_name,
)


_CANONICAL_KEY_COLUMNS = ["series_id", "time", "target"]
_BACKTEST_KEY_COLUMNS = [*_CANONICAL_KEY_COLUMNS, "window"]


# 纯张量->long 转换自本模块迁出至 model_testing/tensor_frames.py（2026-09-06 R5b，
# 供回测评分体共用且不构成包环）；此处 re-export 保持既有消费面零改动。
from model_testing.tensor_frames import (  # noqa: F401,E402
    backtest_tensors_to_long,
    distribution_to_long,
    point_tensor_to_long,
)


def write_forecast_results(
    output_dir: str | Path,
    forecast: PointForecastTensor | MarginalForecastDistribution,
    *,
    extra_columns: Mapping[str, np.ndarray] | None = None,
    history: PointForecastTensor | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frame = (
        point_tensor_to_long(forecast)
        if isinstance(forecast, PointForecastTensor)
        else distribution_to_long(forecast)
    )
    if extra_columns:
        for name, values in extra_columns.items():
            array = np.asarray(values, dtype=float).reshape(-1)
            if len(array) != len(frame):
                raise ValueError(
                    f"extra column {name!r} length {len(array)} != frame rows {len(frame)}"
                )
            frame[name] = array
    destination = output_path / "prediction.csv"
    frame.to_csv(destination, index=False, encoding="utf_8_sig")
    # 预测结果可视化（2026-09-02）：预测曲线 + 前置 5×horizon 历史真实值参照段
    if len(frame) > 0:
        history_frame = (
            _history_long_frame(history) if history is not None else None
        )
        _plot_forecast(frame, output_path, history_frame)
    return destination


def _history_long_frame(history: PointForecastTensor) -> pd.DataFrame:
    """历史张量 → long 帧（actual_value 列），供预测图的历史段绘制。"""
    return point_tensor_to_long(history).rename(
        columns={"predict_value": "actual_value"}
    )


def _plot_forecast_series(
    forecast_frame: pd.DataFrame,
    title: str,
    output_path: Path,
    history_frame: pd.DataFrame | None = None,
    *,
    figsize: tuple = (14, 5),
    dpi: int = 300,
) -> None:
    """正式预测图：预测段 Preds 虚线（+PI 带），可选前置历史段 Trues 实线。

    历史段来自 target_history（as-of origin，与预测段时间轴首尾相接不重叠）；
    Preds 线首点回接历史最后一个真实值，保证两段视觉连续；历史段末端画
    forecast origin 竖线分隔参照区与预测区。
    """
    import matplotlib.pyplot as plt

    forecast_frame = forecast_frame.copy()
    forecast_frame["time"] = pd.to_datetime(forecast_frame["time"])
    forecast_frame = forecast_frame.sort_values("time", kind="stable")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=figsize)
    series_ids = tuple(dict.fromkeys(forecast_frame["series_id"]))
    for series_id in series_ids:
        segment = forecast_frame[forecast_frame["series_id"] == series_id]
        suffix = "" if len(series_ids) == 1 else f" [{series_id}]"
        origin_time = None
        last_history_value = None
        connector_x = segment["time"]
        connector_y = segment["predict_value"].astype(float).values
        if history_frame is not None:
            history_segment = history_frame[
                history_frame["series_id"] == series_id
            ].sort_values("time", kind="stable")
            if not history_segment.empty:
                history_times = pd.to_datetime(history_segment["time"])
                axis.plot(
                    history_times,
                    history_segment["actual_value"].astype(float).values,
                    label=f"Trues{suffix}",
                    lw=1.5,
                )
                # Preds 首点回接历史末个真实值（视觉连续）
                last_history_value = float(
                    history_segment["actual_value"].iloc[-1]
                )
                connector_x = pd.concat(
                    [pd.Series(history_times.iloc[-1]), segment["time"]],
                    ignore_index=True,
                )
                connector_y = np.concatenate(
                    [[last_history_value], segment["predict_value"].astype(float).values]
                )
                origin_time = history_times.iloc[-1]
        axis.plot(
            connector_x,
            connector_y,
            label=f"Preds{suffix}",
            lw=1.5,
            ls="-.",
        )
        quantile_columns = [
            column for column in segment.columns if column.startswith("predict_q")
        ]
        if len(quantile_columns) >= 2:
            # PI 带：有历史时从历史末点起画（与 Preds 回接点对齐），否则只盖预测段
            if origin_time is not None and last_history_value is not None:
                band_x = pd.concat(
                    [pd.Series(origin_time), segment["time"]], ignore_index=True
                )
                band_low = np.concatenate(
                    [
                        [last_history_value],
                        segment[quantile_columns[0]].astype(float).values,
                    ]
                )
                band_high = np.concatenate(
                    [
                        [last_history_value],
                        segment[quantile_columns[-1]].astype(float).values,
                    ]
                )
            else:
                band_x = segment["time"]
                band_low = segment[quantile_columns[0]].astype(float).values
                band_high = segment[quantile_columns[-1]].astype(float).values
            axis.fill_between(
                band_x,
                band_low,
                band_high,
                color="tab:blue",
                alpha=0.15,
                label=f"PI [{quantile_columns[0]},{quantile_columns[-1]}]",
            )
        if origin_time is not None:
            axis.axvline(
                origin_time,
                color="gray",
                lw=1.0,
                ls=":",
                alpha=0.7,
                label="forecast origin",
            )
    axis.set_title(title)
    axis.set_xlabel("Time")
    axis.set_ylabel("Value")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper left", fontsize="small")
    _format_time_axis(axis, forecast_frame["time"])
    figure.autofmt_xdate(rotation=30)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _plot_forecast(
    frame: pd.DataFrame,
    output_dir: Path,
    history_frame: pd.DataFrame | None = None,
) -> None:
    """正式预测可视化：单 target → forecast_prediction.png；多 target →
    prediction_plots/<target>.png。quantile 模式附 predict_q* 区间带；
    提供历史帧时在预测起点前绘制 5×horizon 历史真实值参照段。"""
    targets = tuple(dict.fromkeys(str(target) for target in frame["target"]))
    if len(targets) == 1:
        _plot_forecast_series(
            frame[frame["target"] == targets[0]],
            f"Forecast: {targets[0]}",
            output_dir / "forecast_prediction.png",
            (
                history_frame[history_frame["target"] == targets[0]]
                if history_frame is not None
                else None
            ),
            dpi=300,
        )
        return
    plots_dir = output_dir / "prediction_plots"
    for target in targets:
        _plot_forecast_series(
            frame[frame["target"] == target],
            f"Forecast: {target}",
            plots_dir / f"{_safe_plot_name(target)}.png",
            (
                history_frame[history_frame["target"] == target]
                if history_frame is not None
                else None
            ),
            dpi=300,
        )


class CanonicalResultReader:
    """Read canonical long-schema results."""

    @staticmethod
    def read_prediction(
        path: str | Path,
        *,
        target: str | None = None,
        series_id: object = "__local__",
    ) -> pd.DataFrame:
        frame = pd.read_csv(path)
        if set(_CANONICAL_KEY_COLUMNS + ["predict_value"]).issubset(frame):
            result = frame.copy()
            result["time"] = pd.to_datetime(result["time"])
            return result
        raise ValueError(f"non-canonical prediction file: {path}")

    @staticmethod
    def read_backtest(
        path: str | Path,
        *,
        target: str | None = None,
        series_id: object = "__local__",
    ) -> pd.DataFrame:
        frame = pd.read_csv(path)
        required = {
            "series_id",
            "time",
            "target",
            "actual_value",
            "predict_value",
            "window",
            "plot_valid",
        }
        if required.issubset(frame):
            result = frame.copy()
            result["time"] = pd.to_datetime(result["time"])
            return result
        raise ValueError(f"non-canonical backtest file: {path}")

    @staticmethod
    def read_scores(path: str | Path) -> pd.DataFrame:
        return pd.read_csv(path)


__all__ = [
    "CanonicalResultReader",
    "backtest_tensors_to_long",
    "distribution_to_long",
    "point_tensor_to_long",
    "write_forecast_results",
]
