"""Canonical long-format forecast results and legacy read-only conversion."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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


def _point_forecast(value: PointForecastTensor | MarginalForecastDistribution) -> PointForecastTensor:
    if isinstance(value, PointForecastTensor):
        return value
    if isinstance(value, MarginalForecastDistribution):
        return value.point
    raise TypeError("forecast must be PointForecastTensor or MarginalForecastDistribution")


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


def _safe_plot_name(target: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(target)).strip("-.")
    return value or "target"


def _plot_timeseries(
    axis,
    x,
    y_pred,
    y_true=None,
    quantile_frame: pd.DataFrame | None = None,
    series_suffix: str = "",
) -> None:
    """单条曲线组：actual 实线（可缺）+ predict 虚线 + 可选 quantile 区间带。

    线型与旧版 models/ModelTesting.py::test_results_save 一致（Trues 实线 /
    Preds 点划线 / PI 带 alpha=0.15）；绘图用未掩码原始值，保证线条连续
    （掩码只用于指标计算，不参与绘图——旧版同口径）。
    """
    if y_true is not None:
        axis.plot(x, y_true, label=f"Trues{series_suffix}", lw=1.5)
    axis.plot(x, y_pred, label=f"Preds{series_suffix}", lw=1.5, ls="-.")
    if quantile_frame is not None:
        quantile_columns = [
            column
            for column in quantile_frame.columns
            if column.startswith("predict_q")
        ]
        if len(quantile_columns) >= 2:
            axis.fill_between(
                x,
                quantile_frame[quantile_columns[0]].astype(float).values,
                quantile_frame[quantile_columns[-1]].astype(float).values,
                color="tab:blue",
                alpha=0.15,
                label=(
                    f"PI [{quantile_columns[0]},{quantile_columns[-1]}]"
                ),
            )


def _format_time_axis(axis, times: pd.Series) -> None:
    """日期轴自适应刻度（旧版同款：避免高频数据刻度标签重叠成黑块）。"""
    from matplotlib.dates import AutoDateLocator, DateFormatter

    span_hours = (
        (times.max() - times.min()).total_seconds() / 3600.0
        if len(times) >= 2
        else 0.0
    )
    locator = AutoDateLocator()
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(
        DateFormatter("%m-%d %H:%M" if span_hours <= 48 else "%m-%d")
    )


def _plot_series_pair(
    frame: pd.DataFrame,
    title: str,
    output_path: Path,
    *,
    figsize: tuple = (14, 5),
    dpi: int = 150,
) -> None:
    """单 target（多 series 合轴）的一张 actual/predict 对照图。

    输入为该 target 的 long 行（含 series_id/time/actual_value/predict_value，
    可选 predict_q*）；按时间排序后整条绘制（滑窗 stride=horizon 时窗口首尾
    相接，拼接即连续曲线）。时间戳重复（stride < horizon 的重叠配置）RAISE。
    """
    import matplotlib.pyplot as plt

    plot_frame = frame.copy()
    plot_frame["time"] = pd.to_datetime(plot_frame["time"])
    plot_frame = plot_frame.sort_values("time", kind="stable")
    for series_id, group in plot_frame.groupby("series_id", sort=True):
        duplicated = group["time"].duplicated().any()
        if duplicated:
            raise ValueError(
                f"backtest windows overlap in time for series {series_id!r} "
                f"(stride < horizon?): stitched overview plot requires "
                f"stride == horizon; inspect windows_results/ per-window "
                f"plots instead"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=figsize)
    has_actual = "actual_value" in plot_frame.columns
    for series_id, group in plot_frame.groupby("series_id", sort=True):
        suffix = "" if len(plot_frame["series_id"].unique()) == 1 else f" [{series_id}]"
        _plot_timeseries(
            axis,
            group["time"],
            group["predict_value"].astype(float).values,
            y_true=(
                group["actual_value"].astype(float).values
                if has_actual
                else None
            ),
            quantile_frame=group,
            series_suffix=suffix,
        )
    axis.set_title(title)
    axis.set_xlabel("Time")
    axis.set_ylabel("Value")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper left", fontsize="small")
    _format_time_axis(axis, plot_frame["time"])
    figure.autofmt_xdate(rotation=30)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _plot_window(
    frame: pd.DataFrame,
    window: int,
    targets: tuple[str, ...],
    output_dir: Path,
) -> Path:
    """单窗口一张图：多 target 纵向子图，每子图 actual/predict 对照。

    单窗内部时间天然不重叠，无需拼接防护；输出 windows_results/window_<w>.png。
    """
    window_frame = frame[frame["window"] == window].copy()
    window_frame["time"] = pd.to_datetime(window_frame["time"])
    output_path = output_dir / f"window_{int(window):02d}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    n_panels = max(1, len(targets))
    figure, axes = plt.subplots(
        n_panels, 1, figsize=(14, 4 * n_panels), squeeze=False
    )
    for axis, target in zip(axes[:, 0], targets):
        target_frame = window_frame[window_frame["target"] == target]
        for series_id, group in target_frame.groupby("series_id", sort=True):
            group = group.sort_values("time", kind="stable")
            suffix = (
                ""
                if len(target_frame["series_id"].unique()) == 1
                else f" [{series_id}]"
            )
            _plot_timeseries(
                axis,
                group["time"],
                group["actual_value"].astype(float).values,
                group["predict_value"].astype(float).values,
                quantile_frame=group,
                series_suffix=suffix,
            )
        axis.set_title(str(target))
        axis.set_ylabel("Value")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper left", fontsize="small")
        _format_time_axis(axis, target_frame["time"])
    axes[-1, 0].set_xlabel("Time")
    figure.suptitle(f"Window {int(window)}", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


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


def _plot_backtest(frame: pd.DataFrame, output_path: Path, target: str) -> None:
    """兼容壳：单 target 的拼接总图（写入既有的 test_prediction.png /
    target_plots/<target>.png 布局）。实现统一收敛到 _plot_series_pair。"""
    _plot_series_pair(
        frame[frame["target"] == target],
        str(target),
        output_path,
        dpi=150,
    )


def write_backtest_results(
    output_dir: str | Path,
    cv_plot_df: pd.DataFrame,
    test_scores_df: pd.DataFrame,
    *,
    aggregate_weighting: Mapping[str, float],
    metadata: Mapping[str, Any] | None = None,
    probabilistic_scores_df: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    required = {
        "series_id",
        "time",
        "target",
        "actual_value",
        "predict_value",
        "window",
        "plot_valid",
    }
    missing = required - set(cv_plot_df)
    if missing:
        raise ValueError(f"canonical backtest is missing columns: {sorted(missing)}")
    if cv_plot_df.duplicated(_BACKTEST_KEY_COLUMNS).any():
        raise ValueError("canonical backtest result keys must be unique")
    cv_path = output_path / "cv_plot_df.csv"
    scores_path = output_path / "test_scores_df.csv"
    cv_plot_df.to_csv(cv_path, index=False, encoding="utf_8_sig")
    # 分粒度拆分（2026-09-03 方案 B）：每窗汇总（target/aggregate）留在
    # test_scores_df.csv；per-horizon 明细（horizon/aggregate_horizon）挪到
    # test_scores_horizon_df.csv，恢复「一个回测窗口一行统计」的可读形态。
    horizon_scopes = ("horizon", "aggregate_horizon")
    if "scope" in test_scores_df.columns and (
        set(test_scores_df["scope"].unique()) & set(horizon_scopes)
    ):
        horizon_scores_df = test_scores_df[
            test_scores_df["scope"].isin(horizon_scopes)
        ]
        summary_scores_df = test_scores_df[
            ~test_scores_df["scope"].isin(horizon_scopes)
        ]
        horizon_scores_df.to_csv(
            output_path / "test_scores_horizon_df.csv",
            index=False,
            encoding="utf_8_sig",
        )
        summary_scores_df.to_csv(scores_path, index=False, encoding="utf_8_sig")
    else:
        test_scores_df.to_csv(scores_path, index=False, encoding="utf_8_sig")
    # 概率评估（2026-08-30 接线）：quantile 模式的 pinball/central 区间指标，
    # tidy long schema，仅 quantile 模式产出；与点分数同拆分口径（2026-09-03
    # 方案 B）：汇总（target/aggregate）留主文件，per-horizon 拆独立文件。
    if probabilistic_scores_df is not None:
        probabilistic_scores_df.to_csv(
            output_path / "test_scores_probabilistic_df.csv",
            index=False,
            encoding="utf_8_sig",
        )
        if "scope" in probabilistic_scores_df.columns and (
            set(probabilistic_scores_df["scope"].unique()) & set(horizon_scopes)
        ):
            probabilistic_scores_df[
                probabilistic_scores_df["scope"].isin(horizon_scopes)
            ].to_csv(
                output_path / "test_scores_probabilistic_horizon_df.csv",
                index=False,
                encoding="utf_8_sig",
            )
            probabilistic_scores_df[
                ~probabilistic_scores_df["scope"].isin(horizon_scopes)
            ].to_csv(
                output_path / "test_scores_probabilistic_df.csv",
                index=False,
                encoding="utf_8_sig",
            )
    payload = {
        "result_schema_version": 2,
        "aggregate_weighting": {
            str(target): float(weight)
            for target, weight in aggregate_weighting.items()
        },
        **dict(metadata or {}),
    }
    (output_path / "result_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    targets = tuple(dict.fromkeys(str(target) for target in cv_plot_df["target"]))
    # per-window 可视化（2026-09-02）：windows_results/window_<w>.png，
    # 每窗一文件、多 target 纵向子图；与总图并存。
    windows_dir = output_path / "windows_results"
    window_numbers = sorted(
        int(value) for value in cv_plot_df["window"].unique()
    )
    for window in window_numbers:
        _plot_window(cv_plot_df, window, targets, windows_dir)
    if len(targets) == 1:
        _plot_backtest(cv_plot_df, output_path / "test_prediction.png", targets[0])
    else:
        for target in targets:
            _plot_backtest(
                cv_plot_df,
                output_path / "target_plots" / f"{_safe_plot_name(target)}.png",
                target,
            )
    return cv_path, scores_path


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
    "write_backtest_results",
    "write_forecast_results",
]
