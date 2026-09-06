# -*- coding: utf-8 -*-
"""回测结果写盘与可视化（R5b 自 model_forecasting/results.py 迁出，2026-09-06，方案 v3）。

回测产物（cv_plot_df/test_scores*/windows_results/总图）属于「模型测试」输出，
随 R5 归入 model_testing；预测侧写盘仍在 model_forecasting.results（经 plotting
复用本模块的共享绘图工具）。绘图用未掩码原始值，掩码只用于指标计算。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from model_evaluation.point import EVALUATION_AGGREGATION

from model_testing.tensor_frames import _BACKTEST_KEY_COLUMNS


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
        "aggregation_semantics": dict(EVALUATION_AGGREGATION),
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
