# -*- coding: utf-8 -*-

# ***************************************************
# * File        : outlier_process.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-11
# * Version     : 1.0.021111
# * Description : 原始负荷数据异常标记、清洗与可视化
# * Link        : link
# * Requirement : numpy, pandas, scipy, matplotlib
# ***************************************************

# python libraries
import argparse
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


TIME_COL = "count_data_time"
TARGET_COL = "h_total_use"
ANOMALY_COL = "是否异常"
ANOMALY_TYPE_COL = "异常类型"
ANOMALY_SCORE_COL = "异常分数"

LOCAL_BASELINE_WINDOWS = [25, 145]
ABS_LOW_THRESHOLD = 1000.0
LOCAL_ROBUST_Z_THRESHOLD = 3.0
PERIODIC_ROBUST_Z_THRESHOLD = 2.8
ROBUST_SCALE = 1.4826
EPSILON = 1e-9
MAX_SHORT_RUN_POINTS = 12
LONG_RUN_SCORE_MULTIPLIER = 2.5

MARKED_OUTPUT_NAME = "df_power_outlier_detection.csv"
CLEANED_OUTPUT_NAME = "df_power_remove_outlier.csv"
PLOT_OUTPUT_NAME = "df_power_anomalies.png"


def remove_outliers(df: pd.DataFrame, column: str, threshold: int=3):
    """
    移除异常值（Z-score方法）
    """
    z_scores = np.abs(stats.zscore(df[column]))

    return df[z_scores < threshold]


def robust_scale(values: pd.Series) -> pd.Series:
    scale = values * ROBUST_SCALE
    return scale.mask(scale <= EPSILON)


def finite_max(*series_list: pd.Series) -> pd.Series:
    frame = pd.concat(series_list, axis=1)
    return frame.replace([np.inf, -np.inf], np.nan).max(axis=1).fillna(0.0)


def add_anomaly_type(
    types: pd.Series,
    mask: pd.Series,
    label: str,
) -> pd.Series:
    mask = mask.fillna(False).astype(bool)
    result = types.copy()
    result.loc[mask & (result == "")] = label
    result.loc[mask & (result != "") & ~result.str.contains(label, regex=False)] += f";{label}"
    return result


def keep_short_runs(mask: pd.Series, score: pd.Series, threshold: float) -> pd.Series:
    values = mask.fillna(False).to_numpy(dtype=bool)
    scores = score.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    kept = np.zeros(len(values), dtype=bool)
    start = None
    for pos, is_true in enumerate(values):
        if is_true and start is None:
            start = pos
        if start is not None and (not is_true or pos == len(values) - 1):
            end = pos if is_true else pos - 1
            run = np.arange(start, end + 1)
            if len(run) <= MAX_SHORT_RUN_POINTS:
                kept[run] = True
            else:
                kept[run] = scores[run] >= threshold * LONG_RUN_SCORE_MULTIPLIER
            start = None
    return pd.Series(kept, index=mask.index)


def compute_local_scores(y: pd.Series) -> Dict[str, pd.Series]:
    high_scores: List[pd.Series] = []
    low_scores: List[pd.Series] = []
    for window in LOCAL_BASELINE_WINDOWS:
        baseline = y.rolling(
            window,
            center=True,
            min_periods=max(6, window // 4),
        ).median()
        residual = y - baseline
        scale = robust_scale(
            residual.abs().rolling(
                window,
                center=True,
                min_periods=max(6, window // 4),
            ).median()
        )
        high_scores.append((residual / scale).clip(lower=0.0))
        low_scores.append((-residual / scale).clip(lower=0.0))

    return {
        "high": finite_max(*high_scores),
        "low": finite_max(*low_scores),
    }


def compute_periodic_scores(df: pd.DataFrame, y: pd.Series) -> Dict[str, pd.Series]:
    time_values = pd.to_datetime(df[TIME_COL])
    slot = time_values.dt.hour * 12 + time_values.dt.minute // 5
    slot_baseline = y.groupby(slot).transform("median")
    residual = y - slot_baseline
    slot_mad = residual.abs().groupby(slot).transform("median")
    slot_scale = robust_scale(slot_mad)

    return {
        "high": (residual / slot_scale).clip(lower=0.0),
        "low": (-residual / slot_scale).clip(lower=0.0),
    }


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    local_scores = compute_local_scores(y)
    periodic_scores = compute_periodic_scores(df, y)

    local_high = keep_short_runs(
        local_scores["high"] >= LOCAL_ROBUST_Z_THRESHOLD,
        local_scores["high"],
        LOCAL_ROBUST_Z_THRESHOLD,
    )
    local_low = keep_short_runs(
        local_scores["low"] >= LOCAL_ROBUST_Z_THRESHOLD,
        local_scores["low"],
        LOCAL_ROBUST_Z_THRESHOLD,
    )
    periodic_high = keep_short_runs(
        periodic_scores["high"] >= PERIODIC_ROBUST_Z_THRESHOLD,
        periodic_scores["high"],
        PERIODIC_ROBUST_Z_THRESHOLD,
    )
    periodic_low = keep_short_runs(
        periodic_scores["low"] >= PERIODIC_ROBUST_Z_THRESHOLD,
        periodic_scores["low"],
        PERIODIC_ROBUST_Z_THRESHOLD,
    )
    absolute_low = y < ABS_LOW_THRESHOLD

    marked = df.copy()
    anomaly_type = pd.Series("", index=marked.index, dtype="object")
    anomaly_type = add_anomaly_type(anomaly_type, absolute_low, "absolute_low")
    anomaly_type = add_anomaly_type(anomaly_type, local_high, "local_high")
    anomaly_type = add_anomaly_type(anomaly_type, local_low, "local_low")
    anomaly_type = add_anomaly_type(anomaly_type, periodic_high, "periodic_high")
    anomaly_type = add_anomaly_type(anomaly_type, periodic_low, "periodic_low")

    anomaly_score = finite_max(
        local_scores["high"],
        local_scores["low"],
        periodic_scores["high"],
        periodic_scores["low"],
    )
    is_anomaly = anomaly_type != ""

    marked[ANOMALY_COL] = np.where(is_anomaly, "是", "否")
    marked[ANOMALY_TYPE_COL] = np.where(is_anomaly, anomaly_type, "")
    marked[ANOMALY_SCORE_COL] = np.where(is_anomaly, anomaly_score.round(6), "")
    return marked


def write_marked_csv(route: str, source_path: Path, marked: pd.DataFrame) -> Path:
    df = pd.read_csv(source_path)
    if not df[[TIME_COL, TARGET_COL]].equals(marked[[TIME_COL, TARGET_COL]]):
        raise ValueError(f"{route}: marked data changed original {TIME_COL}/{TARGET_COL} values.")
    output_path = source_path.with_name(MARKED_OUTPUT_NAME)
    marked.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"{route}: saved {output_path}")
    return output_path


def build_cleaned_power(marked: pd.DataFrame) -> pd.DataFrame:
    if ANOMALY_COL not in marked.columns:
        raise ValueError(f"marked data must contain {ANOMALY_COL!r}.")

    cleaned = marked[[TIME_COL, TARGET_COL]].copy()
    cleaned_time = pd.to_datetime(cleaned[TIME_COL])
    cleaned_target = pd.to_numeric(cleaned[TARGET_COL], errors="coerce")
    anomaly_mask = marked[ANOMALY_COL] == "是"

    cleaned_target.loc[anomaly_mask] = np.nan
    cleaned_target.index = cleaned_time
    cleaned_target = cleaned_target.interpolate(method="time").ffill().bfill()
    cleaned_target.index = cleaned.index

    cleaned[TARGET_COL] = cleaned_target
    return cleaned[[TIME_COL, TARGET_COL]]


def write_cleaned_csv(route: str, source_path: Path, marked: pd.DataFrame) -> Path:
    cleaned = build_cleaned_power(marked)
    missing_count = int(cleaned[TARGET_COL].isna().sum())
    output_path = source_path.with_name(CLEANED_OUTPUT_NAME)
    cleaned.to_csv(output_path, index=False, encoding="utf-8")
    print(f"{route}: saved {output_path}, remaining_missing={missing_count}")
    return output_path


def _setup_matplotlib():
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()).joinpath("tsproj_ml_matplotlib")),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_route(route: str, source_path: Path, marked: pd.DataFrame) -> Path:
    plt = _setup_matplotlib()
    plot_df = marked.copy()
    plot_df[TIME_COL] = pd.to_datetime(plot_df[TIME_COL])
    anomaly_mask = plot_df[ANOMALY_COL] == "是"
    anomaly_count = int(anomaly_mask.sum())
    anomaly_rate = anomaly_count / len(plot_df) if len(plot_df) else 0.0

    fig, ax = plt.subplots(figsize=(24, 8))
    ax.plot(
        plot_df[TIME_COL],
        plot_df[TARGET_COL],
        color="#2F5597",
        linewidth=0.9,
        label="h_total_use",
    )

    if anomaly_count:
        anomaly_df = plot_df.loc[anomaly_mask]
        ax.scatter(
            anomaly_df[TIME_COL],
            anomaly_df[TARGET_COL],
            color="#D62728",
            s=24,
            label="Anomaly",
            zorder=3,
        )

    ax.set_title(
        f"{route} raw load with anomalies "
        f"(points={len(plot_df)}, anomalies={anomaly_count}, rate={anomaly_rate:.2%})"
    )
    ax.set_xlabel("time")
    ax.set_ylabel(TARGET_COL)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = source_path.with_name(PLOT_OUTPUT_NAME)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"{route}: saved {output_path}")
    return output_path


def process_power_file(source_path: Path, route: str = None, plot: bool = True) -> Dict[str, Path]:
    source_path = Path(source_path)
    route = route or source_path.parent.name
    df = pd.read_csv(source_path)
    if TIME_COL not in df.columns or TARGET_COL not in df.columns:
        raise ValueError(f"{source_path} must contain {TIME_COL!r} and {TARGET_COL!r}.")

    marked = detect_anomalies(df)
    marked_path = write_marked_csv(route, source_path, marked)
    plot_path = plot_route(route, source_path, marked) if plot else None
    cleaned_path = write_cleaned_csv(route, source_path, marked)
    anomaly_count = int((marked[ANOMALY_COL] == "是").sum())
    type_counts = (
        marked.loc[marked[ANOMALY_COL] == "是", ANOMALY_TYPE_COL]
        .str.split(";")
        .explode()
        .value_counts()
        .to_dict()
    )
    print(f"{route}: total={len(df)}, anomalies={anomaly_count}, type_counts={type_counts}")

    return {
        "marked_csv": marked_path,
        "cleaned_csv": cleaned_path,
        "plot": plot_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="标记、清洗并绘制原始负荷异常点。")
    parser.add_argument("source", type=Path, help="待处理的 df_power.csv 路径。")
    parser.add_argument("--route", type=str, default=None, help="日志中显示的路线名称，默认使用数据目录名。")
    parser.add_argument("--no-plot", action="store_true", help="只输出 CSV，不生成异常点图片。")
    return parser.parse_args()


# 测试代码 main 函数
def main():
    args = parse_args()
    process_power_file(args.source, route=args.route, plot=not args.no_plot)


if __name__ == "__main__":
    main()
