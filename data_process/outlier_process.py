# -*- coding: utf-8 -*-
"""原始负荷数据异常标记、清洗与可视化（配置驱动）。

检测规则（优先级从高到低）：
  1. 物理边界硬截：abs_low / abs_high —— 低于或高于物理合理范围的值直接标记
  2. 孤立跳变检测：spike —— 与短窗口邻域偏差极大的单点/短段（传感器毛刺、通信恢复跳变）
  3. 局部 robust Z-score：local —— 长窗口基线偏离（连续段异常）
  4. 周期 robust Z-score：periodic —— 同时段历史基线偏离

local/periodic Z-score 支持 run-length 过滤（短段保留、长段提高阈值）。

输出（默认源文件同级的 outlier_analysis/ 子目录，文件名从源文件名派生；可在任务中配置 output_dir 覆盖输出目录）：
  outlier_analysis/<stem>_outlier_detection.csv  异常标记（含是否清洗/审核状态）
  outlier_analysis/<stem>_remove_outlier.csv     清洗后数据（异常点 time 插值 + ffill/bfill）
  outlier_analysis/<stem>_anomalies.png          原始/清洗后对比图 + 异常标记
  outlier_analysis/review_days/<stem>/<date>_review.png  统计可疑点的逐日审核图

用法（仓库根目录）：
    uv run python data_process/outlier_process.py <config.yaml>
    uv run python data_process/outlier_process.py <config.yaml> --force

配置 YAML schema（与模型配置完全独立，无 base_config/overrides）：
    # 单任务（顶层平铺）
    source_path: dataset/xxx/df_power.csv
    time_col: count_data_time
    target_col: h_total_use

    # 或多任务（顶层 tasks: 列表）
    tasks:
      - source_path: dataset/xxx/df_power.csv
        time_col: count_data_time
        target_col: h_total_use
        # 可选覆盖默认检测参数
        abs_low_threshold: 500.0

可配置参数（均有默认值）：
    abs_low_threshold:          绝对低值阈值，默认 1000.0
    abs_high_threshold:         绝对高值阈值，默认 inf（不启用）
    spike_window:               spike 邻域半窗口（前后各 N 点），默认 3
    spike_z_threshold:          spike robust Z 阈值，默认 6.0
    spike_abs_diff_threshold:   spike 绝对偏差门槛，默认 0.0（不启用）
    local_baseline_windows:     局部基线窗口列表，默认 [25, 145]
    local_robust_z_threshold:   局部 robust Z 阈值，默认 3.0
    periodic_robust_z_threshold: 周期 robust Z 阈值，默认 2.8

    max_short_run_points:       短段最大点数（≤此长度保留），默认 12
    long_run_score_multiplier:  long run 的阈值倍数，默认 2.5
    review_daily_plots:         对待审核统计点生成逐日全量数据图，默认 false
    auto_clean_statistical:     统计候选是否自动清洗，默认 true（兼容既有场景）
    plot:                       是否生成图片，默认 true
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# 默认检测参数（可在 YAML 中覆盖）
# ---------------------------------------------------------------------------
DEFAULT_ABS_LOW_THRESHOLD = 1000.0
DEFAULT_ABS_HIGH_THRESHOLD = float("inf")
DEFAULT_SPIKE_WINDOW = 3
DEFAULT_SPIKE_Z_THRESHOLD = 6.0
DEFAULT_SPIKE_ABS_DIFF = 0.0
DEFAULT_SPIKE_COMPANION_SPREAD = float("inf")
DEFAULT_LOCAL_BASELINE_WINDOWS = [25, 145]
DEFAULT_LOCAL_ROBUST_Z_THRESHOLD = 3.0
DEFAULT_PERIODIC_ROBUST_Z_THRESHOLD = 2.8
DEFAULT_MAX_SHORT_RUN_POINTS = 12
DEFAULT_LONG_RUN_SCORE_MULTIPLIER = 2.5


ROBUST_SCALE = 1.4826
EPSILON = 1e-9

ANOMALY_COL = "是否异常"
ANOMALY_TYPE_COL = "异常类型"
ANOMALY_SCORE_COL = "异常分数"
AUTO_CLEAN_COL = "是否清洗"
REVIEW_STATUS_COL = "审核状态"

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 检测参数 dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OutlierParams:
    abs_low_threshold: float = DEFAULT_ABS_LOW_THRESHOLD
    abs_high_threshold: float = DEFAULT_ABS_HIGH_THRESHOLD
    spike_window: int = DEFAULT_SPIKE_WINDOW
    spike_z_threshold: float = DEFAULT_SPIKE_Z_THRESHOLD
    spike_abs_diff_threshold: float = DEFAULT_SPIKE_ABS_DIFF
    spike_companion_spread_threshold: float = DEFAULT_SPIKE_COMPANION_SPREAD
    local_baseline_windows: List[int] = None  # type: ignore[assignment]
    local_robust_z_threshold: float = DEFAULT_LOCAL_ROBUST_Z_THRESHOLD
    periodic_robust_z_threshold: float = DEFAULT_PERIODIC_ROBUST_Z_THRESHOLD
    max_short_run_points: int = DEFAULT_MAX_SHORT_RUN_POINTS
    long_run_score_multiplier: float = DEFAULT_LONG_RUN_SCORE_MULTIPLIER
    auto_clean_statistical: bool = True
    auto_clean_spike: bool = False


    def __post_init__(self):
        if self.local_baseline_windows is None:
            object.__setattr__(self, "local_baseline_windows", list(DEFAULT_LOCAL_BASELINE_WINDOWS))


@dataclass(frozen=True)
class OutlierSpec:
    """单个异常检测任务规格。"""
    source_path: Path
    time_col: str
    target_col: str
    params: OutlierParams
    plot: bool = True
    review_daily_plots: bool = False
    route: str = ""
    output_dir: Optional[Path] = None


@dataclass(frozen=True)
class OutlierResult:
    marked_csv: Path
    cleaned_csv: Path
    plot: Optional[Path]
    review_plots: List[Path]
    total_rows: int
    anomaly_count: int
    type_counts: Dict[str, int]


# ---------------------------------------------------------------------------
# 核心检测算法（与原版一致，参数化）
# ---------------------------------------------------------------------------
def remove_outliers(df: pd.DataFrame, column: str, threshold: int = 3) -> pd.DataFrame:
    """移除异常值（Z-score 方法）。"""
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]


def _robust_scale(values: pd.Series) -> pd.Series:
    scale = values * ROBUST_SCALE
    return scale.mask(scale <= EPSILON)


def _finite_max(*series_list: pd.Series) -> pd.Series:
    frame = pd.concat(series_list, axis=1)
    return frame.replace([np.inf, -np.inf], np.nan).max(axis=1).fillna(0.0)


def _add_anomaly_type(types: pd.Series, mask: pd.Series, label: str) -> pd.Series:
    mask = mask.fillna(False).astype(bool)
    result = types.copy()
    result.loc[mask & (result == "")] = label
    result.loc[mask & (result != "") & ~result.str.contains(label, regex=False)] += f";{label}"
    return result


def _keep_short_runs(
    mask: pd.Series,
    score: pd.Series,
    threshold: float,
    max_short_run: int,
    long_run_multiplier: float,
) -> pd.Series:
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
            if len(run) <= max_short_run:
                kept[run] = True
            else:
                kept[run] = scores[run] >= threshold * long_run_multiplier
            start = None
    return pd.Series(kept, index=mask.index)


def _keep_runs_at_least(mask: pd.Series, min_run_points: int) -> pd.Series:
    """仅保留长度不小于给定值的连续候选段。"""
    values = mask.fillna(False).to_numpy(dtype=bool)
    kept = np.zeros(len(values), dtype=bool)
    start = None
    for pos, is_true in enumerate(values):
        if is_true and start is None:
            start = pos
        if start is not None and (not is_true or pos == len(values) - 1):
            end = pos if is_true else pos - 1
            if end - start + 1 >= min_run_points:
                kept[start:end + 1] = True
            start = None
    return pd.Series(kept, index=mask.index)


def _compute_spike_scores(y: pd.Series, half_window: int) -> Dict[str, pd.Series]:
    """孤立跳变检测：用前后各 half_window 个点的中位数做局部基线。

    返回两个信号：
      - "z":  residual / (local MAD × 1.4826)，衡量相对偏离强度
      - "abs_diff": |residual|，衡量绝对偏离量级

    两者必须联合使用：z-score 在低噪声段会爆炸（MAD→0），abs_diff 在
    高量级段会过度触发。只有同时满足 z 和 abs_diff 门槛才是真 spike。
    """
    window = 2 * half_window + 1
    min_p = max(3, window // 2)
    baseline = y.rolling(window, center=True, min_periods=min_p).median()
    residual = y - baseline
    scale = _robust_scale(
        residual.abs().rolling(window, center=True, min_periods=min_p).median()
    )
    spike_z = (residual.abs() / scale).fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return {
        "z": spike_z,
        "abs_diff": residual.abs().fillna(0.0),
        "companion_spread": (y.shift(1) - y.shift(-1)).abs(),
    }


def _compute_local_scores(y: pd.Series, windows: List[int]) -> Dict[str, pd.Series]:
    high_scores: List[pd.Series] = []
    low_scores: List[pd.Series] = []
    for window in windows:
        baseline = y.rolling(window, center=True, min_periods=max(6, window // 4)).median()
        residual = y - baseline
        scale = _robust_scale(
            residual.abs().rolling(window, center=True, min_periods=max(6, window // 4)).median()
        )
        high_scores.append((residual / scale).clip(lower=0.0))
        low_scores.append((-residual / scale).clip(lower=0.0))
    return {"high": _finite_max(*high_scores), "low": _finite_max(*low_scores)}


def _compute_periodic_scores(
    df: pd.DataFrame,
    y: pd.Series,
    time_col: str,
) -> Dict[str, pd.Series]:
    time_values = pd.to_datetime(df[time_col])
    # 自动检测槽位粒度：用中位时间差确定每小时槽数
    deltas = time_values.diff().dropna()
    if len(deltas) > 0:
        median_seconds = deltas.median().total_seconds()
        slots_per_hour = max(1, round(3600 / median_seconds))
    else:
        slots_per_hour = 12  # 默认 5min
    slot = time_values.dt.hour * slots_per_hour + (time_values.dt.minute * 60 + time_values.dt.second) // int(3600 / slots_per_hour)
    slot_baseline = y.groupby(slot).transform("median")
    residual = y - slot_baseline
    slot_mad = residual.abs().groupby(slot).transform("median")
    slot_scale = _robust_scale(slot_mad)
    return {"high": (residual / slot_scale).clip(lower=0.0), "low": (-residual / slot_scale).clip(lower=0.0)}


def detect_anomalies(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    params: OutlierParams,
) -> pd.DataFrame:
    """对 DataFrame 执行异常检测，返回带标记列的副本。

    检测优先级：物理边界 > 孤立跳变 > 局部偏离 > 周期偏离。
    """
    y = pd.to_numeric(df[target_col], errors="coerce")

    # --- 1. 物理边界硬截 ---
    abs_high: float = float(params.abs_high_threshold)
    absolute_low = y <= params.abs_low_threshold
    if np.isfinite(abs_high):
        absolute_high = y > abs_high
    else:
        absolute_high = pd.Series(False, index=y.index)

    # --- 2. 孤立跳变检测（z-score + abs_diff 双条件） ---
    spike_scores = _compute_spike_scores(y, params.spike_window)
    spike_has_large_residual = spike_scores["abs_diff"] >= params.spike_abs_diff_threshold
    spike_has_large_z = spike_scores["z"] >= params.spike_z_threshold
    if np.isfinite(params.spike_companion_spread_threshold):
        spike_has_stable_companions = (
            spike_scores["companion_spread"] <= params.spike_companion_spread_threshold
        )
    else:
        spike_has_stable_companions = pd.Series(False, index=y.index)
    spike_mask = spike_has_large_residual & (spike_has_large_z | spike_has_stable_companions)

    # --- 3. 局部 robust Z-score ---
    local_scores = _compute_local_scores(y, params.local_baseline_windows)
    periodic_scores = _compute_periodic_scores(df, y, time_col)

    local_high = _keep_short_runs(
        local_scores["high"] >= params.local_robust_z_threshold,
        local_scores["high"],
        params.local_robust_z_threshold,
        params.max_short_run_points,
        params.long_run_score_multiplier,
    )
    local_low = _keep_short_runs(
        local_scores["low"] >= params.local_robust_z_threshold,
        local_scores["low"],
        params.local_robust_z_threshold,
        params.max_short_run_points,
        params.long_run_score_multiplier,
    )

    # --- 4. 周期 robust Z-score ---
    periodic_high = _keep_short_runs(
        periodic_scores["high"] >= params.periodic_robust_z_threshold,
        periodic_scores["high"],
        params.periodic_robust_z_threshold,
        params.max_short_run_points,
        params.long_run_score_multiplier,
    )
    periodic_low = _keep_short_runs(
        periodic_scores["low"] >= params.periodic_robust_z_threshold,
        periodic_scores["low"],
        params.periodic_robust_z_threshold,
        params.max_short_run_points,
        params.long_run_score_multiplier,
    )

    marked = df.copy()
    anomaly_type = pd.Series("", index=marked.index, dtype="object")
    anomaly_type = _add_anomaly_type(anomaly_type, absolute_low, "absolute_low")
    anomaly_type = _add_anomaly_type(anomaly_type, absolute_high, "absolute_high")
    anomaly_type = _add_anomaly_type(anomaly_type, spike_mask, "spike")
    anomaly_type = _add_anomaly_type(anomaly_type, local_high, "local_high")
    anomaly_type = _add_anomaly_type(anomaly_type, local_low, "local_low")
    anomaly_type = _add_anomaly_type(anomaly_type, periodic_high, "periodic_high")
    anomaly_type = _add_anomaly_type(anomaly_type, periodic_low, "periodic_low")

    anomaly_score = _finite_max(
        spike_scores["z"],
        local_scores["high"], local_scores["low"],
        periodic_scores["high"], periodic_scores["low"],
    )
    physical_mask = absolute_low | absolute_high
    is_anomaly = anomaly_type != ""
    statistical_mask = is_anomaly & ~physical_mask
    auto_clean_mask = (
        physical_mask
        | (spike_mask & (params.auto_clean_spike | params.auto_clean_statistical))
        | (statistical_mask & ~spike_mask & params.auto_clean_statistical)
    )
    review_mask = statistical_mask & ~params.auto_clean_statistical
    marked[ANOMALY_COL] = np.where(is_anomaly, "是", "否")
    marked[ANOMALY_TYPE_COL] = np.where(is_anomaly, anomaly_type, "")
    marked[ANOMALY_SCORE_COL] = np.where(is_anomaly, anomaly_score.round(6), "")
    marked[AUTO_CLEAN_COL] = np.where(auto_clean_mask, "是", "否")
    marked[REVIEW_STATUS_COL] = np.select(
        [auto_clean_mask, review_mask], ["自动清洗", "待审核"], default="无"
    )
    return marked


# ---------------------------------------------------------------------------
# 清洗
# ---------------------------------------------------------------------------
def _physical_boundary_mask(y: pd.Series, params: OutlierParams) -> pd.Series:
    """返回不满足物理上下界或非数值的点掩码。"""
    invalid = y.isna() | (y < params.abs_low_threshold)
    if np.isfinite(params.abs_high_threshold):
        invalid |= y > params.abs_high_threshold
    return invalid


def _interpolate_by_time(y: pd.Series, time_values: pd.Series) -> pd.Series:
    """按时间插值并补齐边缘空值，保留原行索引。"""
    result = y.copy()
    result.index = pd.to_datetime(time_values)
    result = result.interpolate(method="time").ffill().bfill()
    result.index = y.index
    return result


def build_cleaned_series(
    marked: pd.DataFrame,
    time_col: str,
    target_col: str,
    params: OutlierParams,
) -> pd.DataFrame:
    """清洗异常值，并在输出前强制复检物理边界。

    首轮按检测标记置空；随后把所有非数值和超出物理边界的残留值再次置空。
    该守卫不依赖检测是否漏标，保证清洗 CSV 不会输出越界值。
    """
    if AUTO_CLEAN_COL not in marked.columns and ANOMALY_COL not in marked.columns:
        raise ValueError(
            f"marked data must contain {AUTO_CLEAN_COL!r} or {ANOMALY_COL!r}."
        )
    cleaned = marked[[time_col, target_col]].copy()
    cleaned_time = pd.to_datetime(cleaned[time_col])
    cleaned_target = pd.to_numeric(cleaned[target_col], errors="coerce")
    # 仅物理越界自动清洗；统计可疑点保留原值，交由逐日审核图人工判定。
    if AUTO_CLEAN_COL in marked.columns:
        clean_mask = marked[AUTO_CLEAN_COL] == "是"
    else:
        # 兼容旧版 detection CSV。
        clean_mask = marked[ANOMALY_COL] == "是"
    cleaned_target.loc[clean_mask] = np.nan

    # 检测漏标、源非数值和插值边界都由此处统一兜底。
    for _ in range(2):
        cleaned_target.loc[_physical_boundary_mask(cleaned_target, params)] = np.nan
        cleaned_target = _interpolate_by_time(cleaned_target, cleaned_time)

    final_invalid = _physical_boundary_mask(cleaned_target, params)
    if final_invalid.any():
        raise ValueError(
            "Unable to produce a physical-boundary-compliant cleaned series: "
            f"{int(final_invalid.sum())} invalid rows remain."
        )
    cleaned[target_col] = cleaned_target
    return cleaned[[time_col, target_col]]


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
def _setup_matplotlib():
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()).joinpath("tsproj_ml_matplotlib")),
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_anomalies(
    marked: pd.DataFrame,
    cleaned: pd.DataFrame,
    time_col: str,
    target_col: str,
    route: str,
    output_path: Path,
) -> Path:
    """输出原始异常标记与清洗后结果的双面板对比图。"""
    plt = _setup_matplotlib()
    plot_df = marked.copy()
    plot_df[time_col] = pd.to_datetime(plot_df[time_col])
    cleaned_df = cleaned.copy()
    cleaned_df[time_col] = pd.to_datetime(cleaned_df[time_col])
    anomaly_mask = plot_df[ANOMALY_COL] == "是"
    auto_clean_mask = plot_df[AUTO_CLEAN_COL] == "是"
    review_mask = plot_df[REVIEW_STATUS_COL] == "待审核"
    anomaly_count = int(anomaly_mask.sum())
    anomaly_rate = anomaly_count / len(plot_df) if len(plot_df) else 0.0
    replaced_mask = auto_clean_mask | (
        pd.to_numeric(plot_df[target_col], errors="coerce")
        != pd.to_numeric(cleaned_df[target_col], errors="coerce")
    )

    fig, (raw_ax, clean_ax) = plt.subplots(2, 1, figsize=(24, 12), sharex=True)
    raw_ax.plot(plot_df[time_col], plot_df[target_col],
                color="#2F5597", linewidth=0.8, label="raw")
    if auto_clean_mask.any():
        physical_df = plot_df.loc[auto_clean_mask]
        raw_ax.scatter(physical_df[time_col], physical_df[target_col],
                       color="#D62728", s=24, label="auto-cleaned", zorder=3)
    if review_mask.any():
        review_df = plot_df.loc[review_mask]
        raw_ax.scatter(review_df[time_col], review_df[target_col],
                       color="#FF7F0E", s=24, label="review required", zorder=3)
    raw_ax.set_title(
        f"{route}: raw load and detected anomalies "
        f"(points={len(plot_df)}, anomalies={anomaly_count}, rate={anomaly_rate:.2%})"
    )
    raw_ax.set_ylabel(target_col)
    raw_ax.grid(True, alpha=0.3)
    raw_ax.legend(loc="best")

    # 下方面板不画已替换的极端原始值，否则坐标轴会被脏值拉伸、清洗后的正常区间不可读。
    raw_reference = pd.Series(
        pd.to_numeric(plot_df[target_col], errors="coerce"), index=plot_df.index
    ).mask(replaced_mask)
    clean_ax.plot(plot_df[time_col], raw_reference,
                  color="#B8B8B8", linewidth=0.6, alpha=0.65, label="raw (unchanged)")
    clean_ax.plot(cleaned_df[time_col], cleaned_df[target_col],
                  color="#2CA02C", linewidth=0.9, label="cleaned")
    if replaced_mask.any():
        clean_ax.scatter(
            cleaned_df.loc[replaced_mask, time_col],
            cleaned_df.loc[replaced_mask, target_col],
            color="#FF7F0E", s=20, label="replaced value", zorder=3,
        )
    clean_ax.set_title("Cleaned result (orange points are values replaced during cleaning)")
    clean_ax.set_xlabel("time")
    clean_ax.set_ylabel(target_col)
    clean_ax.grid(True, alpha=0.3)
    clean_ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_review_days(
    marked: pd.DataFrame,
    time_col: str,
    target_col: str,
    route: str,
    output_dir: Path,
) -> List[Path]:
    """对每个含统计可疑点的日期输出当天完整时序，供人工审核。"""
    if REVIEW_STATUS_COL not in marked.columns:
        return []

    plot_df = marked.copy()
    plot_df[time_col] = pd.to_datetime(plot_df[time_col])
    review_mask = plot_df[REVIEW_STATUS_COL] == "待审核"
    if not review_mask.any():
        return []

    plt = _setup_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dates = plot_df.loc[review_mask, time_col].dt.normalize().drop_duplicates().sort_values()
    output_paths: List[Path] = []
    for review_date in review_dates:
        next_date = review_date + pd.Timedelta(days=1)
        day_mask = (plot_df[time_col] >= review_date) & (plot_df[time_col] < next_date)
        day_df = plot_df.loc[day_mask]
        day_review_mask = day_df[REVIEW_STATUS_COL] == "待审核"
        day_clean_mask = day_df[AUTO_CLEAN_COL] == "是"

        fig, ax = plt.subplots(figsize=(16, 7))
        ax.plot(day_df[time_col], day_df[target_col], color="#2F5597", linewidth=1.1, label="raw")
        if day_clean_mask.any():
            auto_cleaned = day_df.loc[day_clean_mask]
            ax.scatter(auto_cleaned[time_col], auto_cleaned[target_col], color="#D62728",
                       s=48, marker="x", label="auto-cleaned", zorder=3)
        review_points = day_df.loc[day_review_mask]
        ax.scatter(review_points[time_col], review_points[target_col], color="#FF7F0E",
                   s=58, label="review required", zorder=3)
        ax.set_title(
            f"{route}: full-day review — {review_date:%Y-%m-%d} "
            f"(review points={len(review_points)})"
        )
        ax.set_xlabel("time")
        ax.set_ylabel(target_col)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.autofmt_xdate()
        fig.tight_layout()
        output_path = output_dir / f"{review_date:%Y-%m-%d}_review.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths


# ---------------------------------------------------------------------------
# 单任务执行
# ---------------------------------------------------------------------------
def _output_paths(source_path: Path, output_dir: Optional[Path] = None) -> Dict[str, Path]:
    stem = source_path.stem
    out_dir = output_dir or (source_path.parent / "outlier_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "marked": out_dir / f"{stem}_outlier_detection.csv",
        "cleaned": out_dir / f"{stem}_remove_outlier.csv",
        "plot": out_dir / f"{stem}_anomalies.png",
        "review_dir": out_dir / "review_days" / stem,
    }


def process_outlier(spec: OutlierSpec) -> OutlierResult:
    """执行单个异常检测任务：检测 → 标记 → 清洗 → 可视化。"""
    source_path = spec.source_path
    route = spec.route or source_path.parent.name

    df = pd.read_csv(source_path)
    if spec.time_col not in df.columns or spec.target_col not in df.columns:
        raise ValueError(f"{source_path} must contain {spec.time_col!r} and {spec.target_col!r}.")

    marked = detect_anomalies(df, spec.time_col, spec.target_col, spec.params)
    paths = _output_paths(source_path, spec.output_dir)

    # 写标记 CSV
    original = pd.read_csv(source_path)
    if not original[[spec.time_col, spec.target_col]].equals(marked[[spec.time_col, spec.target_col]]):
        raise ValueError(f"{route}: marked data changed original values.")
    marked.to_csv(paths["marked"], index=False, encoding="utf-8-sig")

    # 写清洗 CSV
    cleaned = build_cleaned_series(marked, spec.time_col, spec.target_col, spec.params)
    cleaned.to_csv(paths["cleaned"], index=False, encoding="utf-8")

    # 写图片
    plot_path = None
    if spec.plot:
        plot_path = plot_anomalies(
            marked, cleaned, spec.time_col, spec.target_col, route, paths["plot"]
        )

    review_plots: List[Path] = []
    if spec.review_daily_plots:
        review_plots = plot_review_days(
            marked, spec.time_col, spec.target_col, route, paths["review_dir"]
        )

    anomaly_count = int((marked[ANOMALY_COL] == "是").sum())
    type_counts = (
        marked.loc[marked[ANOMALY_COL] == "是", ANOMALY_TYPE_COL]
        .str.split(";").explode().value_counts().to_dict()
    )
    print(f"{route}: total={len(df)}, anomalies={anomaly_count}, type_counts={type_counts}")
    print(f"  -> {paths['marked']}")
    print(f"  -> {paths['cleaned']}")
    if plot_path:
        print(f"  -> {plot_path}")
    if review_plots:
        print(f"  -> {len(review_plots)} review-day plots under {paths['review_dir']}")

    return OutlierResult(
        marked_csv=paths["marked"],
        cleaned_csv=paths["cleaned"],
        plot=plot_path,
        review_plots=review_plots,
        total_rows=len(df),
        anomaly_count=anomaly_count,
        type_counts=type_counts,
    )


# ---------------------------------------------------------------------------
# 配置加载与批量入口（与 data_aggregate 同模式）
# ---------------------------------------------------------------------------
def _load_config(config_path: str | Path) -> dict[str, Any]:
    import yaml
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Outlier config must be a mapping: {config_path}")
    return raw  # type: ignore[return-value]


def _resolve_path(p: str) -> Path:
    pr = Path(p).expanduser()
    return pr if pr.is_absolute() else (PROJECT_ROOT / pr).resolve()


def _build_params(raw: dict[str, Any]) -> OutlierParams:
    return OutlierParams(
        abs_low_threshold=float(raw.get("abs_low_threshold", DEFAULT_ABS_LOW_THRESHOLD)),
        abs_high_threshold=float(raw.get("abs_high_threshold", DEFAULT_ABS_HIGH_THRESHOLD)),
        spike_window=int(raw.get("spike_window", DEFAULT_SPIKE_WINDOW)),
        spike_z_threshold=float(raw.get("spike_z_threshold", DEFAULT_SPIKE_Z_THRESHOLD)),
        spike_abs_diff_threshold=float(raw.get("spike_abs_diff_threshold", DEFAULT_SPIKE_ABS_DIFF)),
        spike_companion_spread_threshold=float(
            raw.get("spike_companion_spread_threshold", DEFAULT_SPIKE_COMPANION_SPREAD)
        ),
        local_baseline_windows=raw.get("local_baseline_windows", None),
        local_robust_z_threshold=float(raw.get("local_robust_z_threshold", DEFAULT_LOCAL_ROBUST_Z_THRESHOLD)),
        periodic_robust_z_threshold=float(raw.get("periodic_robust_z_threshold", DEFAULT_PERIODIC_ROBUST_Z_THRESHOLD)),
        max_short_run_points=int(raw.get("max_short_run_points", DEFAULT_MAX_SHORT_RUN_POINTS)),
        long_run_score_multiplier=float(raw.get("long_run_score_multiplier", DEFAULT_LONG_RUN_SCORE_MULTIPLIER)),
        auto_clean_statistical=bool(raw.get("auto_clean_statistical", True)),
        auto_clean_spike=bool(raw.get("auto_clean_spike", False)),
    )


def _build_spec(raw: dict[str, Any], config_path: Path) -> OutlierSpec:
    missing = [k for k in ("source_path", "time_col", "target_col") if k not in raw]
    if missing:
        raise ValueError(f"Outlier config missing required fields {missing}: {config_path}")
    return OutlierSpec(
        source_path=_resolve_path(raw["source_path"]),
        time_col=raw["time_col"],
        target_col=raw["target_col"],
        params=_build_params(raw),
        plot=bool(raw.get("plot", True)),
        review_daily_plots=bool(raw.get("review_daily_plots", False)),
        route=raw.get("route", ""),
        output_dir=_resolve_path(raw["output_dir"]) if raw.get("output_dir") else None,
    )


def run_outlier_detection(config_path: str | Path, *, force: bool = False) -> None:
    """加载 YAML 异常检测配置并执行。

    配置文件可写单个任务（顶层平铺）或多任务（顶层 tasks: 列表）。
    """
    config_path = Path(config_path).resolve()
    raw = _load_config(config_path)
    task_list = raw["tasks"] if "tasks" in raw else [raw]
    for item in task_list:
        spec = _build_spec(item, config_path)
        # --force 时删除旧输出
        if force:
            for p in _output_paths(spec.source_path, spec.output_dir).values():
                if p.is_dir():
                    continue
                p.unlink(missing_ok=True)
        process_outlier(spec)


def main() -> None:
    parser = argparse.ArgumentParser(description="原始负荷数据异常标记、清洗与可视化（配置驱动）")
    parser.add_argument("config", help="异常检测配置 YAML 路径")
    parser.add_argument("--force", action="store_true", help="删除旧输出强制重建")
    args = parser.parse_args()
    run_outlier_detection(args.config, force=args.force)


if __name__ == "__main__":
    main()
