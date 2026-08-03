# -*- coding: utf-8 -*-
"""原始负荷数据异常标记、清洗与可视化（配置驱动）。

检测规则：绝对低值 + 局部 robust Z-score + 周期 robust Z-score，
支持 run-length 过滤（短段保留、长段提高阈值）。

输出（源文件同级的 outlier_analysis/ 子目录，文件名从源文件名派生）：
  outlier_analysis/<stem>_outlier_detection.csv  异常标记（原始数据 + 是否异常/异常类型/异常分数）
  outlier_analysis/<stem>_remove_outlier.csv     清洗后数据（异常点 time 插值 + ffill/bfill）
  outlier_analysis/<stem>_anomalies.png          全序列折线图 + 红点标异常

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
    local_baseline_windows:     局部基线窗口列表，默认 [25, 145]
    local_robust_z_threshold:   局部 robust Z 阈值，默认 3.0
    periodic_robust_z_threshold: 周期 robust Z 阈值，默认 2.8
    max_short_run_points:       短段最大点数（≤此长度保留），默认 12
    long_run_score_multiplier:  长段分数乘数，默认 2.5
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 检测参数 dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OutlierParams:
    abs_low_threshold: float = DEFAULT_ABS_LOW_THRESHOLD
    local_baseline_windows: List[int] = None  # type: ignore[assignment]
    local_robust_z_threshold: float = DEFAULT_LOCAL_ROBUST_Z_THRESHOLD
    periodic_robust_z_threshold: float = DEFAULT_PERIODIC_ROBUST_Z_THRESHOLD
    max_short_run_points: int = DEFAULT_MAX_SHORT_RUN_POINTS
    long_run_score_multiplier: float = DEFAULT_LONG_RUN_SCORE_MULTIPLIER

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
    route: str = ""


@dataclass(frozen=True)
class OutlierResult:
    marked_csv: Path
    cleaned_csv: Path
    plot: Optional[Path]
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
    """对 DataFrame 执行异常检测，返回带标记列的副本。"""
    y = pd.to_numeric(df[target_col], errors="coerce")
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
    absolute_low = y < params.abs_low_threshold

    marked = df.copy()
    anomaly_type = pd.Series("", index=marked.index, dtype="object")
    anomaly_type = _add_anomaly_type(anomaly_type, absolute_low, "absolute_low")
    anomaly_type = _add_anomaly_type(anomaly_type, local_high, "local_high")
    anomaly_type = _add_anomaly_type(anomaly_type, local_low, "local_low")
    anomaly_type = _add_anomaly_type(anomaly_type, periodic_high, "periodic_high")
    anomaly_type = _add_anomaly_type(anomaly_type, periodic_low, "periodic_low")

    anomaly_score = _finite_max(
        local_scores["high"], local_scores["low"],
        periodic_scores["high"], periodic_scores["low"],
    )
    is_anomaly = anomaly_type != ""
    marked[ANOMALY_COL] = np.where(is_anomaly, "是", "否")
    marked[ANOMALY_TYPE_COL] = np.where(is_anomaly, anomaly_type, "")
    marked[ANOMALY_SCORE_COL] = np.where(is_anomaly, anomaly_score.round(6), "")
    return marked


# ---------------------------------------------------------------------------
# 清洗
# ---------------------------------------------------------------------------
def build_cleaned_series(
    marked: pd.DataFrame,
    time_col: str,
    target_col: str,
) -> pd.DataFrame:
    """异常点置 NaN → time 插值 → ffill/bfill，返回清洗后的 time + target 两列。"""
    if ANOMALY_COL not in marked.columns:
        raise ValueError(f"marked data must contain {ANOMALY_COL!r}.")
    cleaned = marked[[time_col, target_col]].copy()
    cleaned_time = pd.to_datetime(cleaned[time_col])
    cleaned_target = pd.to_numeric(cleaned[target_col], errors="coerce")
    anomaly_mask = marked[ANOMALY_COL] == "是"
    cleaned_target.loc[anomaly_mask] = np.nan
    cleaned_target.index = cleaned_time
    cleaned_target = cleaned_target.interpolate(method="time").ffill().bfill()
    cleaned_target.index = cleaned.index
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
    time_col: str,
    target_col: str,
    route: str,
    output_path: Path,
) -> Path:
    plt = _setup_matplotlib()
    plot_df = marked.copy()
    plot_df[time_col] = pd.to_datetime(plot_df[time_col])
    anomaly_mask = plot_df[ANOMALY_COL] == "是"
    anomaly_count = int(anomaly_mask.sum())
    anomaly_rate = anomaly_count / len(plot_df) if len(plot_df) else 0.0

    fig, ax = plt.subplots(figsize=(24, 8))
    ax.plot(plot_df[time_col], plot_df[target_col],
            color="#2F5597", linewidth=0.9, label=target_col)
    if anomaly_count:
        anomaly_df = plot_df.loc[anomaly_mask]
        ax.scatter(anomaly_df[time_col], anomaly_df[target_col],
                   color="#D62728", s=24, label="Anomaly", zorder=3)
    ax.set_title(
        f"{route} raw load with anomalies "
        f"(points={len(plot_df)}, anomalies={anomaly_count}, rate={anomaly_rate:.2%})"
    )
    ax.set_xlabel("time")
    ax.set_ylabel(target_col)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 单任务执行
# ---------------------------------------------------------------------------
def _output_paths(source_path: Path) -> Dict[str, Path]:
    stem = source_path.stem
    out_dir = source_path.parent / "outlier_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "marked": out_dir / f"{stem}_outlier_detection.csv",
        "cleaned": out_dir / f"{stem}_remove_outlier.csv",
        "plot": out_dir / f"{stem}_anomalies.png",
    }


def process_outlier(spec: OutlierSpec) -> OutlierResult:
    """执行单个异常检测任务：检测 → 标记 → 清洗 → 可视化。"""
    source_path = spec.source_path
    route = spec.route or source_path.parent.name

    df = pd.read_csv(source_path)
    if spec.time_col not in df.columns or spec.target_col not in df.columns:
        raise ValueError(f"{source_path} must contain {spec.time_col!r} and {spec.target_col!r}.")

    marked = detect_anomalies(df, spec.time_col, spec.target_col, spec.params)
    paths = _output_paths(source_path)

    # 写标记 CSV
    original = pd.read_csv(source_path)
    if not original[[spec.time_col, spec.target_col]].equals(marked[[spec.time_col, spec.target_col]]):
        raise ValueError(f"{route}: marked data changed original values.")
    marked.to_csv(paths["marked"], index=False, encoding="utf-8-sig")

    # 写清洗 CSV
    cleaned = build_cleaned_series(marked, spec.time_col, spec.target_col)
    cleaned.to_csv(paths["cleaned"], index=False, encoding="utf-8")

    # 写图片
    plot_path = None
    if spec.plot:
        plot_path = plot_anomalies(marked, spec.time_col, spec.target_col, route, paths["plot"])

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

    return OutlierResult(
        marked_csv=paths["marked"],
        cleaned_csv=paths["cleaned"],
        plot=plot_path,
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
        local_baseline_windows=raw.get("local_baseline_windows", None),
        local_robust_z_threshold=float(raw.get("local_robust_z_threshold", DEFAULT_LOCAL_ROBUST_Z_THRESHOLD)),
        periodic_robust_z_threshold=float(raw.get("periodic_robust_z_threshold", DEFAULT_PERIODIC_ROBUST_Z_THRESHOLD)),
        max_short_run_points=int(raw.get("max_short_run_points", DEFAULT_MAX_SHORT_RUN_POINTS)),
        long_run_score_multiplier=float(raw.get("long_run_score_multiplier", DEFAULT_LONG_RUN_SCORE_MULTIPLIER)),
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
        route=raw.get("route", ""),
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
            for p in _output_paths(spec.source_path).values():
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
