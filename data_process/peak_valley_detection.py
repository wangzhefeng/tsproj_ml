# -*- coding: utf-8 -*-
"""时间序列峰谷检测与提取工具（配置驱动）。

基于 scipy.signal.find_peaks：对目标序列找局部极大值（峰）与
局部极小值（谷，即对 -y 找峰），支持 height/distance/prominence/width
过滤，按幅度排序输出 Top-N 峰谷。

输出（源文件同级的 peak_valley_analysis/ 子目录，文件名从源文件名派生）：
  <stem>_peaks_valleys.csv  峰谷明细（类型/位置索引/时间戳/值/幅度排序）
  <stem>_peaks_valleys.png  全序列折线图 + 峰谷标注

用法（仓库根目录）：
    uv run python data_process/peak_valley_detection.py <config.yaml>
    uv run python data_process/peak_valley_detection.py <config.yaml> --force

配置 YAML schema（与模型配置完全独立，无 base_config/overrides）：
    # 单任务（顶层平铺）
    source_path: dataset/xxx/<file>.csv
    time_col: time
    target_col: value

    # 或多任务（顶层 tasks: 列表）
    tasks:
      - source_path: dataset/xxx/<file>.csv
        time_col: time
        target_col: value
        # 可选覆盖
        height: null        # find_peaks height 过滤（绝对高度下限）
        distance: 1         # 相邻峰谷最小间距（样本数），默认 1
        prominence: null    # 峰谷显著度过滤（相对相邻极值）
        width: null         # 峰宽过滤（样本数）
        top_n: null         # 只保留幅度最大的前 N 个（null=全部）
        plot: true          # 是否生成标注图，默认 true
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 规格 dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PeakValleySpec:
    source_path: Path
    time_col: str
    target_col: str
    height: Optional[float] = None
    distance: int = 1
    prominence: Optional[float] = None
    width: Optional[float] = None
    top_n: Optional[int] = None
    plot: bool = True
    route: str = ""
    output_dir: Optional[Path] = None


# ---------------------------------------------------------------------------
# 核心检测算法
# ---------------------------------------------------------------------------
def _find_peaks_filtered(y: np.ndarray, *, negate: bool, spec: PeakValleySpec) -> list[dict]:
    """对 y（或 -y）执行 find_peaks，返回 [{index, value, ...}] 列表。

    negate=True 时对 -y 找峰，即原始序列的谷。
    """
    series = -y if negate else y
    kwargs: dict[str, Any] = {"distance": int(spec.distance)}
    if spec.height is not None:
        kwargs["height"] = spec.height
    if spec.prominence is not None:
        kwargs["prominence"] = spec.prominence
    if spec.width is not None:
        kwargs["width"] = spec.width
    indices, properties = find_peaks(series, **kwargs)

    rows = []
    for idx in indices:
        raw_value = float(y[idx])
        prominence_val = None
        if "prominences" in properties:
            prominence_val = float(properties["prominences"][np.where(indices == idx)[0][0]])
        rows.append({
            "index": int(idx),
            "value": raw_value,
            "prominence": prominence_val,
        })
    return rows


def detect_peaks_valleys(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    spec: PeakValleySpec,
) -> pd.DataFrame:
    """对 DataFrame 执行峰谷检测，返回带 type/time 列的明细表。"""
    frame = df[[time_col, target_col]].copy()
    frame[time_col] = pd.to_datetime(frame[time_col])
    frame = frame.sort_values(time_col)
    y = pd.to_numeric(frame[target_col], errors="coerce").dropna().to_numpy()
    if len(y) < 3:
        raise ValueError(f"目标列有效样本不足（<3）：{target_col}")

    peak_rows = _find_peaks_filtered(y, negate=False, spec=spec)
    valley_rows = _find_peaks_filtered(y, negate=True, spec=spec)
    for row in peak_rows:
        row["type"] = "peak"
    for row in valley_rows:
        row["type"] = "valley"

    all_rows = peak_rows + valley_rows
    if not all_rows:
        # 返回空表（保持列结构一致）
        return pd.DataFrame(columns=["type", "index", "time", "value", "prominence", "rank"])

    detail = pd.DataFrame(all_rows)
    detail = detail.sort_values("index").reset_index(drop=True)
    detail["time"] = frame[time_col].to_numpy()[detail["index"].to_numpy()]
    detail = detail[["type", "index", "time", "value", "prominence"]]

    # 幅度排序：|值 - 全序列中位数| 越大越显著
    median_val = float(np.median(y))
    detail["amplitude"] = (detail["value"] - median_val).abs()
    detail = detail.sort_values("amplitude", ascending=False).reset_index(drop=True)
    detail["rank"] = np.arange(1, len(detail) + 1)
    detail = detail.sort_values("index").reset_index(drop=True)

    if spec.top_n is not None and spec.top_n > 0:
        detail = detail[detail["rank"] <= spec.top_n].sort_values("index").reset_index(drop=True)

    return detail[["type", "index", "time", "value", "prominence", "rank"]]


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
def _setup_matplotlib():
    import os
    import tempfile

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()).joinpath("tsproj_ml_matplotlib")),
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_peaks_valleys(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    detail: pd.DataFrame,
    route: str,
    output_path: Path,
) -> Path:
    plt = _setup_matplotlib()
    frame = df[[time_col, target_col]].copy()
    frame[time_col] = pd.to_datetime(frame[time_col])
    frame = frame.sort_values(time_col)

    fig, ax = plt.subplots(figsize=(24, 8))
    ax.plot(frame[time_col], frame[target_col],
            color="#2F5597", linewidth=0.9, label=target_col)
    if not detail.empty:
        peak_df = detail[detail["type"] == "peak"]
        valley_df = detail[detail["type"] == "valley"]
        if not peak_df.empty:
            ax.scatter(pd.to_datetime(peak_df["time"]), peak_df["value"],
                       color="#C00000", s=30, marker="^", label="Peak", zorder=3)
        if not valley_df.empty:
            ax.scatter(pd.to_datetime(valley_df["time"]), valley_df["value"],
                       color="#00B050", s=30, marker="v", label="Valley", zorder=3)
    ax.set_title(f"{route} peaks & valleys "
                 f"(peaks={int((detail['type']=='peak').sum()) if not detail.empty else 0}, "
                 f"valleys={int((detail['type']=='valley').sum()) if not detail.empty else 0})")
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
def _output_paths(source_path: Path, output_dir: Optional[Path] = None) -> dict[str, Path]:
    stem = source_path.stem
    out_dir = output_dir or (source_path.parent / "peak_valley_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "csv": out_dir / f"{stem}_peaks_valleys.csv",
        "plot": out_dir / f"{stem}_peaks_valleys.png",
    }


def process_peaks_valleys(spec: PeakValleySpec) -> pd.DataFrame:
    source_path = spec.source_path
    route = spec.route or source_path.parent.name

    df = pd.read_csv(source_path)
    if spec.time_col not in df.columns or spec.target_col not in df.columns:
        raise ValueError(f"{source_path} must contain {spec.time_col!r} and {spec.target_col!r}.")

    detail = detect_peaks_valleys(df, spec.time_col, spec.target_col, spec)
    paths = _output_paths(source_path, spec.output_dir)

    detail.to_csv(paths["csv"], index=False, encoding="utf-8-sig")

    plot_path = None
    if spec.plot:
        plot_path = plot_peaks_valleys(
            df, spec.time_col, spec.target_col, detail, route, paths["plot"]
        )

    n_peaks = int((detail["type"] == "peak").sum()) if not detail.empty else 0
    n_valleys = int((detail["type"] == "valley").sum()) if not detail.empty else 0
    print(f"{route}: peaks={n_peaks}, valleys={n_valleys}, total={len(detail)}")
    print(f"  -> {paths['csv']}")
    if plot_path:
        print(f"  -> {plot_path}")

    return detail


# ---------------------------------------------------------------------------
# 配置加载与批量入口（与 data_aggregate / outlier_process 同模式）
# ---------------------------------------------------------------------------
def _load_config(config_path: str | Path) -> dict[str, Any]:
    import yaml

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Peak/valley config must be a mapping: {config_path}")
    return raw  # type: ignore[return-value]


def _resolve_path(p: str) -> Path:
    pr = Path(p).expanduser()
    return pr if pr.is_absolute() else (PROJECT_ROOT / pr).resolve()


def _build_spec(raw: dict[str, Any], config_path: Path) -> PeakValleySpec:
    missing = [k for k in ("source_path", "time_col", "target_col") if k not in raw]
    if missing:
        raise ValueError(f"Peak/valley config missing required fields {missing}: {config_path}")
    return PeakValleySpec(
        source_path=_resolve_path(raw["source_path"]),
        time_col=raw["time_col"],
        target_col=raw["target_col"],
        height=raw.get("height"),
        distance=int(raw.get("distance", 1)),
        prominence=raw.get("prominence"),
        width=raw.get("width"),
        top_n=raw.get("top_n"),
        plot=bool(raw.get("plot", True)),
        route=raw.get("route", ""),
        output_dir=_resolve_path(raw["output_dir"]) if raw.get("output_dir") else None,
    )


def run_peak_valley_detection(config_path: str | Path, *, force: bool = False) -> None:
    """加载 YAML 峰谷检测配置并执行。

    配置文件可写单个任务（顶层平铺）或多任务（顶层 tasks: 列表）。
    """
    config_path = Path(config_path).resolve()
    raw = _load_config(config_path)
    task_list = raw["tasks"] if "tasks" in raw else [raw]
    for item in task_list:
        spec = _build_spec(item, config_path)
        if force:
            for p in _output_paths(spec.source_path, spec.output_dir).values():
                p.unlink(missing_ok=True)
        process_peaks_valleys(spec)


def main() -> None:
    parser = argparse.ArgumentParser(description="时间序列峰谷检测与提取（配置驱动）")
    parser.add_argument("config", help="峰谷检测配置 YAML 路径")
    parser.add_argument("--force", action="store_true", help="删除旧输出强制重建")
    args = parser.parse_args()
    run_peak_valley_detection(args.config, force=args.force)


if __name__ == "__main__":
    main()
