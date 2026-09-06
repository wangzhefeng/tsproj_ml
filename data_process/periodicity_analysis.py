# -*- coding: utf-8 -*-
"""时间序列周期自动检测工具（配置驱动）。

检测算法核心位于 timeseries_analysis/periods.py（FFT/ACF/STL 纯数值实现）。
本模块承载 DataFrame 整理、时间换算、规格、可视化、报告落盘、
协整诊断与配置驱动 CLI。

输出（源文件同级的 periodicity_analysis/ 子目录，文件名从源文件名派生）：
  <stem>_periodicity_report.csv  结构化报告（指标名/值/说明）
  <stem>_acf_plot.png            自相关函数图（前 max_lags 步）
  <stem>_fft_plot.png            FFT 幅度谱图（正频段）

用法（仓库根目录）：
    env -u PYTHONPATH .venv/bin/python data_process/periodicity_analysis.py <config.yaml>
    env -u PYTHONPATH .venv/bin/python data_process/periodicity_analysis.py <config.yaml> --force

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
        max_lags: 2000           # ACF 最大滞后步数，默认 2000
        seasonal_period: null    # STL 季节周期（样本数）；null=用 ACF 检测周期
        top_n_periods: 3         # 报告输出前 N 个 ACF 周期候选，默认 3
        min_acf: 0.1             # ACF 周期候选最小相关值，默认 0.1（过滤噪声假峰）
        plot: true               # 是否生成 ACF/FFT 图，默认 true
        fft_top_k: 1             # 额外报告 FFT 前 K 个主导周期（>1 时生效），默认 1
        coint_col: null          # 可选：与目标列做 Engle-Granger 协整检验的列名
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from statsmodels.tsa.stattools import acf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from timeseries_analysis.periods import (  # noqa: E402
    DEFAULT_MAX_LAGS,
    DEFAULT_MIN_ACF,
    DEFAULT_TOP_N_PERIODS,
    fft_dominant_period,
    fft_top_periods,
    acf_periods,
    stl_strength,
)


def detect_periodicity(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    *,
    max_lags: int = DEFAULT_MAX_LAGS,
    seasonal_period: Optional[int] = None,
    top_n_periods: int = DEFAULT_TOP_N_PERIODS,
    min_acf: float = DEFAULT_MIN_ACF,
    fft_top_k: int = 1,
) -> dict:
    """对 DataFrame 执行周期检测，返回结构化报告字典。"""
    frame = df[[time_col, target_col]].copy()
    frame[time_col] = pd.to_datetime(frame[time_col])
    frame = frame.sort_values(time_col)
    y = pd.to_numeric(frame[target_col], errors="coerce").dropna().to_numpy()
    if len(y) < 4:
        raise ValueError(f"目标列有效样本不足（<4）：{target_col}")

    # 采样间隔（秒）
    ts = frame[time_col].to_numpy()
    diff_seconds = float(np.median(np.diff(ts.astype("datetime64[ns]")).astype("int64")) / 1e9) if len(ts) > 1 else None

    # 线性去趋势：强趋势会淹没周期信号（ACF 单调下降无局部峰、FFT 主导=序列长度级）
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_detrended = y - (slope * x + intercept)

    report: dict[str, Any] = {
        "n_samples": int(len(y)),
        "sample_interval_seconds": diff_seconds,
    }

    # 1. FFT 主导周期（在去趋势序列上）
    fft_info = fft_dominant_period(y_detrended)
    report["fft_dominant_period_samples"] = fft_info["dominant_period_samples"]
    report["fft_dominant_amplitude"] = fft_info["dominant_amplitude"]
    if fft_info["dominant_period_samples"] and diff_seconds:
        report["fft_dominant_period_seconds"] = fft_info["dominant_period_samples"] * diff_seconds
        report["fft_dominant_period_days"] = fft_info["dominant_period_samples"] * diff_seconds / 86400.0
    if fft_top_k > 1:
        report["fft_top_periods"] = [
            {"period_samples": period, "frequency": freq, "amplitude": amp}
            for period, freq, amp in fft_top_periods(y_detrended, fft_top_k)
        ]

    # 2. ACF 周期候选（在去趋势序列上）
    acf_result = acf_periods(y_detrended, max_lags, top_n_periods, min_acf)
    report["acf_periods"] = [
        {"lag": lag, "acf": value} for lag, value in acf_result
    ]
    report["acf_dominant_period_samples"] = acf_result[0][0] if acf_result else None
    if acf_result and diff_seconds:
        report["acf_dominant_period_days"] = acf_result[0][0] * diff_seconds / 86400.0

    # 3. STL 季节性成分（可选）
    period = seasonal_period if seasonal_period is not None else (acf_result[0][0] if acf_result else None)
    report.update(stl_strength(y, period))
    return report


# ---------------------------------------------------------------------------
# 规格 dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PeriodicitySpec:
    source_path: Path
    time_col: str
    target_col: str
    max_lags: int = DEFAULT_MAX_LAGS
    seasonal_period: Optional[int] = None
    top_n_periods: int = DEFAULT_TOP_N_PERIODS
    min_acf: float = DEFAULT_MIN_ACF
    fft_top_k: int = 1
    coint_col: Optional[str] = None
    plot: bool = True
    route: str = ""
    output_dir: Optional[Path] = None


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


def plot_acf(y: np.ndarray, max_lags: int, output_path: Path) -> Path:
    plt = _setup_matplotlib()
    nlags = min(max_lags, len(y) - 1)
    values = acf(y, nlags=nlags, fft=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.vlines(range(nlags + 1), 0, values)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF")
    ax.set_title("Autocorrelation Function")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_fft_spectrum(y: np.ndarray, output_path: Path) -> Path:
    plt = _setup_matplotlib()
    n = len(y)
    y_detrend = y - np.mean(y)
    spectrum = np.abs(fft(y_detrend))
    freqs = fftfreq(n)
    mask = freqs > 0
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(freqs[mask], spectrum[mask], linewidth=0.8)
    ax.set_xlabel("frequency (cycles/sample)")
    ax.set_ylabel("magnitude")
    ax.set_title("FFT Magnitude Spectrum")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 单任务执行
# ---------------------------------------------------------------------------
def _engle_granger_coint(
    df: pd.DataFrame, time_col: str, target_col: str, coint_col: str
) -> dict[str, Any]:
    """Engle-Granger 协整检验（目标列 vs 对照列，按时间排序后成对完整样本）。"""
    from statsmodels.tsa.stattools import coint

    if coint_col not in df.columns:
        raise ValueError(f"缺少协整对照列 {coint_col!r}")
    frame: pd.DataFrame = df.loc[:, [time_col, target_col, coint_col]].copy()
    frame[time_col] = pd.to_datetime(frame[time_col])
    frame = frame.sort_values(time_col).reset_index(drop=True)
    pair: pd.DataFrame = frame.loc[:, [target_col, coint_col]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    y = np.asarray(pair[target_col], dtype=float)
    x = np.asarray(pair[coint_col], dtype=float)
    if len(y) < 20 or not (np.isfinite(y).all() and np.isfinite(x).all()):
        raise ValueError(f"协整检验有效样本不足（<20）或含非有限值：{target_col} vs {coint_col}")
    statistic, pvalue, _ = coint(y, x)
    return {
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "verdict": "协整" if pvalue < 0.05 else "不协整",
    }


def _output_paths(source_path: Path, output_dir: Optional[Path] = None) -> dict[str, Path]:
    stem = source_path.stem
    out_dir = output_dir or (source_path.parent / "periodicity_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "report": out_dir / f"{stem}_periodicity_report.csv",
        "acf_plot": out_dir / f"{stem}_acf_plot.png",
        "fft_plot": out_dir / f"{stem}_fft_plot.png",
    }


def process_periodicity(spec: PeriodicitySpec) -> dict[str, Any]:
    source_path = spec.source_path
    route = spec.route or source_path.parent.name

    df = pd.read_csv(source_path)
    if spec.time_col not in df.columns or spec.target_col not in df.columns:
        raise ValueError(f"{source_path} must contain {spec.time_col!r} and {spec.target_col!r}.")

    report = detect_periodicity(
        df,
        spec.time_col,
        spec.target_col,
        max_lags=spec.max_lags,
        seasonal_period=spec.seasonal_period,
        top_n_periods=spec.top_n_periods,
        min_acf=spec.min_acf,
        fft_top_k=spec.fft_top_k,
    )
    paths = _output_paths(source_path, spec.output_dir)

    # 写结构化报告 CSV：指标名/值/说明
    rows = [
        {"metric": "n_samples", "value": report["n_samples"], "description": "有效样本数"},
        {"metric": "sample_interval_seconds", "value": report["sample_interval_seconds"], "description": "采样间隔（秒）"},
        {"metric": "fft_dominant_period_samples", "value": report["fft_dominant_period_samples"], "description": "FFT 主导周期（样本数）"},
        {"metric": "fft_dominant_amplitude", "value": report["fft_dominant_amplitude"], "description": "FFT 主导幅度"},
        {"metric": "fft_dominant_period_days", "value": report.get("fft_dominant_period_days"), "description": "FFT 主导周期（天）"},
        {"metric": "acf_dominant_period_samples", "value": report["acf_dominant_period_samples"], "description": "ACF 主导周期（样本数）"},
        {"metric": "acf_dominant_period_days", "value": report.get("acf_dominant_period_days"), "description": "ACF 主导周期（天）"},
        {"metric": "acf_period_candidates", "value": json.dumps(report["acf_periods"], ensure_ascii=False), "description": "ACF 周期候选（lag+acf，按相关值降序）"},
        {"metric": "stl_seasonal_period_used", "value": report["stl_seasonal_period_used"], "description": "STL 使用的季节周期（样本数）"},
        {"metric": "stl_has_seasonal_component", "value": report["stl_has_seasonal_component"], "description": "STL 是否产出季节成分"},
        {"metric": "stl_seasonal_std", "value": report.get("stl_seasonal_std"), "description": "季节成分标准差"},
        {"metric": "stl_residual_std", "value": report.get("stl_residual_std"), "description": "残差成分标准差"},
        {"metric": "stl_seasonal_ratio", "value": report.get("stl_seasonal_ratio"), "description": "季节/残差标准差比（越大季节性越强）"},
    ]
    if report.get("fft_top_periods"):
        rows.append({
            "metric": "fft_top_periods",
            "value": json.dumps(report["fft_top_periods"], ensure_ascii=False),
            "description": f"FFT 前 {len(report['fft_top_periods'])} 个主导周期（period_samples/frequency/amplitude，按幅度降序）",
        })
    if spec.coint_col:
        coint_result = _engle_granger_coint(df, spec.time_col, spec.target_col, spec.coint_col)
        rows.extend([
            {"metric": "coint_col", "value": spec.coint_col, "description": "Engle-Granger 协整检验对照列"},
            {"metric": "coint_statistic", "value": coint_result["statistic"], "description": "协整检验 t 统计量"},
            {"metric": "coint_pvalue", "value": coint_result["pvalue"], "description": "协整检验 p 值（<0.05 判协整）"},
            {"metric": "coint_verdict", "value": coint_result["verdict"], "description": "5% 显著性下的协整判定"},
        ])
    report_df = pd.DataFrame(rows)
    report_df.to_csv(paths["report"], index=False, encoding="utf-8-sig")

    # 写图
    if spec.plot:
        frame = df[[spec.time_col, spec.target_col]].copy()
        frame[spec.time_col] = pd.to_datetime(frame[spec.time_col])
        frame = frame.sort_values(spec.time_col)
        y = pd.to_numeric(frame[spec.target_col], errors="coerce").dropna().to_numpy()
        plot_acf(y, spec.max_lags, paths["acf_plot"])
        plot_fft_spectrum(y, paths["fft_plot"])

    # 打印摘要
    fft_p = report["fft_dominant_period_samples"]
    acf_p = report["acf_dominant_period_samples"]
    print(f"{route}: n={report['n_samples']}, FFT period={fft_p}, ACF period={acf_p}, "
          f"STL seasonal={'yes' if report['stl_has_seasonal_component'] else 'no'}")
    print(f"  -> {paths['report']}")
    if spec.plot:
        print(f"  -> {paths['acf_plot']}")
        print(f"  -> {paths['fft_plot']}")

    return report


# ---------------------------------------------------------------------------
# 配置加载与批量入口（与 data_aggregate / outlier_process 同模式）
# ---------------------------------------------------------------------------
def _load_config(config_path: str | Path) -> dict[str, Any]:
    import yaml

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Periodicity config must be a mapping: {config_path}")
    return raw  # type: ignore[return-value]


def _resolve_path(p: str) -> Path:
    pr = Path(p).expanduser()
    return pr if pr.is_absolute() else (PROJECT_ROOT / pr).resolve()


def _build_spec(raw: dict[str, Any], config_path: Path) -> PeriodicitySpec:
    missing = [k for k in ("source_path", "time_col", "target_col") if k not in raw]
    if missing:
        raise ValueError(f"Periodicity config missing required fields {missing}: {config_path}")
    return PeriodicitySpec(
        source_path=_resolve_path(raw["source_path"]),
        time_col=raw["time_col"],
        target_col=raw["target_col"],
        max_lags=int(raw.get("max_lags", DEFAULT_MAX_LAGS)),
        seasonal_period=raw.get("seasonal_period"),
        top_n_periods=int(raw.get("top_n_periods", DEFAULT_TOP_N_PERIODS)),
        min_acf=float(raw.get("min_acf", DEFAULT_MIN_ACF)),
        fft_top_k=int(raw.get("fft_top_k", 1)),
        coint_col=raw.get("coint_col"),
        plot=bool(raw.get("plot", True)),
        route=raw.get("route", ""),
        output_dir=_resolve_path(raw["output_dir"]) if raw.get("output_dir") else None,
    )


def run_periodicity_analysis(config_path: str | Path, *, force: bool = False) -> None:
    """加载 YAML 周期检测配置并执行。

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
        process_periodicity(spec)


def main() -> None:
    parser = argparse.ArgumentParser(description="时间序列周期自动检测（配置驱动）")
    parser.add_argument("config", help="周期检测配置 YAML 路径")
    parser.add_argument("--force", action="store_true", help="删除旧输出强制重建")
    args = parser.parse_args()
    run_periodicity_analysis(args.config, force=args.force)


if __name__ == "__main__":
    main()
