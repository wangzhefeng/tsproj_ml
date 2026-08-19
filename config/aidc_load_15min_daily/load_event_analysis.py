# -*- coding: utf-8 -*-
"""AIDC 15min 负荷：数据分析 + 事件标签 + 特征工程（A/B 路）。

针对 dataset/aidc_load_15min_daily/ 下的 A/B 路 15min 负荷功率数据：
  1. 事件检测与打标（调 data_process/load_event_detection.py 共享核心，
     与 config/aidc_load_month/load_event_analysis.py 的日频标签同口径）：
       - shift_up/shift_down     持久水平阶跃（设备集中上架/下架）
       - stress_up/stress_down   1~21 天临时水平偏移（压测/临时批量操作）
       - burst_up/burst_down     1.25h~24h 日内持续冲击（压测冲击/突发负荷）
       - spike_up/spike_down     <=1h 功率突变（瞬时突变/切换操作）
  2. 特征工程：
       - feat_*  15min 本频率 trailing 窗口特征（无未来泄漏，可用于预测建模）
       - xf_*    跨频率统计特征：由日频 CSV（aidc_load_month/，1day 粒度）计算的
                 日级水平/趋势，统一 shift(1) 后按日期 merge 到每个 15min 点
                 （昨日口径——行 t 只携带截至 D-1 的日级信息，因果安全）
       - xr_*    跨 route 特征：对侧路负荷、双路总负荷、双路差
       - lbl_    事件标签（含居中窗口检测=有未来信息，仅供离线分析）+
                 lbl_prev_*（昨日口径的日级事件标记，同属标签性质）
  3. 输出（dataset/aidc_load_15min_daily/event_label_features/）：
       - <stem>_labeled_features.csv   逐点标签 + 全部特征
       - <stem>_events.csv             事件明细表
       - <stem>_events_overview.png    全序列事件标注总览图
       - load_event_analysis_report_15min.md  分析报告（两路合并）

用法（仓库根目录）：
    uv run python config/aidc_load_15min_daily/load_event_analysis.py
    uv run python config/aidc_load_15min_daily/load_event_analysis.py --routes A
    uv run python config/aidc_load_15min_daily/load_event_analysis.py --no-plot

注意：lbl_* 标签列由含居中窗口的检测算法产生，仅供离线分析/样本筛选/
评估使用；若用于预测训练请只取历史侧（或仅使用 feat_*/xf_*/xr_* 特征列）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_process.load_event_detection import (  # noqa: E402
    EventDetectionConfig,
    LoadEvent,
    classify_day_events,
    day_intraday_stats,
    detect_intraday_events,
    detect_short_excursions,
    events_to_frame,
    merge_day_events,
    project_events_to_points,
    suppress_boundary_artifacts,
    topdown_day_segments,
)

# ---------------------------------------------------------------------------
# 场景路径
# ---------------------------------------------------------------------------
DATA_15MIN_DIR = Path("dataset/aidc_load_15min_daily")
DATA_DAILY_DIR = Path("dataset/aidc_load_month")
OUTPUT_DIR = DATA_15MIN_DIR / "event_label_features"
ROUTES = {
    "A": {
        "f15": DATA_15MIN_DIR / "A_Loads_15min_mean_20251001_20260731.csv",
        "daily": DATA_DAILY_DIR / "A_Loads_1day_mean_20251001_20260731.csv",
    },
    "B": {
        "f15": DATA_15MIN_DIR / "B_Loads_15min_mean_20251001_20260731.csv",
        "daily": DATA_DAILY_DIR / "B_Loads_1day_mean_20251001_20260731.csv",
    },
}

KIND_PRIORITY = {
    "spike_up": 0, "spike_down": 0,
    "burst_up": 1, "burst_down": 1,
    "stress_up": 2, "stress_down": 2,
    "shift_up": 3, "shift_down": 3,
}


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_series(path: Path) -> pd.Series:
    df = pd.read_csv(PROJECT_ROOT / path, parse_dates=["time"])
    s = df.set_index("time")["value"].sort_index().astype(float)
    if s.index.has_duplicates:
        s = s.groupby(level=0).mean()
    return s.asfreq("15min")


def load_aggregate_meta(path: Path) -> dict:
    meta_path = path.with_name(path.name + ".aggregate.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


# ---------------------------------------------------------------------------
# 事件检测（与日频脚本同口径：日水平 = 15min 聚合的逐日中位数）
# ---------------------------------------------------------------------------
def detect_route_events(s15: pd.Series, cfg: EventDetectionConfig) -> tuple[list[LoadEvent], list[LoadEvent], list[LoadEvent]]:
    """返回 (merged_day_events, intraday_events, all_events)。"""
    day_stats = day_intraday_stats(s15)
    day_level = day_stats["intraday_median"].where(day_stats["intraday_n_points"] >= 90).dropna()

    segments = topdown_day_segments(day_level, cfg)
    seg_events = classify_day_events(segments, cfg)
    short_events = detect_short_excursions(day_level, cfg)
    day_events = merge_day_events(seg_events, short_events)

    intraday = detect_intraday_events(s15, cfg)
    intraday = suppress_boundary_artifacts(intraday, day_events, cfg)

    all_events = sorted(day_events + intraday, key=lambda e: (e.start, e.end))
    return day_events, intraday, all_events


# ---------------------------------------------------------------------------
# 15min 本频率特征（全部 trailing 窗口，无未来泄漏）
# ---------------------------------------------------------------------------
def build_intra_features(s: pd.Series) -> pd.DataFrame:
    idx = s.index
    feat = pd.DataFrame(index=idx)
    tod = idx.hour * 60 + idx.minute
    feat["feat_hour"] = idx.hour
    feat["feat_minute_of_day"] = tod
    feat["feat_tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    feat["feat_tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    feat["feat_dow"] = idx.dayofweek
    feat["feat_is_weekend"] = (idx.dayofweek >= 5).astype(int)
    feat["feat_month"] = idx.month

    for w, tag in ((4, "1h"), (16, "4h"), (96, "24h"), (672, "7d")):
        roll = s.rolling(w, min_periods=max(4, w // 4))
        feat[f"feat_roll_{tag}_mean"] = roll.mean()
        feat[f"feat_roll_{tag}_std"] = roll.std()
        feat[f"feat_roll_{tag}_min"] = roll.min()
        feat[f"feat_roll_{tag}_max"] = roll.max()

    for lag, tag in ((1, "1"), (4, "4"), (96, "96"), (672, "672")):
        feat[f"feat_diff_{tag}"] = s - s.shift(lag)
    feat["feat_diff_96_pct"] = s.pct_change(96).replace([np.inf, -np.inf], np.nan) * 100

    med7d = s.rolling(672, min_periods=96).median()
    mad7d = (s - med7d).abs().rolling(672, min_periods=96).median()
    feat["feat_robust_z_7d"] = (s - med7d) / (1.4826 * mad7d).clip(lower=50.0)

    # 周内同时刻基线（同 dow+slot 的过去 7 次中位数）
    slot_key = pd.Series(list(zip(idx.dayofweek, idx.time.astype(str))), index=idx)
    grouped = pd.DataFrame({"v": s.values, "key": slot_key.values}, index=idx).groupby("key")["v"]
    weekly_base = grouped.transform(lambda x: x.shift(1).rolling(7, min_periods=3).median())
    feat["feat_weekly_base_dev"] = s - weekly_base
    feat["feat_weekly_base_dev_pct"] = feat["feat_weekly_base_dev"] / weekly_base * 100

    # 当日累计（日内 trailing，无泄漏）
    day_key = idx.normalize()
    feat["feat_day_cum_mean"] = s.groupby(day_key).expanding().mean().reset_index(level=0, drop=True).reindex(idx)
    feat["feat_day_cum_max"] = s.groupby(day_key).expanding().max().reset_index(level=0, drop=True).reindex(idx)
    feat["feat_day_cum_min"] = s.groupby(day_key).expanding().min().reset_index(level=0, drop=True).reindex(idx)
    return feat


# ---------------------------------------------------------------------------
# 跨频率特征：由日频 CSV 计算的日级特征（merge 到 15min 点）
# ---------------------------------------------------------------------------
def build_cross_freq_day_features(daily_path: Path, day_events: list[LoadEvent]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """日级统计特征（昨日口径）+ 日级事件标记（标签性质，单独返回）。

    统计特征统一 shift(1)：D 日的 15min 点拿到的是 D-1 日的日级统计，
    消除「当天全天统计提前到当天 00:00 可见」的日内未来信息。
    事件标记源自居中窗口检测（含未来信息），属 lbl_ 家族，不入 xf_。
    返回 (day_stats_shifted, day_event_flags)，均为日频索引。
    """
    daily = load_series_as_daily(daily_path)
    day = pd.DataFrame(index=daily.index)
    day["xf_day_value"] = daily
    day["xf_day_diff1"] = daily.diff()
    day["xf_day_diff1_pct"] = daily.pct_change() * 100
    day["xf_day_rol7_mean"] = daily.rolling(7, min_periods=4).mean()
    day["xf_day_rol30_mean"] = daily.rolling(30, min_periods=14).mean()
    day["xf_day_rol7_std"] = daily.rolling(7, min_periods=4).std()
    med30 = daily.rolling(30, min_periods=14).median().shift(1)
    mad30 = (daily - med30).abs().rolling(30, min_periods=14).median().shift(1)
    day["xf_day_z30_robust"] = (daily - med30) / (1.4826 * mad30).clip(lower=30.0)

    # 30 天线性趋势斜率（kW/天，trailing）
    slope = {}
    vals = daily.to_numpy()
    for i in range(len(vals)):
        lo = max(0, i - 29)
        y = vals[lo:i + 1]
        x = np.arange(len(y))
        if len(y) >= 10 and not np.isnan(y).any():
            slope[daily.index[i]] = float(np.polyfit(x, y, 1)[0])
        else:
            slope[daily.index[i]] = np.nan
    day["xf_day_slope30"] = pd.Series(slope)

    # 统一昨日口径：D 日的行携带 D-1 日的统计（首日为 NaN）
    day_stats = day.shift(1)

    # 日级事件标记（来自共享检测核心，与日频脚本一致；含居中窗口=标签性质）
    ev_flags = pd.DataFrame(index=daily.index)
    ev_flags["lbl_prev_day_event_day"] = 0
    ev_flags["lbl_prev_day_shift_day"] = 0
    ev_flags["lbl_prev_day_stress_day"] = 0
    ev_flags["lbl_prev_day_event_type"] = "none"
    for ev in day_events:
        if ev.start not in ev_flags.index:
            continue
        if ev.kind.startswith("shift"):
            ev_flags.loc[ev.start, "lbl_prev_day_shift_day"] = 1
            ev_flags.loc[ev.start, "lbl_prev_day_event_day"] = 1
            ev_flags.loc[ev.start, "lbl_prev_day_event_type"] = ev.kind
        else:
            for d in pd.date_range(ev.start, ev.end, freq="1D"):
                if d in ev_flags.index:
                    ev_flags.loc[d, "lbl_prev_day_stress_day"] = 1
                    ev_flags.loc[d, "lbl_prev_day_event_day"] = 1
                    ev_flags.loc[d, "lbl_prev_day_event_type"] = ev.kind
    # 同样昨日口径（与统计特征一致，供「截至昨日的事件状态」类使用）
    ev_flags = ev_flags.shift(1)
    flag_cols = ["lbl_prev_day_event_day", "lbl_prev_day_shift_day", "lbl_prev_day_stress_day"]
    ev_flags[flag_cols] = ev_flags[flag_cols].fillna(0).astype(int)
    ev_flags["lbl_prev_day_event_type"] = ev_flags["lbl_prev_day_event_type"].fillna("none")
    return day_stats, ev_flags


def load_series_as_daily(path: Path) -> pd.Series:
    df = pd.read_csv(PROJECT_ROOT / path, parse_dates=["time"])
    s = df.set_index("time")["value"].sort_index().astype(float)
    if s.index.has_duplicates:
        s = s.groupby(level=0).mean()
    return s.asfreq("1D")


# ---------------------------------------------------------------------------
# 事件 -> 逐点标签（含 event_id / 距事件距离）
# ---------------------------------------------------------------------------
def build_point_labels(s15: pd.Series, day_events: list[LoadEvent],
                       intraday_events: list[LoadEvent],
                       all_events: list[LoadEvent]) -> pd.DataFrame:
    labels = project_events_to_points(s15.index, day_events, intraday_events)

    # 每个 15min 点归属的事件（优先级最高者），以及事件进行中/距离信息
    n = len(s15)
    event_id = np.full(n, -1, dtype=int)
    event_src = np.full(n, "none", dtype=object)
    days_into = np.full(n, np.nan)
    idx = s15.index

    event_spans = []  # (start_ts, end_ts, kind, source, id)
    for eid, ev in enumerate(all_events, start=1):
        if ev.kind.startswith("shift"):
            # 与 project_events_to_points 的过渡带一致：起始日前 1 天 ~ 后 1 天
            span_start = (ev.start - pd.Timedelta(days=1)).normalize()
            span_end = (ev.start + pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=23, minutes=45)
        else:
            span_start = ev.start
            span_end = ev.end if ev.source == "intraday" else ev.end.normalize() + pd.Timedelta(hours=23, minutes=45)
        event_spans.append((span_start, span_end, ev.kind, ev.source, eid))

    for i, t in enumerate(idx):
        best = None
        for span_start, span_end, kind, source, eid in event_spans:
            if span_start <= t <= span_end:
                if best is None or KIND_PRIORITY[kind] < KIND_PRIORITY[best[2]]:
                    best = (span_start, span_end, kind, source, eid)
        if best is not None:
            event_id[i] = best[4]
            event_src[i] = best[3]
            days_into[i] = (t - best[0]).total_seconds() / 86400.0

    labels["lbl_event_id"] = event_id
    labels["lbl_event_src"] = event_src
    labels["lbl_days_into_event"] = np.round(days_into, 4)

    # 距上一个事件结束的天数
    ends = np.array([ev.end for ev in all_events], dtype="datetime64[ns]")
    days_since = np.full(n, np.nan)
    for i, t in enumerate(idx):
        t64 = np.datetime64(t)
        prev = ends[ends < t64]
        if len(prev):
            days_since[i] = (t64 - prev.max()) / np.timedelta64(1, "D")
    labels["lbl_days_since_event"] = np.round(days_since, 3)
    return labels


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
def _setup_matplotlib():
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tsproj_ml_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


KIND_COLORS = {
    "shift_up": "#C0392B", "shift_down": "#2471A3",
    "stress_up": "#E67E22", "stress_down": "#8E44AD",
    "burst_up": "#D4AC0D", "burst_down": "#16A085",
    "spike_up": "#111111", "spike_down": "#7F8C8D",
}


def plot_route(s15: pd.Series, day_events, intraday_events, out_png: Path, route: str) -> None:
    plt = _setup_matplotlib()
    day_stats = day_intraday_stats(s15)
    fig, axes = plt.subplots(2, 1, figsize=(26, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(s15.index, s15.values, color="#2F5597", linewidth=0.35, label="15min load")
    ax.plot(day_stats.index, day_stats["intraday_median"], color="#C00000",
            linewidth=1.4, label="daily median")

    for ev in day_events:
        color = KIND_COLORS.get(ev.kind, "#AAAAAA")
        if ev.kind.startswith("stress"):
            ax.axvspan(ev.start, ev.end + pd.Timedelta(days=1), color=color, alpha=0.14)
        else:  # shift：起点竖线 + 标注幅度
            ax.axvline(ev.start, color=color, linestyle="--", linewidth=1.2)
            ax.annotate(f"{ev.kind}\n{ev.amplitude:+.0f}kW",
                        xy=(ev.start, ev.base_level + ev.amplitude),
                        fontsize=7, color=color,
                        xytext=(5, 0), textcoords="offset points")
    mid_points = [(ev.start + (ev.end - ev.start) / 2, ev.kind) for ev in intraday_events]
    kinds_seen = {k for _, k in mid_points}
    for kind in sorted(kinds_seen):
        pts = [m for m, k in mid_points if k == kind]
        vals = [s15.asof(p) for p in pts]
        ax.scatter(pts, vals, marker="x", s=46, color=KIND_COLORS.get(kind, "#000000"),
                   label=f"intraday {kind} ({len(pts)})", zorder=5)
    ax.set_title(f"AIDC {route} 15min load event overview "
                 f"(day-level events={len(day_events)}, intraday events={len(intraday_events)})")
    ax.set_ylabel("load (kW)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=3)

    ax2 = axes[1]
    std = day_stats["intraday_std"]
    ax2.plot(std.index, std, color="#5B6770", linewidth=0.9, label="intraday std")
    base = std.shift(1).rolling(60, min_periods=14).quantile(0.95)
    ax2.plot(base.index, base, color="#C00000", linewidth=1.1, linestyle="--",
             label="rolling p95 (60d)")
    ax2.set_ylabel("intraday std")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8)

    fig.autofmt_xdate()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 单路处理
# ---------------------------------------------------------------------------
def process_route(route: str, cfg: EventDetectionConfig, peer_series: pd.Series,
                  make_plot: bool) -> dict:
    paths = ROUTES[route]
    s15 = load_series(paths["f15"])
    meta = load_aggregate_meta(paths["f15"])

    day_events, intraday_events, all_events = detect_route_events(s15, cfg)

    feat = build_intra_features(s15)
    day_stats, day_ev_flags = build_cross_freq_day_features(paths["daily"], day_events)
    labels = build_point_labels(s15, day_events, intraday_events, all_events)

    out = pd.DataFrame({"time": s15.index, "value": s15.values}).set_index("time")
    out = out.join(feat)
    day_cols = day_stats.reindex(out.index.normalize())
    day_cols.index = out.index
    out = out.join(day_cols)
    ev_cols = day_ev_flags.reindex(out.index.normalize())
    ev_cols.index = out.index
    out = out.join(ev_cols)
    out = out.join(labels)

    # 跨 route 特征
    aligned_peer = peer_series.reindex(out.index)
    out["xr_peer_value"] = aligned_peer
    out["xr_total_load"] = out["value"] + aligned_peer
    out["xr_route_diff"] = out["value"] - aligned_peer
    out["xr_route_diff_pct"] = out["xr_route_diff"] / aligned_peer * 100

    stem = paths["f15"].stem
    out_dir = PROJECT_ROOT / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    features_csv = out_dir / f"{stem}_labeled_features.csv"
    events_csv = out_dir / f"{stem}_events.csv"
    out.reset_index().to_csv(features_csv, index=False, encoding="utf-8-sig")
    events_frame = events_to_frame(all_events)
    events_frame.to_csv(events_csv, index=False, encoding="utf-8-sig")

    png = None
    if make_plot:
        png = out_dir / f"{stem}_events_overview.png"
        plot_route(s15, day_events, intraday_events, png, route)

    label_cols = [c for c in out.columns if c.startswith("lbl_")]
    kind_counts = events_frame["kind"].value_counts().to_dict() if not events_frame.empty else {}
    return {
        "route": route,
        "series": s15,
        "features": out,
        "events": events_frame,
        "day_events": day_events,
        "intraday_events": intraday_events,
        "meta": meta,
        "paths": {"features": features_csv, "events": events_csv, "png": png},
        "label_coverage": float(out["lbl_event"].mean()),
        "label_cols": label_cols,
        "kind_counts": kind_counts,
    }


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def write_report(results: list[dict], cfg: EventDetectionConfig, out_md: Path) -> None:
    lines = ["# AIDC 15min 负荷事件标签与特征工程报告", ""]
    lines.append(f"- 生成命令：`uv run python config/aidc_load_15min_daily/load_event_analysis.py`")
    lines.append(f"- 检测核心：`data_process/load_event_detection.py`（与日频脚本共享同一口径）")
    lines.append("")
    lines.append("## 1. 事件分类学")
    lines.append("")
    lines.append("| 标签 | 含义 | 业务解释 | 判定方式 |")
    lines.append("|---|---|---|---|")
    lines.append("| shift_up / shift_down | 持久水平阶跃 | 设备集中上架/下架 | 分段水平差 ≥ max(300kW, 2%) 且后续不回落 |")
    lines.append("| stress_up / stress_down | 1~21 天临时水平偏移 | 压力测试 / 临时批量操作 / 检修迁移 | 阶跃后在 21 天内回落到基线 55% 以内 |")
    lines.append("| burst_up / burst_down | 1.25h~24h 日内持续冲击 | 压测冲击 / 突发负荷块 | 15min 残差 > max(6σ, 350kW) 连续 5~96 点 |")
    lines.append("| spike_up / spike_down | ≤1h 功率突变 | 瞬时功率突变 / 切换操作 | 15min 残差异常连续 ≤4 点 |")
    lines.append("")
    lines.append(f"阈值配置：seg=({cfg.seg_min_kw}kW/{cfg.seg_min_pct:.0%}), "
                 f"short=({cfg.short_min_kw}kW/{cfg.short_min_pct:.0%}), "
                 f"intraday=(k={cfg.intraday_k_sigma}σ, {cfg.intraday_min_kw}kW)；"
                 f"日级事件的基线水平取 15min 聚合的逐日中位数（只统计 ≥90 点的完整日）。")
    lines.append("")

    for res in results:
        route = res["route"]
        meta = res["meta"]
        s = res["series"]
        lines.append(f"## 2.{results.index(res) + 1} {route} 路数据概况")
        lines.append("")
        lines.append(f"- 样本数：{len(s)}（{s.index.min()} ~ {s.index.max()}，15min）")
        if meta:
            lines.append(f"- 来源：{meta.get('config', {}).get('source_freq')} -> "
                         f"{meta.get('config', {}).get('target_freq')} "
                         f"{meta.get('config', {}).get('method')} 聚合，线性填补 "
                         f"{meta.get('filled_value_count')} 点（{meta.get('gap_segment_count')} 段缺失，"
                         f"最长 {meta.get('max_gap_length')} 点）")
        lines.append(f"- 负荷区间：{s.min():.0f} ~ {s.max():.0f} kW，中位数 {s.median():.0f} kW")
        lines.append(f"- 事件总数：{len(res['events'])}（日级 {len(res['day_events'])}，"
                     f"日内 {len(res['intraday_events'])}）；"
                     f"逐点标签覆盖率 {res['label_coverage']:.2%}")
        lines.append("")
        if not res["events"].empty:
            lines.append("### 事件明细")
            lines.append("")
            lines.append("| event_id | 开始 | 结束 | 类型 | 来源 | 基线(kW) | 幅度(kW) | 幅度(%) | 持续(天) |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for _, row in res["events"].iterrows():
                lines.append(
                    f"| {int(row['event_id'])} | {row['event_start']} | {row['event_end']} | "
                    f"{row['kind']} | {row['source']} | {row['base_level_kw']:.0f} | "
                    f"{row['amplitude_kw']:+.0f} | {row['amplitude_pct']:+.1f} | {row['duration']} |")
            lines.append("")

    lines.append("## 3. 特征字典")
    lines.append("")
    lines.append("| 前缀 | 内容 | 示例 |")
    lines.append("|---|---|---|")
    lines.append("| feat_ | 15min 本频率 trailing 特征：日历（hour/dow/month/tod_sin/cos）、"
                 "滚动统计（1h/4h/24h/7d 的 mean/std/min/max）、差分（Δ1/Δ4/Δ96/Δ672 及 Δ96 百分比）、"
                 "7d robust z、周内同时刻基线偏离、当日累计 mean/max/min | feat_roll_24h_mean |")
    lines.append("| xf_ | 跨频率统计特征（来自 aidc_load_month 日频 CSV，统一昨日口径 shift(1)）："
                 "昨日日均值、昨日日环比、截至昨日的 7/30 天滚动、30 天 robust z、"
                 "30 天趋势斜率 | xf_day_z30_robust |")
    lines.append("| xr_ | 跨 route 特征：对侧路负荷、双路总负荷、双路差及占比 | xr_total_load |")
    lines.append("| lbl_ | 事件标签（检测含居中窗口，仅供分析/筛选，不作在线预测特征）：8 类事件 0/1 列、"
                 "lbl_event、lbl_event_type、lbl_event_id、lbl_event_src、lbl_days_into_event、lbl_days_since_event；"
                 "另含 lbl_prev_*（昨日口径日级事件标记，同属标签性质） | lbl_stress_up |")
    lines.append("")
    lines.append("## 4. 使用注意")
    lines.append("")
    lines.append("- lbl_* 标签由含未来信息的居中窗口检测产生，用于离线分析、样本筛选、事件期评估；")
    lines.append("  lbl_prev_*（昨日口径事件标记）虽已 shift(1)，其检测仍含居中窗口，同属标签性质。")
    lines.append("  feat_* 为 trailing 窗口统计、xf_* 为昨日口径日级统计、xr_* 为对侧路同时刻值，"
                 "三者在时序因果上均无未来泄漏，可直接用于预测建模")
    lines.append("  （喂入框架 custom_features 外生通路时注意：嵌入 value_t/peer_t 的列需先 lag，"
                 "避免同行监督泄漏——原点 t 的特征只能用于预测 t+h，h>=1）。")
    lines.append("- 15min 与日频（config/aidc_load_month/load_event_analysis.py）的日级事件"
                 "由同一检测核心、同一日水平序列（15min 逐日中位数）产生，标签跨频率一致。")
    out_md.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="AIDC 15min 负荷事件标签与特征工程")
    parser.add_argument("--routes", nargs="+", default=["A", "B"], choices=sorted(ROUTES))
    parser.add_argument("--no-plot", action="store_true", help="跳过总览图生成")
    args = parser.parse_args()

    cfg = EventDetectionConfig()
    series_map = {r: load_series(ROUTES[r]["f15"]) for r in args.routes}

    results = []
    for route in args.routes:
        peer_route = "B" if route == "A" else "A"
        peer = series_map.get(peer_route)
        if peer is None:
            peer = load_series(ROUTES[peer_route]["f15"])
        res = process_route(route, cfg, peer, make_plot=not args.no_plot)
        results.append(res)
        print(f"[{route}] rows={len(res['features'])}, events={len(res['events'])} "
              f"(day={len(res['day_events'])}, intraday={len(res['intraday_events'])}), "
              f"label_coverage={res['label_coverage']:.2%}")
        print(f"  -> {res['paths']['features']}")
        print(f"  -> {res['paths']['events']}")
        if res["paths"]["png"]:
            print(f"  -> {res['paths']['png']}")

    out_md = PROJECT_ROOT / OUTPUT_DIR / "load_event_analysis_report_15min.md"
    write_report(results, cfg, out_md)
    print(f"  -> {out_md}")


if __name__ == "__main__":
    main()
