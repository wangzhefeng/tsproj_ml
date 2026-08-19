# -*- coding: utf-8 -*-
"""AIDC 日频负荷：数据分析 + 事件标签 + 特征工程（A/B 路）。

针对 dataset/aidc_load_month/ 下的 A/B 路日频负荷功率数据
（文件名 1day_mean，即 1day 粒度；目录名 aidc_load_month 沿用场景命名）：
  1. 事件检测与打标（调 data_process/load_event_detection.py 共享核心，
     与 config/aidc_load_15min_daily/load_event_analysis.py 同口径；
     日水平基线取 15min 数据聚合的逐日中位数，保证两频率标签一致）：
       - shift_up/shift_down_day     持久水平阶跃日（设备集中上架/下架）
       - stress_up/stress_down_day   临时水平偏移日（压测/临时批量操作）
       - burst_up/burst_down_day     日内持续冲击日（压测冲击/突发负荷）
       - spike_up/spike_down_day     功率突变日（瞬时突变/切换操作）
       - volatile_day                日内波动异常日（15min std 超 60 天滚动 p95）
  2. 特征工程：
       - feat_*  日频本频率 trailing 窗口特征（无未来泄漏）
       - xf_*    跨频率统计特征：由 15min CSV（aidc_load_15min_daily/）计算的
                 逐日日内统计（max/min/range/std/p95-p5/最大 15min 跳变/
                 峰值时刻/变异系数等），因果安全（当天日内统计当日末可知）
       - xr_*    跨 route 特征：对侧日负荷、双路总负荷、双路差
       - lbl_    事件标签（含居中窗口检测=有未来信息，仅供离线分析）+
                 lbl_prev_*（昨日口径的 spike/burst 点数，同属标签性质）
                 注意 lbl_volatile_day 的阈值严格 trailing，是唯一因果标签
  3. 输出（dataset/aidc_load_month/event_label_features/）：
       - <stem>_labeled_features.csv   逐日标签 + 全部特征
       - <stem>_events.csv             事件明细表（与 15min 脚本一致）
       - <stem>_events_overview.png    全序列事件标注总览图
       - load_event_analysis_report_daily.md  分析报告（两路合并）

用法（仓库根目录）：
    uv run python config/aidc_load_month/load_event_analysis.py
    uv run python config/aidc_load_month/load_event_analysis.py --routes A
    uv run python config/aidc_load_month/load_event_analysis.py --no-plot
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
    project_events_to_days,
    suppress_boundary_artifacts,
    topdown_day_segments,
)

# ---------------------------------------------------------------------------
# 场景路径
# ---------------------------------------------------------------------------
DATA_DAILY_DIR = Path("dataset/aidc_load_month")
DATA_15MIN_DIR = Path("dataset/aidc_load_15min_daily")
OUTPUT_DIR = DATA_DAILY_DIR / "event_label_features"
ROUTES = {
    "A": {
        "daily": DATA_DAILY_DIR / "A_Loads_1day_mean_20251001_20260731.csv",
        "f15": DATA_15MIN_DIR / "A_Loads_15min_mean_20251001_20260731.csv",
    },
    "B": {
        "daily": DATA_DAILY_DIR / "B_Loads_1day_mean_20251001_20260731.csv",
        "f15": DATA_15MIN_DIR / "B_Loads_15min_mean_20251001_20260731.csv",
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
def load_daily(path: Path) -> pd.Series:
    df = pd.read_csv(PROJECT_ROOT / path, parse_dates=["time"])
    s = df.set_index("time")["value"].sort_index().astype(float)
    if s.index.has_duplicates:
        s = s.groupby(level=0).mean()
    return s.asfreq("1D")


def load_15min(path: Path) -> pd.Series:
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
# 事件检测（与 15min 脚本同口径：日水平 = 15min 聚合的逐日中位数）
# ---------------------------------------------------------------------------
def detect_route_events(s15: pd.Series, cfg: EventDetectionConfig):
    day_stats = day_intraday_stats(s15)
    day_level = day_stats["intraday_median"].where(day_stats["intraday_n_points"] >= 90).dropna()

    segments = topdown_day_segments(day_level, cfg)
    seg_events = classify_day_events(segments, cfg)
    short_events = detect_short_excursions(day_level, cfg)
    day_events = merge_day_events(seg_events, short_events)

    intraday = detect_intraday_events(s15, cfg)
    intraday = suppress_boundary_artifacts(intraday, day_events, cfg)

    all_events = sorted(day_events + intraday, key=lambda e: (e.start, e.end))
    return day_events, intraday, all_events, day_stats


# ---------------------------------------------------------------------------
# 日频本频率特征（trailing，无未来泄漏）
# ---------------------------------------------------------------------------
def build_daily_features(daily: pd.Series) -> pd.DataFrame:
    idx = daily.index
    feat = pd.DataFrame(index=idx)
    feat["feat_dow"] = idx.dayofweek
    feat["feat_is_weekend"] = (idx.dayofweek >= 5).astype(int)
    feat["feat_month"] = idx.month
    feat["feat_day_of_month"] = idx.day

    for w in (7, 14, 30):
        roll = daily.rolling(w, min_periods=max(3, w // 3))
        feat[f"feat_rol{w}_mean"] = roll.mean()
        feat[f"feat_rol{w}_std"] = roll.std()
    feat["feat_med30"] = daily.rolling(30, min_periods=14).median().shift(1)

    feat["feat_diff1"] = daily.diff()
    feat["feat_diff1_pct"] = daily.pct_change() * 100
    feat["feat_diff7"] = daily - daily.shift(7)
    feat["feat_diff7_pct"] = daily.pct_change(7) * 100
    feat["feat_diff30"] = daily - daily.shift(30)
    feat["feat_diff30_pct"] = daily.pct_change(30) * 100
    feat["feat_diff1_of_diff7"] = feat["feat_diff7"].diff()

    med30 = feat["feat_med30"]
    mad30 = (daily - med30).abs().rolling(30, min_periods=14).median().shift(1)
    feat["feat_z30_robust"] = (daily - med30) / (1.4826 * mad30).clip(lower=30.0)

    slope = {}
    vals = daily.to_numpy()
    for i in range(len(vals)):
        lo = max(0, i - 29)
        y = vals[lo:i + 1]
        x = np.arange(len(y))
        if len(y) >= 10 and not np.isnan(y).any():
            slope[idx[i]] = float(np.polyfit(x, y, 1)[0])
        else:
            slope[idx[i]] = np.nan
    feat["feat_slope30"] = pd.Series(slope)
    return feat


# ---------------------------------------------------------------------------
# 跨频率特征：由 15min 数据计算的逐日日内统计
# ---------------------------------------------------------------------------
def build_cross_freq_features(s15: pd.Series, intraday_events: list[LoadEvent],
                              day_stats: pd.DataFrame) -> pd.DataFrame:
    """日内统计特征（因果安全）+ 事件点数统计（标签性质，已移入 lbl_prev_*）。

    xf_intraday_* 是当天已完整结束的日内统计，在 daily 场景语义
    （输入右界=昨日末，预测下一自然日）下无未来泄漏。
    spike/burst 点数来自居中窗口检测（MAD 定标看未来 <=3.5 天），
    属标签性质：重命名为 lbl_prev_spike_pts / lbl_prev_burst_pts。
    """
    stats = day_stats.copy()
    # spike / burst 点数统计（按事件起点归属到天；标签性质，昨日口径）
    spike_pts = {}
    burst_pts = {}
    for ev in intraday_events:
        d = ev.start.normalize()
        n_pts = int(round(ev.n_days * 96))
        if ev.kind.startswith("spike"):
            spike_pts[d] = spike_pts.get(d, 0) + n_pts
        elif ev.kind.startswith("burst"):
            burst_pts[d] = burst_pts.get(d, 0) + n_pts
    pts = pd.DataFrame(index=stats.index)
    pts["lbl_prev_spike_pts"] = pd.Series(spike_pts, dtype=float)
    pts["lbl_prev_burst_pts"] = pd.Series(burst_pts, dtype=float)
    pts = pts.fillna(0.0)
    # 昨日口径：D 日的行携带 D-1 日的事件点数（与检测确认延迟对齐）
    pts = pts.shift(1).fillna(0.0)
    stats = stats.join(pts)
    stats["xf_intraday_range_pct"] = stats["intraday_range"] / stats["intraday_mean"] * 100
    keep = stats.rename(columns={
        "intraday_mean": "xf_intraday_mean",
        "intraday_median": "xf_intraday_median",
        "intraday_std": "xf_intraday_std",
        "intraday_min": "xf_intraday_min",
        "intraday_max": "xf_intraday_max",
        "intraday_range": "xf_intraday_range",
        "intraday_p95": "xf_intraday_p95",
        "intraday_p5": "xf_intraday_p5",
        "intraday_p95_p5_gap": "xf_intraday_p95_p5_gap",
        "intraday_cv": "xf_intraday_cv",
        "intraday_max_abs_step": "xf_intraday_max_abs_step",
        "intraday_peak_time_frac": "xf_intraday_peak_time_frac",
        "intraday_n_points": "xf_intraday_n_points",
    })
    return keep


# ---------------------------------------------------------------------------
# 逐日标签 + 事件元信息
# ---------------------------------------------------------------------------
def build_day_labels(day_index: pd.DatetimeIndex, day_events, intraday_events,
                     day_stats, all_events) -> pd.DataFrame:
    labels = project_events_to_days(day_index, day_events, intraday_events, day_stats)

    # 每日事件元信息：归属事件 id / 事件内第几天 / 距上一事件天数 / 事件幅度
    idx = labels.index
    event_id = np.full(len(idx), -1, dtype=int)
    days_into = np.full(len(idx), np.nan)
    amplitude = np.full(len(idx), np.nan)

    spans = []
    for eid, ev in enumerate(all_events, start=1):
        if ev.kind.startswith("shift"):
            d_start, d_end = ev.start, ev.start
        elif ev.source == "intraday":
            d_start, d_end = ev.start.normalize(), ev.start.normalize()
        else:
            d_start, d_end = ev.start, ev.end
        spans.append((d_start, d_end, ev.kind, eid, ev.amplitude))

    for i, d in enumerate(idx):
        best = None
        for d_start, d_end, kind, eid, amp in spans:
            if d_start <= d <= d_end:
                if best is None or KIND_PRIORITY[kind] < KIND_PRIORITY[best[2]]:
                    best = (d_start, d_end, kind, eid, amp)
        if best is not None:
            event_id[i] = best[3]
            days_into[i] = (d - best[0]).days + 1
            amplitude[i] = best[4]

    labels["lbl_event_id"] = event_id
    labels["lbl_days_into_event"] = days_into
    labels["lbl_event_amplitude_kw"] = amplitude

    ends = np.array([ev.end for ev in all_events], dtype="datetime64[ns]")
    days_since = np.full(len(idx), np.nan)
    for i, d in enumerate(idx):
        d64 = np.datetime64(d)
        prev = ends[ends < d64]
        if len(prev):
            days_since[i] = (d64 - prev.max()) / np.timedelta64(1, "D")
    labels["lbl_days_since_event"] = days_since
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


def plot_route(daily: pd.Series, day_events, intraday_events, day_stats,
               out_png: Path, route: str) -> None:
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(26, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(daily.index, daily.values, color="#2F5597", linewidth=1.0,
            marker="o", markersize=2.2, label="daily mean load")
    for ev in day_events:
        color = KIND_COLORS.get(ev.kind, "#AAAAAA")
        if ev.kind.startswith("stress"):
            ax.axvspan(ev.start, ev.end + pd.Timedelta(days=1), color=color, alpha=0.16)
        else:
            ax.axvline(ev.start, color=color, linestyle="--", linewidth=1.3)
            ax.annotate(f"{ev.kind}\n{ev.amplitude:+.0f}kW",
                        xy=(ev.start, ev.base_level + ev.amplitude),
                        fontsize=7, color=color, xytext=(5, 0), textcoords="offset points")
    burst_days = sorted({ev.start.normalize() for ev in intraday_events})
    if burst_days:
        ax.scatter(burst_days, daily.reindex(burst_days), marker="x", s=60,
                   color="#111111", label=f"intraday event days ({len(burst_days)})", zorder=5)
    ax.set_title(f"AIDC {route} daily load event overview "
                 f"(day-level events={len(day_events)}, intraday event days={len(burst_days)})")
    ax.set_ylabel("load (kW)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    ax2 = axes[1]
    rng = day_stats["intraday_range"]
    ax2.plot(rng.index, rng, color="#5B6770", linewidth=0.9, label="intraday range (from 15min)")
    base = rng.shift(1).rolling(60, min_periods=14).quantile(0.95)
    ax2.plot(base.index, base, color="#C00000", linewidth=1.1, linestyle="--",
             label="rolling p95 (60d)")
    ax2.set_ylabel("intraday range")
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
def process_route(route: str, cfg: EventDetectionConfig, peer_daily: pd.Series,
                  make_plot: bool) -> dict:
    paths = ROUTES[route]
    daily = load_daily(paths["daily"])
    s15 = load_15min(paths["f15"])
    meta = load_aggregate_meta(paths["daily"])

    day_events, intraday_events, all_events, day_stats = detect_route_events(s15, cfg)

    feat = build_daily_features(daily)
    xf = build_cross_freq_features(s15, intraday_events, day_stats)
    labels = build_day_labels(daily.index, day_events, intraday_events, day_stats, all_events)

    out = pd.DataFrame({"time": daily.index, "value": daily.values}).set_index("time")
    out = out.join(feat)
    out = out.join(xf)
    out = out.join(labels)

    out["xr_peer_value"] = peer_daily.reindex(out.index)
    out["xr_total_load"] = out["value"] + out["xr_peer_value"]
    out["xr_route_diff"] = out["value"] - out["xr_peer_value"]
    out["xr_route_diff_pct"] = out["xr_route_diff"] / out["xr_peer_value"] * 100

    stem = paths["daily"].stem
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
        plot_route(daily, day_events, intraday_events, day_stats, png, route)

    return {
        "route": route,
        "series": daily,
        "features": out,
        "events": events_frame,
        "day_events": day_events,
        "intraday_events": intraday_events,
        "meta": meta,
        "paths": {"features": features_csv, "events": events_csv, "png": png},
        "label_coverage": float(out["lbl_event_day"].mean()),
        "kind_counts": events_frame["kind"].value_counts().to_dict() if not events_frame.empty else {},
        "type_counts": out["lbl_event_type"].value_counts().to_dict(),
    }


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def write_report(results: list[dict], cfg: EventDetectionConfig, out_md: Path) -> None:
    lines = ["# AIDC 日频负荷事件标签与特征工程报告", ""]
    lines.append("- 数据：dataset/aidc_load_month/ 下 A/B 路 1day 粒度负荷"
                 "（目录名沿用场景命名 aidc_load_month）")
    lines.append("- 生成命令：`uv run python config/aidc_load_month/load_event_analysis.py`")
    lines.append("- 检测核心：`data_process/load_event_detection.py`（与 15min 脚本共享同一口径；"
                 "日水平基线取 15min 数据聚合的逐日中位数，保证跨频率标签一致）")
    lines.append("")
    lines.append("## 1. 事件分类学（日粒度标签）")
    lines.append("")
    lines.append("| 标签 | 含义 | 业务解释 | 判定方式 |")
    lines.append("|---|---|---|---|")
    lines.append("| shift_up_day / shift_down_day | 持久水平阶跃起始日 | 设备集中上架/下架 | 分段水平差 ≥ max(300kW, 2%) 且后续不回落 |")
    lines.append("| stress_up_day / stress_down_day | 临时水平偏移日 | 压力测试 / 临时批量操作 / 检修迁移 | 阶跃后 21 天内回落到基线 55% 以内（含 1~2 天短时偏移） |")
    lines.append("| burst_up_day / burst_down_day | 日内持续冲击日 | 压测冲击 / 突发负荷块 | 15min 残差 > max(6σ, 350kW) 连续 5~96 点 |")
    lines.append("| spike_up_day / spike_down_day | 功率突变日 | 瞬时功率突变 / 切换操作 | 15min 残差异常连续 ≤4 点 |")
    lines.append("| volatile_day | 日内波动异常日 | 调度/负载剧烈变化 | 15min 日内 std 超 60 天滚动 p95 |")
    lines.append("")
    lines.append(f"阈值配置：seg=({cfg.seg_min_kw}kW/{cfg.seg_min_pct:.0%}), "
                 f"short=({cfg.short_min_kw}kW/{cfg.short_min_pct:.0%}), "
                 f"intraday=(k={cfg.intraday_k_sigma}σ, {cfg.intraday_min_kw}kW)。")
    lines.append("")

    for pos, res in enumerate(results, start=1):
        route, meta, s = res["route"], res["meta"], res["series"]
        lines.append(f"## 2.{pos} {route} 路数据概况")
        lines.append("")
        lines.append(f"- 样本数：{len(s)} 天（{s.index.min().date()} ~ {s.index.max().date()}）")
        if meta:
            lines.append(f"- 来源：{meta.get('config', {}).get('source_freq')} -> "
                         f"{meta.get('config', {}).get('target_freq')} "
                         f"{meta.get('config', {}).get('method')} 聚合，线性填补 "
                         f"{meta.get('filled_value_count')} 点")
        lines.append(f"- 日均负荷区间：{s.min():.0f} ~ {s.max():.0f} kW，中位数 {s.median():.0f} kW")
        lines.append(f"- 事件总数：{len(res['events'])}（日级 {len(res['day_events'])}，"
                     f"日内 {len(res['intraday_events'])}）；事件日覆盖率 {res['label_coverage']:.2%}")
        tc = res["type_counts"]
        lines.append(f"- 逐日类型分布：{ {k: int(v) for k, v in sorted(tc.items())} }")
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
    lines.append("| feat_ | 日频本频率 trailing 特征：日历（dow/month/dom/weekend）、滚动统计（7/14/30d mean/std、30d 中位数）、"
                 "差分（日环比/周环比/月环比绝对+百分比、周环比的一阶差分）、30d robust z、30d 线性趋势斜率 | feat_z30_robust |")
    lines.append("| xf_ | 跨频率统计特征（来自 aidc_load_15min_daily 15min CSV，因果安全）：日内 mean/median/std/min/max/range/p95/p5/"
                 "p95-p5 差/cv/相邻点最大跳变/峰值时刻/点数、range 占比 | xf_intraday_range |")
    lines.append("| xr_ | 跨 route 特征：对侧路日负荷、双路总负荷、双路差及占比 | xr_total_load |")
    lines.append("| lbl_ | 事件标签（检测含居中窗口，仅供分析/筛选）：8 类事件日 0/1 列、volatile_day、event_day、"
                 "event_type、event_id、days_into_event、days_since_event、event_amplitude_kw；"
                 "另含 lbl_prev_spike_pts/lbl_prev_burst_pts（昨日口径事件点数，同属标签性质） | lbl_shift_up_day |")
    lines.append("")
    lines.append("## 4. 使用注意")
    lines.append("")
    lines.append("- lbl_* 标签由含未来信息的居中窗口检测产生，用于离线分析、事件期样本筛选、评估；"
                 "lbl_prev_*（昨日口径 spike/burst 点数）虽已 shift(1)，其检测仍含居中窗口，同属标签性质。"
                 "唯一例外：lbl_volatile_day 的阈值严格 trailing（std.shift(1).rolling(60).p95），"
                 "是因果标签，经 lag 消费后可作建模特征。")
    lines.append("- feat_* 为日频 trailing 统计、xf_intraday_* 为当天已完整结束的日内统计、xr_* 为对侧路当日值，"
                 "在本场景语义（输入右界=昨日末、预测下一自然日）下三者均无未来泄漏。"
                 "嵌入 value_D/peer_D 的列喂入 custom_features 外生通路前需先 lag，避免同行监督泄漏。")
    lines.append("- 15min（config/aidc_load_15min_daily/load_event_analysis.py）与日频脚本的事件明细表"
                 "内容一致（同一检测核心、同一基线序列），可互相对账。")
    out_md.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="AIDC 日频负荷事件标签与特征工程")
    parser.add_argument("--routes", nargs="+", default=["A", "B"], choices=sorted(ROUTES))
    parser.add_argument("--no-plot", action="store_true", help="跳过总览图生成")
    args = parser.parse_args()

    cfg = EventDetectionConfig()
    daily_map = {r: load_daily(ROUTES[r]["daily"]) for r in args.routes}

    results = []
    for route in args.routes:
        peer_route = "B" if route == "A" else "A"
        peer = daily_map.get(peer_route)
        if peer is None:
            peer = load_daily(ROUTES[peer_route]["daily"])
        res = process_route(route, cfg, peer, make_plot=not args.no_plot)
        results.append(res)
        print(f"[{route}] days={len(res['features'])}, events={len(res['events'])} "
              f"(day={len(res['day_events'])}, intraday={len(res['intraday_events'])}), "
              f"event_day_coverage={res['label_coverage']:.2%}")
        print(f"  -> {res['paths']['features']}")
        print(f"  -> {res['paths']['events']}")
        if res["paths"]["png"]:
            print(f"  -> {res['paths']['png']}")

    out_md = PROJECT_ROOT / OUTPUT_DIR / "load_event_analysis_report_daily.md"
    write_report(results, cfg, out_md)
    print(f"  -> {out_md}")


if __name__ == "__main__":
    main()
