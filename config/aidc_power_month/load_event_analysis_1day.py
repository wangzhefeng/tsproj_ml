# -*- coding: utf-8 -*-
"""AIDC 日用电量（1D sum kWh）事件标签与特征工程。

复用 ``config/aidc_load_month/load_event_analysis.py`` 的事件检测与特征函数：

- 日频 ``feat_*`` 基于本场景日用电量目标（kWh）计算；
- ``xf_*`` 与事件检测来自 15min 负荷功率（kW），保留日内形态；
- ``xr_*`` 使用 A/B 路日用电量（kWh），列名显式携带单位；
- ``lbl_*`` 仍是离线标签，除 volatile 外不可直接作为未来预测特征。

输出目录：``dataset/aidc_power_month/freq_1day/event_label_features/``。

用法：
    env -u PYTHONPATH .venv/bin/python config/aidc_power_month/load_event_analysis_1day.py
    env -u PYTHONPATH .venv/bin/python config/aidc_power_month/load_event_analysis_1day.py --routes A
    env -u PYTHONPATH .venv/bin/python config/aidc_power_month/load_event_analysis_1day.py --no-plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.aidc_load_month.load_event_analysis import (  # noqa: E402
    KIND_COLORS,
    _setup_matplotlib,
    build_cross_freq_features,
    build_daily_features,
    build_day_labels,
    detect_route_events,
    load_aggregate_meta,
)
from data_process.load_event_detection import (  # noqa: E402
    EventDetectionConfig,
    events_to_frame,
)


DATA_ENERGY_DIR = Path("dataset/aidc_power_month/freq_1day")
DATA_15MIN_DIR = Path("dataset/aidc_load_15min_daily")
OUTPUT_DIR = DATA_ENERGY_DIR / "event_label_features"
ROUTES = {
    "A": {
        "energy": DATA_ENERGY_DIR / "A_Loads_1day_sum_20251001_20260731.csv",
        "f15": DATA_15MIN_DIR / "A_Loads_15min_mean_20251001_20260731.csv",
    },
    "B": {
        "energy": DATA_ENERGY_DIR / "B_Loads_1day_sum_20251001_20260731.csv",
        "f15": DATA_15MIN_DIR / "B_Loads_15min_mean_20251001_20260731.csv",
    },
}


def load_series(path: Path, freq: str) -> pd.Series:
    """读取 time/value CSV，去重并规则化到指定频率。"""
    frame = pd.read_csv(PROJECT_ROOT / path, parse_dates=["time"])
    series = frame.set_index("time")["value"].sort_index().astype(float)
    if series.index.has_duplicates:
        series = series.groupby(level=0).mean()
    return cast(pd.Series, series.asfreq(freq))


def build_energy_daily_features(energy_daily: pd.Series) -> pd.DataFrame:
    """构造日用电量 trailing 特征，并按 kWh 量纲修正 robust MAD 下限。"""
    features = build_daily_features(energy_daily)
    med30 = energy_daily.rolling(30, min_periods=14).median().shift(1)
    mad30 = (energy_daily - med30).abs().rolling(30, min_periods=14).median().shift(1)
    # 负荷脚本使用 30kW 下限；日用电量 E=Pmean×24，对应 720kWh。
    features["feat_z30_robust"] = (
        (energy_daily - med30) / (1.4826 * mad30).clip(lower=30.0 * 24.0)
    )
    return features


def build_energy_feature_frame(
    energy_daily: pd.Series,
    peer_energy_daily: pd.Series,
    load_15min: pd.Series,
    cfg: EventDetectionConfig | None = None,
) -> dict[str, Any]:
    """构建单路日用电量特征帧；事件检测仍使用 15min 功率口径。"""
    cfg = cfg or EventDetectionConfig()
    energy_daily = energy_daily.sort_index().asfreq("1D").astype(float)
    peer_energy_daily = peer_energy_daily.sort_index().asfreq("1D").astype(float)
    load_15min = load_15min.sort_index().asfreq("15min").astype(float)

    day_events, intraday_events, all_events, day_stats = detect_route_events(load_15min, cfg)
    daily_features = build_energy_daily_features(energy_daily)
    cross_freq = build_cross_freq_features(load_15min, intraday_events, day_stats)
    labels = build_day_labels(
        cast(pd.DatetimeIndex, energy_daily.index),
        day_events,
        intraday_events,
        day_stats,
        all_events,
    )

    output = pd.DataFrame(
        {"time": energy_daily.index, "value": energy_daily.values}
    ).set_index("time")
    output = output.join(daily_features)
    output = output.join(cross_freq)
    output = output.join(labels)

    output["xr_peer_energy_kwh"] = peer_energy_daily.reindex(output.index)
    output["xr_total_energy_kwh"] = output["value"] + output["xr_peer_energy_kwh"]
    output["xr_route_diff_kwh"] = output["value"] - output["xr_peer_energy_kwh"]
    output["xr_route_diff_pct"] = (
        output["xr_route_diff_kwh"] / output["xr_peer_energy_kwh"] * 100.0
    )

    return {
        "features": output,
        "events": events_to_frame(all_events),
        "day_events": day_events,
        "intraday_events": intraday_events,
        "day_stats": day_stats,
    }


def plot_route(
    energy_daily: pd.Series,
    day_events,
    intraday_events,
    day_stats: pd.DataFrame,
    out_png: Path,
    route: str,
) -> None:
    """绘制日用电量趋势、事件位置和 15min 日内功率范围。"""
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(26, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax = axes[0]
    ax.plot(
        energy_daily.index,
        energy_daily.values,
        color="#2F5597",
        linewidth=1.0,
        marker="o",
        markersize=2.2,
        label="daily energy",
    )
    for event in day_events:
        color = KIND_COLORS.get(event.kind, "#AAAAAA")
        if event.kind.startswith("stress"):
            ax.axvspan(
                event.start,
                event.end + pd.Timedelta(days=1),
                color=color,
                alpha=0.16,
            )
        else:
            ax.axvline(event.start, color=color, linestyle="--", linewidth=1.3)
            event_day = event.start.normalize()
            y_value = energy_daily.reindex([event_day]).iloc[0]
            if pd.notna(y_value):
                ax.annotate(
                    f"{event.kind}\n{event.amplitude:+.0f}kW",
                    xy=(event_day, y_value),
                    fontsize=7,
                    color=color,
                    xytext=(5, 0),
                    textcoords="offset points",
                )
    intraday_days = sorted({event.start.normalize() for event in intraday_events})
    if intraday_days:
        ax.scatter(
            intraday_days,
            energy_daily.reindex(intraday_days),
            marker="x",
            s=60,
            color="#111111",
            label=f"intraday event days ({len(intraday_days)})",
            zorder=5,
        )
    ax.set_title(
        f"AIDC {route} daily energy event overview "
        f"(day-level events={len(day_events)}, intraday event days={len(intraday_days)})"
    )
    ax.set_ylabel("daily energy (kWh)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    ax2 = axes[1]
    intraday_range = day_stats["intraday_range"]
    threshold = intraday_range.shift(1).rolling(60, min_periods=14).quantile(0.95)
    ax2.plot(
        intraday_range.index,
        intraday_range,
        color="#5B6770",
        linewidth=0.9,
        label="intraday load range (kW)",
    )
    ax2.plot(
        threshold.index,
        threshold,
        color="#C00000",
        linewidth=1.1,
        linestyle="--",
        label="rolling p95 (60d)",
    )
    ax2.set_ylabel("intraday range (kW)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8)

    fig.autofmt_xdate()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def process_route(
    route: str,
    cfg: EventDetectionConfig,
    peer_energy_daily: pd.Series,
    make_plot: bool,
) -> dict[str, Any]:
    paths = ROUTES[route]
    energy_daily = load_series(paths["energy"], "1D")
    load_15min = load_series(paths["f15"], "15min")
    result = build_energy_feature_frame(
        energy_daily=energy_daily,
        peer_energy_daily=peer_energy_daily,
        load_15min=load_15min,
        cfg=cfg,
    )

    out_dir = PROJECT_ROOT / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = paths["energy"].stem
    features_csv = out_dir / f"{stem}_labeled_features.csv"
    events_csv = out_dir / f"{stem}_events.csv"
    result["features"].reset_index().to_csv(
        features_csv,
        index=False,
        encoding="utf-8-sig",
    )
    result["events"].to_csv(events_csv, index=False, encoding="utf-8-sig")

    png = None
    if make_plot:
        png = out_dir / f"{stem}_events_overview.png"
        plot_route(
            energy_daily,
            result["day_events"],
            result["intraday_events"],
            result["day_stats"],
            png,
            route,
        )

    result.update(
        {
            "route": route,
            "series": energy_daily,
            "meta": load_aggregate_meta(paths["energy"]),
            "paths": {"features": features_csv, "events": events_csv, "png": png},
            "label_coverage": float(result["features"]["lbl_event_day"].mean()),
            "type_counts": result["features"]["lbl_event_type"].value_counts().to_dict(),
        }
    )
    return result


def write_report(results: list[dict[str, Any]], out_md: Path) -> None:
    lines = [
        "# AIDC 日用电量事件标签与特征工程报告（freq_1day）",
        "",
        "- 目标：`dataset/aidc_power_month/freq_1day/` 的 1D sum 日用电量（kWh）",
        "- 日内来源：`dataset/aidc_load_15min_daily/` 的 15min 负荷功率（kW）",
        "- 检测核心：复用 `data_process/load_event_detection.py`，事件明细应与负荷日频场景一致",
        "- 生成命令：`env -u PYTHONPATH .venv/bin/python config/aidc_power_month/load_event_analysis_1day.py`",
        "",
    ]
    for result in results:
        route = result["route"]
        series = result["series"]
        lines.extend(
            [
                f"## {route} 路",
                "",
                f"- 样本数：{len(series)} 天（{series.index.min().date()} ~ {series.index.max().date()}）",
                f"- 日用电量区间：{series.min():.0f} ~ {series.max():.0f} kWh，中位数 {series.median():.0f} kWh",
                f"- 事件数：{len(result['events'])}；事件日覆盖率：{result['label_coverage']:.2%}",
                f"- 每日类型分布：{ {k: int(v) for k, v in sorted(result['type_counts'].items())} }",
                "",
            ]
        )

    lines.extend(
        [
            "## 特征单位与使用边界",
            "",
            "| 前缀 | 口径 | 用途 |",
            "|---|---|---|",
            "| `feat_*` | 日用电量 kWh 的 trailing 特征 | 历史状态分析；未来消费需满足预测原点信息集 |",
            "| `xf_*` | 15min 负荷功率 kW 的日内形态统计 | 日内波动状态；同日 level 列不可作为 future 实际值 |",
            "| `xr_*` | A/B 路日用电量 kWh 及相对比例 | 跨路历史状态 |",
            "| `lbl_*` | 事件标签，事件幅度仍为 kW | 仅离线分析/评估；居中检测标签禁止直接作未来特征 |",
            "",
            "注意：日用电量与日均负荷满足 `energy_kWh = mean_load_kW × 24`。因此 `value`、",
            "`xf_intraday_mean` 以及同日负荷 level 代理不能直接作为目标日 future 外生输入。",
            "本产物是完整分析表，不等同于在线安全的模型输入表；在线建模需另行派生预测原点快照。",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="AIDC 日用电量 1D 事件标签与特征工程")
    parser.add_argument("--routes", nargs="+", default=["A", "B"], choices=sorted(ROUTES))
    parser.add_argument("--no-plot", action="store_true", help="跳过事件总览图")
    args = parser.parse_args()

    cfg = EventDetectionConfig()
    energy_map = {
        route: load_series(ROUTES[route]["energy"], "1D")
        for route in args.routes
    }
    results = []
    for route in args.routes:
        peer_route = "B" if route == "A" else "A"
        peer = energy_map.get(peer_route)
        if peer is None:
            peer = load_series(ROUTES[peer_route]["energy"], "1D")
        result = process_route(
            route,
            cfg,
            peer_energy_daily=peer,
            make_plot=not args.no_plot,
        )
        results.append(result)
        print(
            f"[{route}] days={len(result['features'])}, events={len(result['events'])}, "
            f"event_day_coverage={result['label_coverage']:.2%}"
        )
        print(f"  -> {result['paths']['features']}")
        print(f"  -> {result['paths']['events']}")
        if result["paths"]["png"] is not None:
            print(f"  -> {result['paths']['png']}")

    report = PROJECT_ROOT / OUTPUT_DIR / "load_event_analysis_report_1day_energy.md"
    write_report(results, report)
    print(f"  -> {report}")


if __name__ == "__main__":
    main()
