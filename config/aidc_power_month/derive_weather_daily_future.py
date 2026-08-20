# -*- coding: utf-8 -*-
"""2026-08 未来日度气象统计：上一年同日纯代理（严格 forecast 信息集）。

预测原点为 2026-07-31，不得使用 2026-08 的任何实测值。目标日 t 的天气 =
2025 年同日历日的实测日统计（闰日 02-29 回落 02-28），ts 标签平移到
2026-08。available_at = 2026-07-31（预测原点本身），全表无 NaN。
"""
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_PATH = PROJECT_ROOT / "dataset/aidc_power_month/freq_1day/weather_daily_stats_20251001_20260731.csv"
OUTPUT_PATH = PROJECT_ROOT / "dataset/aidc_power_month/freq_1day/weather_daily_stats_future_20260801_20260831.csv"
TARGET_START, TARGET_END = "2026-08-01", "2026-08-31"
AVAILABLE_AT = "2026-07-31"
WEATHER_COLS = ["rt_tt2", "rt_tt2_max", "rt_tt2_min", "cal_rh", "rt_ssr", "rt_ws10", "rt_dt"]


def _prior_year_day(ts: pd.Timestamp) -> pd.Timestamp:
    """上一年同日历日；闰日 02-29 回落 02-28。"""
    if ts.month == 2 and ts.day == 29:
        return pd.Timestamp(year=ts.year - 1, month=2, day=28)
    return pd.Timestamp(year=ts.year - 1, month=ts.month, day=ts.day)


def build_future_daily_proxy() -> pd.DataFrame:
    source = pd.read_csv(SOURCE_PATH)
    source["ts"] = pd.to_datetime(source["ts"])
    source = source.drop_duplicates(subset="ts", keep="last").set_index("ts")

    records = []
    missing = []
    for target in pd.date_range(TARGET_START, TARGET_END, freq="1D"):
        src = _prior_year_day(target)
        if src not in source.index:
            missing.append(str(target.date()))
            continue
        row = source.loc[src, WEATHER_COLS]
        records.append({
            "ts": target,
            **row.to_dict(),
            "weather_source": "prior_year_proxy",
            "source_ts": src,
            "available_at": AVAILABLE_AT,
        })
    if missing:
        raise ValueError(f"Prior-year daily weather missing for {len(missing)} target days: {missing[:10]}")

    future = pd.DataFrame(records)
    if future[WEATHER_COLS].isna().to_numpy().any():
        raise ValueError("Future daily proxy contains NaN weather values.")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    future.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"future daily weather proxy saved: {OUTPUT_PATH}")
    print(f"rows={len(future)}, range={future['ts'].min().date()} -> {future['ts'].max().date()}")
    return future


if __name__ == "__main__":
    build_future_daily_proxy()
