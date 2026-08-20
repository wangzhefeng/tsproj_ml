# -*- coding: utf-8 -*-
"""日频滑窗回测的 ex-ante 气象代理（严格信息集）。

规则：目标日 t 的回测天气 = 上一年同日历日（去年同月同日）的实测日统计。
闰日 02-29 回落取 02-28。available_at = 目标日前一个月的月末，保证在每个
自然月 fold 的预测原点（月初）之前可得。输出带 available_at 溯源列。
"""
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_PATH = PROJECT_ROOT / "dataset/aidc_power_month/freq_1day/weather_daily_stats_20251001_20260731.csv"
OUTPUT_PATH = PROJECT_ROOT / "dataset/aidc_power_month/freq_1day/weather_daily_stats_backtest_proxy_20260101_20260731.csv"
TARGET_START, TARGET_END = "2026-01-01", "2026-07-31"
WEATHER_COLS = ["rt_tt2", "rt_tt2_max", "rt_tt2_min", "cal_rh", "rt_ssr", "rt_ws10", "rt_dt"]


def _prior_year_day(ts: pd.Timestamp) -> pd.Timestamp:
    """上一年同日历日；闰日 02-29 回落 02-28。"""
    if ts.month == 2 and ts.day == 29:
        return pd.Timestamp(year=ts.year - 1, month=2, day=28)
    return pd.Timestamp(year=ts.year - 1, month=ts.month, day=ts.day)


def build_backtest_daily_proxy() -> pd.DataFrame:
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
            "available_at": (target.to_period("M") - 1).to_timestamp("M"),
        })
    if missing:
        raise ValueError(f"Prior-year daily weather missing for {len(missing)} target days: {missing[:10]}")

    backtest = pd.DataFrame(records)
    if backtest[WEATHER_COLS].isna().to_numpy().any():
        raise ValueError("Backtest daily proxy contains NaN weather values.")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    backtest.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"backtest daily weather proxy saved: {OUTPUT_PATH}")
    print(f"rows={len(backtest)}, range={backtest['ts'].min().date()} -> {backtest['ts'].max().date()}")
    return backtest


if __name__ == "__main__":
    build_backtest_daily_proxy()
