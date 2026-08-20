# -*- coding: utf-8 -*-
"""构造 2026-08 月频未来气象代理（严格 forecast 信息集）。

预测原点为 2026-07-31，因此不得使用 2026-08 的任何实测值。本脚本从历史月度
统计中复制 2025-08 同月值，生成 ts=2026-08-31 的 prior-year proxy。
"""
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_PATH = PROJECT_ROOT / "dataset/aidc_power_month/freq_1month/weather_monthly_stats_202510_202607.csv"
OUTPUT_PATH = PROJECT_ROOT / "dataset/aidc_power_month/freq_1month/weather_monthly_stats_future_202608.csv"
SOURCE_MONTH = pd.Timestamp("2025-08-31")
TARGET_MONTH = pd.Timestamp("2026-08-31")
WEATHER_COLS = ["rt_tt2", "rt_tt2_max", "rt_tt2_min", "cal_rh", "rt_ssr", "rt_ws10", "rt_dt"]


def build_future_monthly_proxy() -> pd.DataFrame:
    source = pd.read_csv(SOURCE_PATH)
    source["ts"] = pd.to_datetime(source["ts"])
    rows = source.loc[source["ts"] == SOURCE_MONTH, ["ts"] + WEATHER_COLS]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one source row for {SOURCE_MONTH.date()}, got {len(rows)}")

    future = rows.copy().reset_index(drop=True)
    future["ts"] = TARGET_MONTH
    future["weather_source"] = "prior_year_proxy"
    future["source_ts"] = SOURCE_MONTH
    future["available_at"] = pd.Timestamp("2026-07-31")
    if future[WEATHER_COLS].isna().to_numpy().any():
        raise ValueError("Future monthly proxy contains NaN weather values.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    future.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"future monthly weather proxy saved: {OUTPUT_PATH}")
    print(future.to_string(index=False))
    return future


if __name__ == "__main__":
    build_future_monthly_proxy()
