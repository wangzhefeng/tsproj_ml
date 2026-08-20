# -*- coding: utf-8 -*-
"""构造月频滑窗回测的 ex-ante 气象代理。

当前可评估目标月份为 2026-04～07。每个月只使用上一年同月的历史气象统计，
避免把测试月整月实测气象传给模型。输出时间戳保留目标月月末标签。
"""
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_PATH = PROJECT_ROOT / "dataset/aidc_power_month/freq_1month/weather_monthly_stats_202510_202607.csv"
OUTPUT_PATH = PROJECT_ROOT / "dataset/aidc_power_month/freq_1month/weather_monthly_stats_backtest_proxy_202604_202607.csv"
TARGET_MONTHS = pd.date_range("2026-04-30", "2026-07-31", freq="1ME")
WEATHER_COLS = ["rt_tt2", "rt_tt2_max", "rt_tt2_min", "cal_rh", "rt_ssr", "rt_ws10", "rt_dt"]


def build_backtest_monthly_proxy() -> pd.DataFrame:
    source = pd.read_csv(SOURCE_PATH)
    source["ts"] = pd.to_datetime(source["ts"])
    source = source.set_index("ts")

    records = []
    for target_ts in TARGET_MONTHS:
        source_ts = target_ts - pd.DateOffset(years=1)
        # DateOffset 保持月末语义；闰年等边界统一归一到该月月末。
        source_ts = source_ts.to_period("M").to_timestamp("M")
        if source_ts not in source.index:
            raise ValueError(f"Missing prior-year weather source for target {target_ts.date()}: {source_ts.date()}")
        row = source.loc[source_ts, WEATHER_COLS]
        record = {"ts": target_ts, **row.to_dict()}
        record.update({
            "weather_source": "prior_year_proxy",
            "source_ts": source_ts,
            "available_at": (target_ts.to_period("M") - 1).to_timestamp("M"),
        })
        records.append(record)

    backtest = pd.DataFrame(records)
    if backtest[WEATHER_COLS].isna().to_numpy().any():
        raise ValueError("Backtest monthly proxy contains NaN weather values.")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    backtest.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"backtest monthly weather proxy saved: {OUTPUT_PATH}")
    print(backtest.to_string(index=False))
    return backtest


if __name__ == "__main__":
    build_backtest_monthly_proxy()
