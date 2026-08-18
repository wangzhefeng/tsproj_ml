# -*- coding: utf-8 -*-
"""2026-08 未来月度气象统计构造：前半月实测 + 后半月去年同期仿真。

为 freq_1month 场景的 is_forecasting 提供下月（2026-08）外生特征：
  - 2026-08-01 ~ 08-14：weather_future_in_20260801_20260814.csv 的 rt_ 实测列
    （生成该文件时这段已成过去，rt_ 列已由实测回填，优于 pred_ 预报列）
  - 2026-08-15 ~ 08-31：weather_in_20250101_20260731.csv 的 2025-08-15 ~ 08-31
    去年同期实测仿真（去年同期法；数据室内负荷对气温年周期敏感，误差可控）

为什么不用 pred_ 预报列：历史逐时配对统计显示 pred_ws10 相对 rt_ws10 系统性
高估约 2 倍、pred_ssrd 高估 1.1~1.2 倍（见 2026-08-18 分析），rt_ 口径优先。

输出 schema 与 weather_monthly_stats_202510_202607.csv 完全一致
（ts, rt_tt2, rt_tt2_max, rt_tt2_min, cal_rh, rt_ssr, rt_ws10, rt_dt），
ts = 2026-08-31（月末标签，与 freq_1month 未来时间索引对齐），
列名复用框架 weather 白名单 → extend_future_weather_feature 白名单直连分支
按时间戳 merge，无需 pred_ 映射。

统计口径与 derive_weather_monthly.py 一致：
  rt_tt2 月均 / rt_tt2_max 月最高 / rt_tt2_min 月最低 / cal_rh 月均（Magnus-Tetens
  由 rt_tt2+rt_dt 计算）/ rt_ssr 月总辐射 / rt_ws10 月均风速 / rt_dt 月均露点

用法（仓库根目录）：
    uv run python config/aidc_power_month/derive_weather_monthly_future.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ACTUAL_PATH = PROJECT_ROOT / "dataset/aidc_load_5min/weather_future_in_20260801_20260814.csv"
HISTORY_PATH = PROJECT_ROOT / "dataset/aidc_load_5min/weather_in_20250101_20260731.csv"
OUTPUT_PATH = PROJECT_ROOT / "dataset/aidc_power_month/freq_1month/weather_monthly_stats_future_202608.csv"

# 未来月标签（月末 00:00:00，与 1ME resample 标签约定一致）
FUTURE_MONTH_END = "2026-08-31"
# 实测段 / 仿真段的日界（含头含尾）
ACTUAL_START, ACTUAL_END = "2026-08-01", "2026-08-14"
SIM_LAST_YEAR_START, SIM_LAST_YEAR_END = "2025-08-15", "2025-08-31"


def _calc_rh(tt2_k: pd.Series, dt_k: pd.Series) -> pd.Series:
    """Magnus-Tetens 公式：rt_tt2/rt_dt 为 Kelvin，输出相对湿度 %（与历史脚本同口径）。"""
    t_air = tt2_k - 273.15
    t_dew = dt_k - 273.15
    e_s_td = 6.1078 * np.exp((17.2693 * t_dew) / (237.29 + t_dew))
    e_s_t = 6.1078 * np.exp((17.2693 * t_air) / (237.29 + t_air))
    return pd.Series(np.clip((e_s_td / e_s_t) * 100, 0, 100), index=tt2_k.index)


def _load_rt_segment(path: Path, start: str, end: str, label: str) -> pd.DataFrame:
    """读取一个时间段的 1h rt_ 实测段，返回 [ts, rt_tt2, rt_dt, rt_ssr, rt_ws10]。"""
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"])
    mask = (df["ts"] >= start) & (df["ts"] <= end + " 23:59:59")
    seg: pd.DataFrame = df.loc[mask].copy()
    rt_cols = ["rt_tt2", "rt_dt", "rt_ssr", "rt_ws10"]
    for col in rt_cols:
        seg[col] = pd.to_numeric(seg[col], errors="coerce")
    sub = seg[rt_cols]
    missing_mask = sub.isna().to_numpy().any(axis=1)
    n_missing = int(missing_mask.sum())
    if n_missing:
        # 个别缺失点由月统计的 mean/max/min 天然跳过（sum 口径的 rt_ssr 缺失会轻微低估，
        # 2026-08 段缺失 6/336 点、2025-08 段缺失 1/408 点，量级可忽略）
        print(f"  [warn] {label}: {n_missing} 行含缺失 rt_ 值（统计时跳过）")
    print(f"  {label}: {len(seg)} 行 ({seg['ts'].min()} -> {seg['ts'].max()})")
    return pd.DataFrame(seg.loc[:, ["ts"] + rt_cols])


def build_future_monthly_stats() -> pd.DataFrame:
    # ------------------------------------------------------------------
    # 1. 拼接 实测段（2026-08-01~14） + 去年同期仿真段（2025-08-15~31）
    # ------------------------------------------------------------------
    actual = _load_rt_segment(ACTUAL_PATH, ACTUAL_START, ACTUAL_END, "实测段 2026-08-01~14")
    sim = _load_rt_segment(HISTORY_PATH, SIM_LAST_YEAR_START, SIM_LAST_YEAR_END, "仿真段 2025-08-15~31（去年同期）")
    composed = pd.concat([actual, sim], ignore_index=True)
    expected_hours = 31 * 24
    if len(composed) != expected_hours:
        raise ValueError(f"拼接后 {len(composed)} 行 != 完整 8 月 {expected_hours} 行")

    # ------------------------------------------------------------------
    # 2. 计算逐时相对湿度 + 按整月统计（与 derive_weather_monthly 同口径）
    # ------------------------------------------------------------------
    valid = composed["rt_tt2"].notna() & composed["rt_dt"].notna()
    composed["cal_rh"] = np.nan
    composed.loc[valid, "cal_rh"] = _calc_rh(composed.loc[valid, "rt_tt2"], composed.loc[valid, "rt_dt"])

    row = {
        "ts": FUTURE_MONTH_END,
        "rt_tt2": composed["rt_tt2"].mean(),        # 月均气温（占白名单位）
        "rt_tt2_max": composed["rt_tt2"].max(),     # 月最高气温
        "rt_tt2_min": composed["rt_tt2"].min(),     # 月最低气温
        "cal_rh": composed["cal_rh"].mean(),        # 月均湿度
        "rt_ssr": composed["rt_ssr"].sum(),         # 月总辐射
        "rt_ws10": composed["rt_ws10"].mean(),      # 月均风速
        "rt_dt": composed["rt_dt"].mean(),          # 月均露点（框架会用 tt2+dt 重算 RH）
    }
    monthly = pd.DataFrame([row])
    monthly["ts"] = pd.to_datetime(monthly["ts"])

    # ------------------------------------------------------------------
    # 3. 输出
    # ------------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"future monthly weather stats saved: {OUTPUT_PATH}")
    print(monthly.to_string(index=False))
    return monthly


if __name__ == "__main__":
    build_future_monthly_stats()
