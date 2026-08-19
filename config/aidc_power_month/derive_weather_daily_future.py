# -*- coding: utf-8 -*-
"""2026-08 未来日度气象统计构造：前半月实测 + 后半月去年同期仿真。

为 freq_1day 场景的 is_forecasting 提供未来 30 天（2026-08-01 ~ 08-30，
predict_steps=30、date_range inclusive="left" 的精确未来索引）的日度气象特征：
  - 2026-08-01 ~ 08-14：weather_future_in_20260801_20260814.csv 的 rt_ 实测列
    （生成该文件时这段已成过去，rt_ 列已由实测回填，优于 pred_ 预报列）
  - 2026-08-15 ~ 08-30：weather_in_20250101_20260731.csv 的 2025-08-15 ~ 08-30
    去年同期逐日替换（整日取自去年同日期，保持日内相关结构）

为什么不用 pred_ 预报列：历史逐时配对统计显示 pred_ws10 相对 rt_ws10 系统性
高估约 2 倍、pred_ssrd 高估 1.1~1.2 倍（2026-08-18 分析），rt_ 口径优先。

统计口径与 derive_weather_daily.py 完全一致（日均/最高/最低气温、Magnus-Tetens
逐时湿度取日均、日总辐射、日均风速、日均露点），输出 schema 相同
（ts, rt_tt2, rt_tt2_max, rt_tt2_min, cal_rh, rt_ssr, rt_ws10, rt_dt），
ts = 当日 00:00，列名复用框架 weather 白名单 → extend_future_weather_feature
白名单直连分支按时间戳 merge，无需 pred_ 映射。

注意：未来文件任何白名单列 NaN 会导致整行被框架丢弃（dropna 行为），
故统计后对缺失日做保险处理：气温三兄弟/风速/露点用前一日值回填（日度
连续性强），cal_rh 缺失时由回填后的 tt2/dt 重算，rt_ssr 用去年同期值补。

用法（仓库根目录）：
    uv run python config/aidc_power_month/derive_weather_daily_future.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ACTUAL_PATH = PROJECT_ROOT / "dataset/aidc_load_5min/weather_future_in_20260801_20260814.csv"
HISTORY_PATH = PROJECT_ROOT / "dataset/aidc_load_5min/weather_in_20250101_20260731.csv"
OUTPUT_PATH = PROJECT_ROOT / "dataset/aidc_power_month/freq_1day/weather_daily_stats_future_20260801_20260830.csv"

# 实测段 / 仿真段的日界
ACTUAL_START, ACTUAL_END = "2026-08-01", "2026-08-14"
SIM_LAST_YEAR_START, SIM_LAST_YEAR_END = "2025-08-15", "2025-08-30"
# 未来窗口总天数（= freq_1day predict_steps，date_range inclusive="left" 上界 08-31）
FUTURE_DAYS = 30


def _calc_rh(tt2_k: pd.Series, dt_k: pd.Series) -> pd.Series:
    """Magnus-Tetens 公式：rt_tt2/rt_dt 为 Kelvin，输出相对湿度 %。"""
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
    missing_mask = seg[rt_cols].isna().to_numpy().any(axis=1)
    n_missing = int(missing_mask.sum())
    if n_missing:
        print(f"  [warn] {label}: {n_missing} 行含缺失 rt_ 值（统计时跳过）")
    print(f"  {label}: {len(seg)} 行 ({seg['ts'].min()} -> {seg['ts'].max()})")
    return pd.DataFrame(seg.loc[:, ["ts"] + rt_cols])


def _aggregate_daily(seg: pd.DataFrame, label: str) -> pd.DataFrame:
    """单段 1h 数据 → 日统计（口径与 derive_weather_daily.py 一致）。"""
    valid = seg["rt_tt2"].notna() & seg["rt_dt"].notna()
    seg["cal_rh"] = np.nan
    seg.loc[valid, "cal_rh"] = _calc_rh(seg.loc[valid, "rt_tt2"], seg.loc[valid, "rt_dt"])

    seg = seg.set_index("ts")
    daily = pd.DataFrame(index=seg.resample("1D").mean().index)
    daily["rt_tt2"] = seg["rt_tt2"].resample("1D").mean()
    daily["rt_tt2_max"] = seg["rt_tt2"].resample("1D").max()
    daily["rt_tt2_min"] = seg["rt_tt2"].resample("1D").min()
    daily["cal_rh"] = seg["cal_rh"].resample("1D").mean()
    daily["rt_ssr"] = seg["rt_ssr"].resample("1D").sum()
    daily["rt_ws10"] = seg["rt_ws10"].resample("1D").mean()
    daily["rt_dt"] = seg["rt_dt"].resample("1D").mean()
    print(f"  {label} 日统计: {len(daily)} 天")
    return daily.reset_index()


def build_future_daily_stats() -> pd.DataFrame:
    # ------------------------------------------------------------------
    # 1. 实测段（2026-08-01~14）与去年同期仿真段（2025-08-15~30）各自聚合成日统计
    # ------------------------------------------------------------------
    actual = _aggregate_daily(
        _load_rt_segment(ACTUAL_PATH, ACTUAL_START, ACTUAL_END, "实测段 2026-08-01~14"),
        "实测段",
    )
    sim = _aggregate_daily(
        _load_rt_segment(HISTORY_PATH, SIM_LAST_YEAR_START, SIM_LAST_YEAR_END, "仿真段 2025-08-15~30（去年同期）"),
        "仿真段",
    )

    # 仿真段日期平移到 2026：2025-08-15 -> 2026-08-15
    sim["ts"] = sim["ts"] + pd.DateOffset(years=1)

    # ------------------------------------------------------------------
    # 平移后直接拼接。校验总天数 = 30
    # ------------------------------------------------------------------
    composed = pd.concat([actual, sim], ignore_index=True).sort_values("ts").reset_index(drop=True)
    if len(composed) != FUTURE_DAYS:
        raise ValueError(f"拼接后 {len(composed)} 天 != 未来窗口 {FUTURE_DAYS} 天")

    # ------------------------------------------------------------------
    # 2. 缺失日保险处理（未来文件 dropna 是整行丢弃，必须无 NaN）
    # ------------------------------------------------------------------
    wl_cols = ["rt_tt2", "rt_tt2_max", "rt_tt2_min", "cal_rh", "rt_ssr", "rt_ws10", "rt_dt"]
    nan_mask = composed[wl_cols].isna().to_numpy().any(axis=1)
    if nan_mask.any():
        days = composed.loc[nan_mask, "ts"].dt.strftime("%Y-%m-%d").tolist()
        print(f"  [fix] 缺失日 {days}：连续统计用前一日回填，rt_ssr 用去年同期原值")
        # 连续性统计量：前一日值回填（bfill 兜底首行）
        for col in ["rt_tt2", "rt_tt2_max", "rt_tt2_min", "rt_ws10", "rt_dt"]:
            composed[col] = composed[col].ffill().bfill()
        # cal_rh 用回填后的 tt2/dt 重算
        need_rh = composed["cal_rh"].isna()
        if bool(need_rh.any()):
            composed.loc[need_rh, "cal_rh"] = _calc_rh(
                composed.loc[need_rh, "rt_tt2"], composed.loc[need_rh, "rt_dt"]
            ).to_numpy()
        # rt_ssr 前一日回填
        composed["rt_ssr"] = composed["rt_ssr"].ffill().bfill()

    # ------------------------------------------------------------------
    # 3. 输出
    # ------------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    composed.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"future daily weather stats saved: {OUTPUT_PATH}")
    print(f"rows={len(composed)}, range={composed['ts'].min().date()} -> {composed['ts'].max().date()}")
    print(composed.to_string(index=False))
    return composed


if __name__ == "__main__":
    build_future_daily_stats()
