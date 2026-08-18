# -*- coding: utf-8 -*-
"""aidc_load_5min 的 1h 气象数据 → 月频统计特征聚合。

将 dataset/aidc_load_5min/weather_in_20250101_20260731.csv（1h 采样，rt_ 实测列）
按月聚合出统计特征，ts 标签 = 月末 00:00:00（与 freq_1month 目标序列的时间标签
对齐，供框架 weather 通路按精确时间戳 merge）。

设计要点：
  1. 历史段只用 rt_ 实测列（不混入 pred_ 预测列，避免滑窗测试信息穿越）
  2. 统计特征按物理意义选取（数据中心空调/通风负荷驱动因素）：
     - rt_tt2_mean / rt_tt2_max / rt_tt2_min : 月均/最高/最低 2m 气温（℃）
     - cal_rh_mean                           : 月均相对湿度（%）
     - rt_ssr_sum                            : 月总辐射（日累积加和）
     - rt_ws10_mean                          : 月均 10m 风速
  3. cal_rh 由 rt_tt2/rt_dt 经 Magnus-Tetens 公式计算（列在原始文件中不存在）
  4. 列名复用框架 weather 通路的 6 个白名单名（rt_ssr/rt_ws10/rt_tt2/cal_rh/rt_ps/rt_rain）
     —— 框架 extend_weather_feature 只认这些名字，统计列直接用白名单名 =
     让月度统计值走原生 weather merge 通路，无需改框架代码。
     映射：rt_tt2_mean→rt_tt2, cal_rh_mean→cal_rh, rt_ws10_mean→rt_ws10, rt_ssr_sum→rt_ssr

注意：同一物理量不同统计口径（mean/max/min/sum）与白名单一对一冲突，
此处选择：月均值占白名单位（rt_tt2→mean），max/min/sum 变体保留在输出
CSV 中供 custom_features 或人工分析使用（框架 weather 通路只用白名单 4 列）。

输入:
  dataset/aidc_load_5min/weather_in_20250101_20260731.csv
输出:
  dataset/aidc_power_month/freq_1month/weather_monthly_stats_202510_202607.csv

用法（仓库根目录）：
    uv run python config/aidc_power_month/derive_weather_monthly.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_PATH = PROJECT_ROOT / "dataset/aidc_load_5min/weather_in_20250101_20260731.csv"
OUTPUT_PATH = PROJECT_ROOT / "dataset/aidc_power_month/freq_1month/weather_monthly_stats_202510_202607.csv"


def _calc_rh(tt2_k: pd.Series, dt_k: pd.Series) -> pd.Series:
    """Magnus-Tetens 公式：rt_tt2/rt_dt 为 Kelvin，输出相对湿度 %。"""
    t_air = tt2_k - 273.15
    t_dew = dt_k - 273.15
    e_s_td = 6.1078 * np.exp((17.2693 * t_dew) / (237.29 + t_dew))
    e_s_t = 6.1078 * np.exp((17.2693 * t_air) / (237.29 + t_air))
    return pd.Series(np.clip((e_s_td / e_s_t) * 100, 0, 100), index=tt2_k.index)


def build_monthly_stats() -> pd.DataFrame:
    # ------------------------------------------------------------------
    # 1. 加载 1h 气象，计算逐时相对湿度
    # ------------------------------------------------------------------
    df = pd.read_csv(SOURCE_PATH)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").drop_duplicates(subset="ts").reset_index(drop=True)

    for col in ["rt_tt2", "rt_dt", "rt_ssr", "rt_ws10"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = df["rt_tt2"].notna() & df["rt_dt"].notna()
    df["cal_rh"] = np.nan
    df.loc[valid, "cal_rh"] = _calc_rh(df.loc[valid, "rt_tt2"], df.loc[valid, "rt_dt"])

    # ------------------------------------------------------------------
    # 2. 按月聚合统计特征（resample 1ME，标签=月末）
    # ------------------------------------------------------------------
    df = df.set_index("ts")
    monthly = pd.DataFrame(index=df.resample("1ME").mean().index)
    monthly["rt_tt2"] = df["rt_tt2"].resample("1ME").mean()          # 月均气温（占白名单位）
    monthly["rt_tt2_max"] = df["rt_tt2"].resample("1ME").max()       # 月最高气温
    monthly["rt_tt2_min"] = df["rt_tt2"].resample("1ME").min()       # 月最低气温
    monthly["cal_rh"] = df["cal_rh"].resample("1ME").mean()          # 月均湿度
    monthly["rt_ssr"] = df["rt_ssr"].resample("1ME").sum()           # 月总辐射
    monthly["rt_ws10"] = df["rt_ws10"].resample("1ME").mean()        # 月均风速
    # rt_dt（月均露点）：框架 extend_weather_feature 无条件用 rt_tt2+rt_dt 重算 cal_rh，
    # 统计文件必须携带 rt_dt；月均露点+月均气温重算的 RH 与逐时 RH 月均值的差异 < 2%（近似）
    monthly["rt_dt"] = df["rt_dt"].resample("1ME").mean()            # 月均露点

    monthly = monthly.reset_index()
    monthly = monthly.rename(columns={"ts": "ts"})

    # ------------------------------------------------------------------
    # 3. 输出
    # ------------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"monthly weather stats saved: {OUTPUT_PATH}")
    print(f"rows={len(monthly)}, months={monthly['ts'].dt.strftime('%Y-%m').tolist()}")
    return monthly


if __name__ == "__main__":
    build_monthly_stats()
