# -*- coding: utf-8 -*-
"""aidc_load_5min 的 1h 气象数据 → 日频统计特征聚合。

将 dataset/aidc_load_5min/weather_in_20250101_20260731.csv（1h 采样，rt_ 实测列）
按日聚合出统计特征，ts 标签 = 当日 00:00:00（与 freq_1day 目标序列的时间标签
对齐，供框架 weather 通路按精确时间戳 merge）。

设计要点（与 derive_weather_monthly.py 同模式，统计粒度从月改为日）：
  1. 历史段只用 rt_ 实测列（不混入 pred_ 预测列，避免滑窗测试信息穿越）
  2. 统计特征按物理意义选取（数据中心空调/通风负荷驱动因素）：
     - rt_tt2 / rt_tt2_max / rt_tt2_min : 日均/最高/最低 2m 气温（K）
     - cal_rh                           : 日均相对湿度（%）
     - rt_ssr                           : 日总辐射（逐时累加）
     - rt_ws10                          : 日均 10m 风速
  3. cal_rh 由 rt_tt2/rt_dt 经 Magnus-Tetens 公式逐时计算后取日均值
  4. 列名复用框架 weather 通路白名单（rt_tt2/cal_rh/rt_ssr/rt_ws10），统计列
     直接用白名单名 = 走原生 weather merge 通路，无需改框架代码；
     mean 占白名单位，max/min 变体保留在输出 CSV 供 custom_features 使用
  5. rt_dt（日均露点）必须携带：框架 extend_weather_feature 无条件用
     rt_tt2+rt_dt 重算 cal_rh，缺失该列会使 cal_rh 变 NaN

数据质量（2025-10-01 后模型可见区间）：304 天中仅 5 天含 1~3 小时缺失，
无全天缺失；mean/max/min 口径天然跳过 NaN。源文件 2025 年存在 4 段全天
缺测（03-01~09、06-19~23、07-11~15、07-19，共 20 天）→ 日统计全列 NaN，
由本脚本的镜像年回填补齐（优先去年同日历日期；去年无数据时回落次年，
2025 缺测日的镜像取自 2026 年同月日，均已核实完整）。rt_ssr 的 sum 聚合
带 min_count=1：全天缺测日显式为 NaN 再回填，避免 pandas sum() 的假 0。
当前模型 history_length 只取 2025-10 之后，填补段暂不进模型——为的是
未来扩展 history 或人工分析时数据完整。

输入:
  dataset/aidc_load_5min/weather_in_20250101_20260731.csv
输出:
  dataset/aidc_power_month/freq_1day/weather_daily_stats_20251001_20260731.csv
  （文件名为目标序列对齐区间；内容为源文件全量 2025-01-01 起的逐日统计，
   与月度脚本的全量输出行为一致）

用法（仓库根目录）：
    env -u PYTHONPATH .venv/bin/python config/aidc_power_month/derive_weather_daily.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_PATH = PROJECT_ROOT / "dataset/aidc_load_5min/weather_in_20250101_20260731.csv"
OUTPUT_PATH = PROJECT_ROOT / "dataset/aidc_power_month/freq_1day/weather_daily_stats_20251001_20260731.csv"


def _calc_rh(tt2_k: pd.Series, dt_k: pd.Series) -> pd.Series:
    """Magnus-Tetens 公式：rt_tt2/rt_dt 为 Kelvin，输出相对湿度 %。"""
    t_air = tt2_k - 273.15
    t_dew = dt_k - 273.15
    e_s_td = 6.1078 * np.exp((17.2693 * t_dew) / (237.29 + t_dew))
    e_s_t = 6.1078 * np.exp((17.2693 * t_air) / (237.29 + t_air))
    return pd.Series(np.clip((e_s_td / e_s_t) * 100, 0, 100), index=tt2_k.index)


def build_daily_stats() -> pd.DataFrame:
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
    # 2. 按日聚合统计特征（resample 1D，标签=当日 00:00）
    # ------------------------------------------------------------------
    df = df.set_index("ts")
    daily = pd.DataFrame(index=df.resample("1D").mean().index)
    daily["rt_tt2"] = df["rt_tt2"].resample("1D").mean()          # 日均气温（占白名单位）
    daily["rt_tt2_max"] = df["rt_tt2"].resample("1D").max()       # 日最高气温
    daily["rt_tt2_min"] = df["rt_tt2"].resample("1D").min()       # 日最低气温
    daily["cal_rh"] = df["cal_rh"].resample("1D").mean()          # 日均湿度
    daily["rt_ssr"] = df["rt_ssr"].resample("1D").sum(min_count=1)  # 日总辐射（全天缺测=NaN，防止假 0）
    daily["rt_ws10"] = df["rt_ws10"].resample("1D").mean()        # 日均风速
    # rt_dt（日均露点）：框架 extend_weather_feature 无条件用 rt_tt2+rt_dt 重算 cal_rh，
    # 统计文件必须携带 rt_dt；日均露点+日均气温重算的 RH 与逐时 RH 日均值的差异可忽略
    daily["rt_dt"] = df["rt_dt"].resample("1D").mean()            # 日均露点

    daily = daily.reset_index()

    # ------------------------------------------------------------------
    # 3. 全天缺测日回填：镜像年（同日历日期，优先去年、无数据回落次年）
    # ------------------------------------------------------------------
    stat_cols = ["rt_tt2", "rt_tt2_max", "rt_tt2_min", "cal_rh", "rt_ssr", "rt_ws10", "rt_dt"]
    daily = daily.set_index("ts")
    nan_mask = daily[stat_cols].isna().to_numpy().any(axis=1) | (daily[stat_cols] == 0).to_numpy().any(axis=1)
    filled_dates = []
    for ts in daily.index[nan_mask]:
        for offset_years in (-1, 1):
            mirror = ts + pd.DateOffset(years=offset_years)
            # 2/29 镜像不存在时顺延一天
            if mirror not in daily.index:
                mirror = mirror + pd.Timedelta(days=1)
            if mirror in daily.index and not bool(daily.loc[mirror, stat_cols].isna().any()):
                daily.loc[ts] = daily.loc[mirror, stat_cols].to_numpy()
                filled_dates.append(f"{ts.date()} <- {mirror.date()}")
                break
        else:
            print(f"  [warn] {ts.date()} 无可用镜像年数据，保持缺失")
    if filled_dates:
        print(f"  [fill] 镜像年回填 {len(filled_dates)} 天:")
        for line in filled_dates:
            print(f"    {line}")
    daily = daily.reset_index()

    # ------------------------------------------------------------------
    # 4. 输出
    # ------------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"daily weather stats saved: {OUTPUT_PATH}")
    print(f"rows={len(daily)}, range={daily['ts'].min()} -> {daily['ts'].max()}")
    nan_days = int(daily[["rt_tt2", "cal_rh", "rt_ssr", "rt_ws10"]].isna().to_numpy().any(axis=1).sum())
    print(f"rows with NaN in whitelist cols: {nan_days} (2025-01~09 缺失段)")
    return daily


if __name__ == "__main__":
    build_daily_stats()
