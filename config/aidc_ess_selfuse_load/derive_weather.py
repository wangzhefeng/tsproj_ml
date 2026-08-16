# -*- coding: utf-8 -*-
"""exogenous_weather_raw/ 下 derived 版本气象数据处理。

将 1h 采样的原始气象数据（weather_in/weather_future_in）在 1h 原生网格上计算聚合派生特征，
再 ffill 升采样到 5min 全网格，供 custom_features 注册表使用。

处理规则：
  1. 保留 4 列有物理关联的原始特征：rt_ssr（辐射）、rt_tt2（温度）、cal_rh（湿度）、rt_ws10（风速）
     删除 rt_ps（气压，corr −0.09 无关联）、rt_rain（95% 为 0 接近常数列）
  2. 派生 3 列聚合特征（1h 原生网格计算，消除 1h→5min 插值伪影）：
     - tt2_mean_3h: 3h 滑动平均温度（热惯性 → 空调负荷滞后响应）
     - tt2_diff_1h: 1h 温度变化率（升温/降温趋势 → 空调负荷方向）
     - ssr_mean_3h: 3h 滑动平均辐射（日照累积 → 棚内温升）
  3. cal_rh 由 rt_tt2/rt_dt 经 Magnus-Tetens 公式计算（历史段）；未来段直接用 pred_rh
  4. ffill 升采样到 5min（阶跃保持，不产生线性插值的虚假中间值）

输入:
  weather_in_20251001_20260728.csv      （历史段，rt_ 列，1h 采样）
  weather_future_in_20260729_20260731.csv （未来段，pred_ 列，1h 采样）
输出:
  weather_derived_in_20251001_20260728.csv      （历史段，5min 全网格）
  weather_derived_future_20260729_20260731.csv  （未来段，5min 全网格）

用法（仓库根目录）：
    uv run python config/aidc_ess_selfuse_load/derive_weather.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "dataset/aidc_ess_selfuse_load/exogenous_weather_raw"
HISTORY_END = pd.Timestamp("2026-07-28 23:55:00")
FUTURE_END = pd.Timestamp("2026-07-31 23:55:00")


def _calc_rh(tt2_k: pd.Series, dt_k: pd.Series) -> pd.Series:
    """Magnus-Tetens 公式：rt_tt2/rt_dt 为 Kelvin，输出相对湿度 %"""
    t_air = tt2_k - 273.15
    t_dew = dt_k - 273.15
    e_s_td = 6.1078 * np.exp((17.2693 * t_dew) / (237.29 + t_dew))
    e_s_t = 6.1078 * np.exp((17.2693 * t_air) / (237.29 + t_air))
    return pd.Series(np.clip((e_s_td / e_s_t) * 100, 0, 100), index=tt2_k.index)


def _build_derived(df: pd.DataFrame, col_map: dict, label: str) -> pd.DataFrame:
    """在 1h 原生网格上构建保留列 + 派生列，然后 ffill 升采样到 5min。"""
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").drop_duplicates("ts").set_index("ts")

    out = pd.DataFrame(index=df.index)
    for raw_col, out_col in col_map.items():
        out[out_col] = pd.to_numeric(df[raw_col], errors="coerce")

    # cal_rh：历史段从 rt_tt2/rt_dt 计算，未来段直接用 pred_rh
    if "cal_rh" not in out.columns or out["cal_rh"].isna().all():
        if "rt_dt" in df.columns:
            out["cal_rh"] = _calc_rh(
                pd.to_numeric(df["rt_tt2"], errors="coerce"),
                pd.to_numeric(df["rt_dt"], errors="coerce"),
            )

    # 派生聚合特征（1h 原生网格）
    out["tt2_mean_3h"] = out["rt_tt2"].rolling(3, min_periods=1).mean()
    out["tt2_diff_1h"] = out["rt_tt2"].diff(1)
    out["ssr_mean_3h"] = out["rt_ssr"].rolling(3, min_periods=1).mean()

    # 升采样到 5min：ffill 阶跃保持
    # 1h 网格末点为整点（如 23:00），需延展 55min 覆盖到 23:55，否则末段 11 个 5min 点缺失
    idx5 = pd.date_range(out.index.min(), out.index.max() + pd.Timedelta(minutes=55), freq="5min")
    out5 = out.reindex(idx5).ffill().bfill().fillna(0.0)
    out5.index.name = "time"

    print(f"  {label}: {len(df)} 行(1h) → {len(out5)} 行(5min), NaN={out5.isna().sum().sum()}")
    return out5


if __name__ == "__main__":
    # 历史段：rt_ 列
    hist_map = {
        "rt_ssr": "rt_ssr",
        "rt_tt2": "rt_tt2",
        "rt_ws10": "rt_ws10",
    }
    w_in = pd.read_csv(DATA_DIR / "weather_in_20251001_20260728.csv")
    hist = _build_derived(w_in, hist_map, "history")
    hist = hist[hist.index <= HISTORY_END]
    hist.to_csv(DATA_DIR / "weather_derived_in_20251001_20260728.csv")
    print(f"  -> weather_derived_in_20251001_20260728.csv: {len(hist)} 行")

    # 未来段：pred_ 列映射到 rt_ 语义
    fut_map = {
        "pred_ssrd": "rt_ssr",
        "pred_tt2": "rt_tt2",
        "pred_ws10": "rt_ws10",
        "pred_rh": "cal_rh",
    }
    w_fut = pd.read_csv(DATA_DIR / "weather_future_in_20260729_20260731.csv")
    fut = _build_derived(w_fut, fut_map, "future")
    fut = fut[fut.index <= FUTURE_END]
    fut.to_csv(DATA_DIR / "weather_derived_future_20260729_20260731.csv")
    print(f"  -> weather_derived_future_20260729_20260731.csv: {len(fut)} 行")

    print("Done.")
