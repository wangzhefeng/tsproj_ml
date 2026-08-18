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
  weather_in_20250101_20260731.csv（rt_ 实测列 + pred_ 预测列，1h 采样）
  weather_future_in_20260801_20260814.csv（pred_ 预测列，1h 采样）
输出:
  weather_derived_in_20250101_20260731.csv      （扩展历史，5min 全网格）
  weather_derived_future_20260801_20260814.csv  （扩展未来，5min 全网格）

用法（仓库根目录）：
    uv run python config/aidc_ess_selfuse_load/derive_weather.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "dataset/aidc_ess_selfuse_load/exogenous_weather_raw"
SOURCE_PATH = DATA_DIR / "weather_in_20250101_20260731.csv"
FUTURE_SOURCE_PATH = DATA_DIR / "weather_future_in_20260801_20260814.csv"
FULL_HISTORY_START = pd.Timestamp("2025-01-01 00:00:00")
FULL_HISTORY_END = pd.Timestamp("2026-07-31 23:55:00")
EXTENDED_FUTURE_START = pd.Timestamp("2026-08-01 00:00:00")
EXTENDED_FUTURE_END = pd.Timestamp("2026-08-14 23:55:00")
SHORT_GAP_LIMIT = 5


def _calc_rh(tt2_k: pd.Series, dt_k: pd.Series) -> pd.Series:
    """Magnus-Tetens 公式：rt_tt2/rt_dt 为 Kelvin，输出相对湿度 %"""
    t_air = tt2_k - 273.15
    t_dew = dt_k - 273.15
    e_s_td = 6.1078 * np.exp((17.2693 * t_dew) / (237.29 + t_dew))
    e_s_t = 6.1078 * np.exp((17.2693 * t_air) / (237.29 + t_air))
    return pd.Series(np.clip((e_s_td / e_s_t) * 100, 0, 100), index=tt2_k.index)


def _build_derived(df: pd.DataFrame, col_map: dict, label: str) -> pd.DataFrame:
    """在 1h 原生网格上构建保留列 + 派生列，然后 ffill 升采样到 5min。"""
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").drop_duplicates("ts").set_index("ts")

    # 先补齐严格 1h 网格，再计算 rolling/diff，避免原始缺时导致“3h/1h”
    # 实际跨越更长时间。短缺口先按时间插值；长缺口再按相同小时跨日插值，
    # 保留天气日内形态，避免把单一值连续 ffill 数天。
    hourly_index = pd.date_range(df.index.min(), df.index.max(), freq="1h")
    df = df.reindex(hourly_index)
    required_cols = list(col_map)
    if "cal_rh" not in col_map.values():
        required_cols.append("rt_dt")
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{label}: 原始天气缺少必需列 {missing_cols}")
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_any = pd.Series(
        df[required_cols].isna().to_numpy().any(axis=1),
        index=df.index,
        dtype=bool,
    )
    run_group = missing_any.ne(missing_any.shift()).cumsum()
    missing_runs = missing_any.groupby(run_group).sum()
    max_gap = int(missing_runs.max()) if not missing_runs.empty else 0
    filled = (
        df[required_cols]
        .interpolate(method="time", limit=SHORT_GAP_LIMIT, limit_direction="both")
    )
    for col in required_cols:
        series = filled[col].copy()
        series_hours = pd.Series(series.index, index=series.index).dt.hour.to_numpy()
        for hour in range(24):
            hour_mask = series_hours == hour
            series.loc[hour_mask] = series.loc[hour_mask].interpolate(
                method="time", limit_direction="both"
            )
        filled[col] = series.ffill().bfill()
    df.loc[:, required_cols] = filled
    remaining_nan = int(df[required_cols].isna().sum().sum())
    if remaining_nan:
        raise ValueError(f"{label}: 小时网格填充后仍有 {remaining_nan} 个缺失值")

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

    print(
        f"  {label}: {len(df)} 行(1h) → {len(out5)} 行(5min), "
        f"max_gap={max_gap}h, NaN={out5.isna().sum().sum()}"
    )
    return out5


if __name__ == "__main__":
    weather_all = pd.read_csv(SOURCE_PATH)
    weather_all["ts"] = pd.to_datetime(weather_all["ts"])
    weather_future_all = pd.read_csv(FUTURE_SOURCE_PATH)
    weather_future_all["ts"] = pd.to_datetime(weather_future_all["ts"])

    # 历史段：rt_ 列
    hist_map = {
        "rt_ssr": "rt_ssr",
        "rt_tt2": "rt_tt2",
        "rt_ws10": "rt_ws10",
    }

    # 扩展历史产物
    w_full_hist = pd.DataFrame(weather_all.loc[
        (weather_all["ts"] >= FULL_HISTORY_START)
        & (weather_all["ts"] <= FULL_HISTORY_END),
        :,
    ].copy())
    full_hist = _build_derived(w_full_hist, hist_map, "full_history")
    full_hist = full_hist[full_hist.index <= FULL_HISTORY_END]
    full_hist.to_csv(DATA_DIR / "weather_derived_in_20250101_20260731.csv")
    print(f"  -> weather_derived_in_20250101_20260731.csv: {len(full_hist)} 行")

    # 未来段：pred_ 列映射到 rt_ 语义
    fut_map = {
        "pred_ssrd": "rt_ssr",
        "pred_tt2": "rt_tt2",
        "pred_ws10": "rt_ws10",
        "pred_rh": "cal_rh",
    }

    # 扩展未来产物（2026-08-01~2026-08-14）
    w_extended_fut = pd.DataFrame(weather_future_all.loc[
        (weather_future_all["ts"] >= EXTENDED_FUTURE_START)
        & (weather_future_all["ts"] <= EXTENDED_FUTURE_END),
        :,
    ].copy())
    extended_fut = _build_derived(w_extended_fut, fut_map, "extended_future")
    extended_fut = extended_fut[extended_fut.index <= EXTENDED_FUTURE_END]
    extended_fut.to_csv(DATA_DIR / "weather_derived_future_20260801_20260814.csv")
    print(f"  -> weather_derived_future_20260801_20260814.csv: {len(extended_fut)} 行")

    print("Done.")
