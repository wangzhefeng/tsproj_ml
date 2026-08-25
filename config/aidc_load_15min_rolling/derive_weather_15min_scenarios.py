# -*- coding: utf-8 -*-
"""为 aidc_load_15min_rolling 场景构建严格信息集气象数据。

输入：
  dataset/aidc_load_5min/weather_in_20250101_20260731.csv（1h 实测）

输出（写入 dataset/aidc_load_15min_rolling/）：
  weather_15min_20250101_20260731T1345.csv
  weather_15min_backtest_proxy_20260101_20260731.csv
  weather_15min_future_proxy_20260731T1400_20260831.csv

口径：
  1. 历史实测：1h 数据先补齐缺测，再按时间线性插值到 15min；cal_rh 由
     插值后的 rt_tt2 / rt_dt 通过 Magnus-Tetens 公式逐点重算。
  2. 滑窗回测：目标时刻 t 使用上一年同月同日同一 15min 时刻的历史值；
     available_at 为目标月前一个月月末，保证严格早于目标月。
  3. 正式预测：2026-07-31 14:00~2026-08-31 使用 2025 年同期逐 15min
     代理；available_at 固定为 2026-07-31 00:00，覆盖 07-31 14:00 的
     intraday 预测原点。history 严格截止到预测原点前一时刻 13:45。
"""
from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_PATH = PROJECT_ROOT / "dataset/aidc_load_5min/weather_in_20250101_20260731.csv"
SCENARIO_DIR = PROJECT_ROOT / "dataset/aidc_load_15min_rolling"

HISTORY_NAME = "weather_15min_20250101_20260731T1345.csv"
BACKTEST_NAME = "weather_15min_backtest_proxy_20260101_20260731.csv"
FUTURE_NAME = "weather_15min_future_proxy_20260731T1400_20260831.csv"

HISTORY_START = pd.Timestamp("2025-01-01 00:00:00")
HISTORY_END = pd.Timestamp("2026-07-31 13:45:00")
BACKTEST_START = pd.Timestamp("2026-01-01 00:00:00")
BACKTEST_END = pd.Timestamp("2026-07-31 23:45:00")
FUTURE_START = pd.Timestamp("2026-07-31 14:00:00")
FUTURE_END = pd.Timestamp("2026-08-31 23:45:00")
FUTURE_AVAILABLE_AT = pd.Timestamp("2026-07-31 00:00:00")

SOURCE_COLS = ["rt_tt2", "rt_dt", "rt_ssr", "rt_ws10"]
WEATHER_COLS = ["rt_tt2", "cal_rh", "rt_ssr", "rt_ws10", "rt_dt"]


def _calc_rh(tt2_k: pd.Series, dt_k: pd.Series) -> pd.Series:
    """Magnus-Tetens 公式：Kelvin 气温/露点 → 相对湿度百分比。"""
    t_air = tt2_k - 273.15
    t_dew = dt_k - 273.15
    e_s_td = 6.1078 * np.exp((17.2693 * t_dew) / (237.29 + t_dew))
    e_s_t = 6.1078 * np.exp((17.2693 * t_air) / (237.29 + t_air))
    return pd.Series(np.clip((e_s_td / e_s_t) * 100, 0, 100), index=tt2_k.index)


def _same_clock_prior_year(ts: pd.Timestamp) -> pd.Timestamp:
    """上一年同月同日同时刻；闰日回落到 02-28。"""
    day = min(ts.day, calendar.monthrange(ts.year - 1, ts.month)[1])
    return ts.replace(year=ts.year - 1, day=day)


def _fill_hourly_source() -> pd.DataFrame:
    """规则化 1h 实测并补齐缺测；镜像年优先，时间插值兜底。"""
    source = pd.read_csv(SOURCE_PATH)
    source["ts"] = pd.to_datetime(source["ts"], errors="raise")
    source = source.sort_values("ts").drop_duplicates(subset="ts", keep="last")
    source = source.set_index("ts")

    for col in SOURCE_COLS:
        if col not in source.columns:
            raise ValueError(f"Weather source missing required column: {col}")
        source[col] = pd.to_numeric(source[col], errors="coerce")

    hourly_index = pd.date_range(HISTORY_START, HISTORY_END.floor("1h"), freq="1h")
    hourly = source[SOURCE_COLS].reindex(hourly_index)

    # 对成段缺测优先使用镜像年同月同日同时刻，避免跨季节长距离插值。
    for ts in hourly.index[hourly.isna().any(axis=1)]:
        for offset in (-1, 1):
            try:
                mirror = ts + pd.DateOffset(years=offset)
            except ValueError:
                mirror = ts.replace(month=2, day=28) + pd.DateOffset(years=offset)
            if mirror in hourly.index:
                for col in SOURCE_COLS:
                    if pd.isna(hourly.at[ts, col]) and pd.notna(hourly.at[mirror, col]):
                        hourly.at[ts, col] = hourly.at[mirror, col]

    hourly = hourly.interpolate(method="time", limit_direction="both").ffill().bfill()
    if hourly[SOURCE_COLS].isna().to_numpy().any():
        raise ValueError("Hourly weather still contains NaN after mirror-year and interpolation fill.")
    hourly.index.name = "ts"
    return hourly


def build_history_15min() -> pd.DataFrame:
    hourly = _fill_hourly_source()
    index_15min = pd.date_range(HISTORY_START, HISTORY_END, freq="15min")
    history = hourly.reindex(index_15min).interpolate(method="time").ffill().bfill()
    history["cal_rh"] = _calc_rh(history["rt_tt2"], history["rt_dt"])
    history = history.reset_index(names="ts")[["ts", *WEATHER_COLS]]
    if history[WEATHER_COLS].isna().to_numpy().any():
        raise ValueError("15min historical weather contains NaN values.")
    return history


def _build_prior_year_proxy(
    history: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    available_at_mode: str,
) -> pd.DataFrame:
    indexed = history.set_index("ts")
    records = []
    missing = []
    for target in pd.date_range(start, end, freq="15min"):
        source_ts = _same_clock_prior_year(target)
        if source_ts not in indexed.index:
            missing.append(target)
            continue
        if available_at_mode == "prior_month_end":
            available_at = (target.to_period("M") - 1).to_timestamp("M")
        elif available_at_mode == "fixed_forecast_origin":
            available_at = FUTURE_AVAILABLE_AT
        else:
            raise ValueError(f"Unsupported available_at_mode: {available_at_mode}")
        values = indexed.loc[source_ts, WEATHER_COLS]
        records.append({
            "ts": target,
            **values.to_dict(),
            "weather_source": "prior_year_proxy",
            "source_ts": source_ts,
            "available_at": available_at,
        })
    if missing:
        preview = ", ".join(str(ts) for ts in missing[:10])
        raise ValueError(f"Prior-year 15min weather missing {len(missing)} target timestamps: {preview}")
    proxy = pd.DataFrame(records)
    if proxy[WEATHER_COLS].isna().to_numpy().any():
        raise ValueError("15min weather proxy contains NaN values.")
    return proxy


def _write_outputs(history: pd.DataFrame, backtest: pd.DataFrame, future: pd.DataFrame) -> None:
    """只写当前场景的数据目录，不改写其他场景产物。"""
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    history.to_csv(SCENARIO_DIR / HISTORY_NAME, index=False, encoding="utf-8")
    backtest.to_csv(SCENARIO_DIR / BACKTEST_NAME, index=False, encoding="utf-8")
    future.to_csv(SCENARIO_DIR / FUTURE_NAME, index=False, encoding="utf-8")


def main() -> None:
    history = build_history_15min()
    backtest = _build_prior_year_proxy(
        history,
        BACKTEST_START,
        BACKTEST_END,
        available_at_mode="prior_month_end",
    )
    future = _build_prior_year_proxy(
        history,
        FUTURE_START,
        FUTURE_END,
        available_at_mode="fixed_forecast_origin",
    )
    _write_outputs(history, backtest, future)

    print(f"history:  rows={len(history)}, range={history['ts'].min()} -> {history['ts'].max()}")
    print(f"backtest: rows={len(backtest)}, range={backtest['ts'].min()} -> {backtest['ts'].max()}")
    print(f"future:   rows={len(future)}, range={future['ts'].min()} -> {future['ts'].max()}")
    print(f"saved: {SCENARIO_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
