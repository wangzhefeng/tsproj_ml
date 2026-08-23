# -*- coding: utf-8 -*-
"""构建 ESS 5min 派生天气的严格 history/backtest/future 三文件。

数据边界：
- history 截止 2026-07-28 23:55，只使用 rt_* 实测；
- backtest 使用历史文件中每个预测日的 pred_*，携带 source_ts/available_at；
- future 从 2026-07-29 开始，只使用 pred_*，并声明在预测原点前可用。

首次运行兼容旧原始文件名，并将其中 07-29～07-31 段迁入新的 future 原始文件；
写入新文件后删除旧边界文件，避免两个信息集并存。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "dataset/aidc_ess_selfuse_load/exogenous_weather_raw"

RAW_HISTORY_PATH = DATA_DIR / "weather_in_20250101_20260728.csv"
RAW_FUTURE_PATH = DATA_DIR / "weather_future_in_20260729_20260814.csv"
LEGACY_RAW_HISTORY_PATH = DATA_DIR / "weather_in_20250101_20260731.csv"
LEGACY_RAW_FUTURE_PATH = DATA_DIR / "weather_future_in_20260801_20260814.csv"

HISTORY_OUTPUT_PATH = DATA_DIR / "weather_derived_in_20250101_20260728.csv"
BACKTEST_OUTPUT_PATH = DATA_DIR / "weather_derived_backtest_forecast_20260628_20260728.csv"
FUTURE_OUTPUT_PATH = DATA_DIR / "weather_derived_future_20260729_20260814.csv"
LEGACY_DERIVED_PATHS = (
    DATA_DIR / "weather_derived_in_20250101_20260731.csv",
    DATA_DIR / "weather_derived_future_20260801_20260814.csv",
)

HISTORY_START = pd.Timestamp("2025-01-01 00:00:00")
HISTORY_END = pd.Timestamp("2026-07-28 23:55:00")
BACKTEST_START = pd.Timestamp("2026-06-28 00:00:00")
BACKTEST_END = HISTORY_END
FUTURE_START = pd.Timestamp("2026-07-29 00:00:00")
FUTURE_END = pd.Timestamp("2026-08-14 23:55:00")
FUTURE_AVAILABLE_AT = pd.Timestamp("2026-07-28 23:55:00")
SHORT_GAP_LIMIT = 5

WEATHER_COLS = [
    "rt_ssr",
    "rt_tt2",
    "cal_rh",
    "rt_ws10",
    "tt2_mean_3h",
    "tt2_diff_1h",
    "ssr_mean_3h",
]


def _calc_rh(tt2_k: pd.Series, dt_k: pd.Series) -> pd.Series:
    """Magnus-Tetens 公式：Kelvin 气温/露点 → 相对湿度百分比。"""
    t_air = tt2_k - 273.15
    t_dew = dt_k - 273.15
    e_s_td = 6.1078 * np.exp((17.2693 * t_dew) / (237.29 + t_dew))
    e_s_t = 6.1078 * np.exp((17.2693 * t_air) / (237.29 + t_air))
    return pd.Series(np.clip((e_s_td / e_s_t) * 100, 0, 100), index=tt2_k.index)



def _regularize_hourly(df: pd.DataFrame, required_cols: list[str], label: str) -> pd.DataFrame:
    """规则化严格 1h 网格：短缺口时间插值，长缺口同小时插值兜底。"""
    frame = df.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], errors="raise")
    frame = frame.sort_values("ts").drop_duplicates("ts", keep="last").set_index("ts")
    missing_cols = [column for column in required_cols if column not in frame.columns]
    if missing_cols:
        raise ValueError(f"{label}: 原始天气缺少必需列 {missing_cols}")
    for column in required_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    hourly_index = pd.date_range(frame.index.min(), frame.index.max(), freq="1h")
    frame = frame.reindex(hourly_index)
    missing_any = frame[required_cols].isna().any(axis=1)
    run_group = missing_any.ne(missing_any.shift()).cumsum()
    missing_runs = missing_any.groupby(run_group).sum()
    max_gap = int(missing_runs.max()) if not missing_runs.empty else 0

    filled = frame[required_cols].interpolate(
        method="time",
        limit=SHORT_GAP_LIMIT,
        limit_direction="both",
    )
    hours = pd.Series(filled.index, index=filled.index).dt.hour.to_numpy()
    for column in required_cols:
        series = filled[column].copy()
        for hour in range(24):
            mask = hours == hour
            series.loc[mask] = series.loc[mask].interpolate(method="time", limit_direction="both")
        filled[column] = series.ffill().bfill()
    if filled.isna().to_numpy().any():
        raise ValueError(f"{label}: 小时网格填充后仍有缺失值")
    print(f"  {label}: hourly_rows={len(filled)}, max_gap={max_gap}h")
    return filled


def _build_derived(df: pd.DataFrame, col_map: dict[str, str], label: str) -> pd.DataFrame:
    """在 1h 原生网格计算派生量，再 ffill 到完整 5min 网格。"""
    required_cols = list(col_map)
    if "cal_rh" not in col_map.values():
        required_cols.append("rt_dt")
    hourly = _regularize_hourly(df, required_cols, label)

    derived = pd.DataFrame(index=hourly.index)
    for raw_col, output_col in col_map.items():
        derived[output_col] = hourly[raw_col]
    if "cal_rh" not in derived.columns:
        derived["cal_rh"] = _calc_rh(hourly["rt_tt2"], hourly["rt_dt"])

    derived["tt2_mean_3h"] = derived["rt_tt2"].rolling(3, min_periods=1).mean()
    derived["tt2_diff_1h"] = derived["rt_tt2"].diff(1)
    derived["ssr_mean_3h"] = derived["rt_ssr"].rolling(3, min_periods=1).mean()

    index_5min = pd.date_range(
        derived.index.min(),
        derived.index.max() + pd.Timedelta(minutes=55),
        freq="5min",
    )
    result = derived.reindex(index_5min).ffill().bfill()
    result.index.name = "time"
    result = result.reset_index()[["time", *WEATHER_COLS]]
    if result[WEATHER_COLS].isna().to_numpy().any():
        raise ValueError(f"{label}: 5min 派生天气仍有缺失值")
    return result


def _build_historical_forecast(
    raw_history: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """由历史 pred_* 构造每个CV预测日当时可用的天气预测。"""
    derived = _build_derived(
        raw_history,
        {
            "pred_ssrd": "rt_ssr",
            "pred_tt2": "rt_tt2",
            "pred_rh": "cal_rh",
            "pred_ws10": "rt_ws10",
        },
        "historical_forecast",
    )
    derived = derived[
        (derived["time"] >= start) & (derived["time"] <= end)
    ].reset_index(drop=True)
    expected = pd.date_range(start, end, freq="5min")
    if not pd.DatetimeIndex(derived["time"]).equals(expected):
        missing = expected.difference(pd.DatetimeIndex(derived["time"]))
        preview = ", ".join(str(ts) for ts in missing[:10])
        raise ValueError(f"Historical forecast 缺少 {len(missing)} 个目标点：{preview}")
    derived["weather_source"] = "historical_forecast"
    derived["source_ts"] = derived["time"]
    derived["available_at"] = derived["time"].dt.normalize() - pd.Timedelta(minutes=5)
    return derived


def _load_and_split_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取新边界文件；首次运行时从旧文件迁移 07-29～07-31。"""
    if RAW_HISTORY_PATH.exists() and RAW_FUTURE_PATH.exists():
        history = pd.read_csv(RAW_HISTORY_PATH)
        future = pd.read_csv(RAW_FUTURE_PATH)
    else:
        if not LEGACY_RAW_HISTORY_PATH.exists() or not LEGACY_RAW_FUTURE_PATH.exists():
            raise FileNotFoundError("ESS weather raw source files are incomplete.")
        old_history = pd.read_csv(LEGACY_RAW_HISTORY_PATH)
        old_future = pd.read_csv(LEGACY_RAW_FUTURE_PATH)
        old_history["ts"] = pd.to_datetime(old_history["ts"], errors="raise")
        old_future["ts"] = pd.to_datetime(old_future["ts"], errors="raise")
        history = old_history[old_history["ts"] <= HISTORY_END.floor("1h")].copy()
        moved = old_history[
            (old_history["ts"] >= FUTURE_START)
            & (old_history["ts"] <= FUTURE_END.floor("1h"))
        ].copy()
        future = pd.concat([moved, old_future], ignore_index=True)
        future = future.sort_values("ts").drop_duplicates("ts", keep="last")
        history.to_csv(RAW_HISTORY_PATH, index=False)
        future.to_csv(RAW_FUTURE_PATH, index=False)

    for frame in (history, future):
        frame["ts"] = pd.to_datetime(frame["ts"], errors="raise")
    if history["ts"].max() > HISTORY_END:
        raise ValueError("Raw history contains timestamps after 2026-07-28.")
    expected_future_hours = pd.date_range(FUTURE_START, FUTURE_END.floor("1h"), freq="1h")
    missing = expected_future_hours.difference(pd.DatetimeIndex(future["ts"]))
    if len(missing):
        raise ValueError(f"Raw future missing {len(missing)} hourly timestamps.")
    return history, future


def _remove_legacy_outputs() -> None:
    for path in (*LEGACY_DERIVED_PATHS, LEGACY_RAW_HISTORY_PATH, LEGACY_RAW_FUTURE_PATH):
        if path.exists() and path not in {RAW_HISTORY_PATH, RAW_FUTURE_PATH}:
            path.unlink()


def main() -> None:
    raw_history, raw_future = _load_and_split_raw()
    history_raw = raw_history[
        (raw_history["ts"] >= HISTORY_START)
        & (raw_history["ts"] <= HISTORY_END.floor("1h"))
    ].copy()
    future_raw = raw_future[
        (raw_future["ts"] >= FUTURE_START)
        & (raw_future["ts"] <= FUTURE_END.floor("1h"))
    ].copy()

    history = _build_derived(
        history_raw,
        {"rt_ssr": "rt_ssr", "rt_tt2": "rt_tt2", "rt_ws10": "rt_ws10"},
        "history_actual",
    )
    history = history[
        (history["time"] >= HISTORY_START) & (history["time"] <= HISTORY_END)
    ].reset_index(drop=True)

    backtest = _build_historical_forecast(history_raw, BACKTEST_START, BACKTEST_END)

    future = _build_derived(
        future_raw,
        {
            "pred_ssrd": "rt_ssr",
            "pred_tt2": "rt_tt2",
            "pred_rh": "cal_rh",
            "pred_ws10": "rt_ws10",
        },
        "future_forecast",
    )
    future = future[
        (future["time"] >= FUTURE_START) & (future["time"] <= FUTURE_END)
    ].reset_index(drop=True)
    future["weather_source"] = "forecast"
    future["source_ts"] = future["time"]
    future["available_at"] = FUTURE_AVAILABLE_AT

    HISTORY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(HISTORY_OUTPUT_PATH, index=False)
    backtest.to_csv(BACKTEST_OUTPUT_PATH, index=False)
    future.to_csv(FUTURE_OUTPUT_PATH, index=False)
    _remove_legacy_outputs()

    print(f"history:  rows={len(history)}, range={history['time'].min()} -> {history['time'].max()}")
    print(f"backtest: rows={len(backtest)}, range={backtest['time'].min()} -> {backtest['time'].max()}")
    print(f"future:   rows={len(future)}, range={future['time'].min()} -> {future['time'].max()}")


if __name__ == "__main__":
    main()
