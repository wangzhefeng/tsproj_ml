# -*- coding: utf-8 -*-

# ***************************************************
# * File        : liantong_august_prepare.py
# * Description : 联通 IT 8 月离线准备（真实日期 + 因果填补 + 完整历史天气）
# * 规则：保留全部真实时间戳，8-19 目标仅用此前数据估计，并显式标记。
# * 天气不平移；使用六列映射、湿度派生、夜间辐射填零和小时内 hold。
# * 填充后统一参与训练和评分；估计来源仅保留在离线 metadata。
# ***************************************************
"""离线数据准备入口；不修改原始 df_power.csv 与供应商天气宽表。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
SCENARIO_DIR = Path("dataset/aidc_electricity_computility/electricity/2026-08-31/liantong_IT")
DEFAULT_POWER = SCENARIO_DIR / "df_power.csv"
DEFAULT_WEATHER = Path("dataset/shared/weather/extracted/actual/weather_in_20250101_20260831.csv")
# argparse default 需要绝对路径（cwd 无关）
DEFAULT_POWER_ABS = PROJECT_ROOT / DEFAULT_POWER
DEFAULT_WEATHER_ABS = PROJECT_ROOT / DEFAULT_WEATHER
OUTPUT_DIR_ABS = PROJECT_ROOT / SCENARIO_DIR

GAP_DAY = pd.Timestamp("2026-08-19")
ONE_DAY = pd.Timedelta(days=1)
AUGUST_START = pd.Timestamp("2026-08-01")
SEPTEMBER_START = pd.Timestamp("2026-09-01")
AUGUST_END = pd.Timestamp("2026-08-31 23:55:00")


# 模型面天气列（与 aidc_ess_selfuse_load 接入方式一致）
RT_COLUMNS = ["rt_ssr", "rt_tt2", "cal_rh", "rt_ws10", "rt_ps", "rt_rain"]
PRED_COLUMNS = ["pred_ssrd", "pred_tt2", "pred_rh", "pred_ws10", "pred_ps", "pred_rain"]
NIGHT_HOURS = {20, 21, 22, 23, 0, 1, 2}


def fill_past_slot_mean(past: pd.Series) -> tuple[np.ndarray, dict]:
    """前 18 天同一 5min 时刻算术均值；不接收缺口之后的数据。"""
    expected = pd.date_range(AUGUST_START, GAP_DAY, freq="5min", inclusive="left")
    if not past.index.equals(expected) or not np.isfinite(past).all():
        raise ValueError("同槽均值填充需要 8-01 至 8-18 完整有限历史")
    daily = past.to_numpy(dtype=float).reshape(-1, 288)
    prediction = daily.mean(axis=0)
    audit = {
        "method": "past_18day_slot_mean",
        "history_days": 18,
        "usage": "normal training and scoring; no imputation mask",
        "fill_history_start": past.index[0].isoformat(),
        "fill_history_end": past.index[-1].isoformat(),
        "filled_start": GAP_DAY.isoformat(),
        "filled_end": (GAP_DAY + ONE_DAY - pd.Timedelta(minutes=5)).isoformat(),
        "filled_rows": 288,
        "estimated_not_observed": True,
    }
    return prediction, audit


def calc_rh(tt2_k: pd.Series, dt_k: pd.Series) -> pd.Series:
    """Magnus–Tetens 相对湿度；露点>气温的过饱和截断 100（与 build_scenario_weather 同合同）。"""
    t_air = tt2_k - 273.15
    t_dew = dt_k - 273.15
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        rh = 100.0 * np.exp(17.2693 * t_dew / (237.29 + t_dew) - 17.2693 * t_air / (237.29 + t_air))
    rh = pd.Series(rh, index=tt2_k.index, dtype=float)
    rh[(t_dew > t_air) & np.isfinite(rh)] = 100.0
    rh[~np.isfinite(rh)] = np.nan
    return rh


def resample_hold(hourly: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """小时 → 5min hold：仅在原生小时气泡内保持，缺口不传播。"""
    index = pd.date_range(hourly["ts"].iloc[0], end, freq="5min")
    out = hourly.set_index("ts").reindex(index.floor("1h"))
    out.index = index
    out.index.name = "ts"
    return out.reset_index()


def prepare_target(power_csv: Path) -> pd.DataFrame:
    power = pd.read_csv(power_csv, usecols=["time", "value"], float_precision="round_trip")
    times = pd.to_datetime(power["time"], errors="raise")
    expected = pd.date_range(AUGUST_START, AUGUST_END, freq="5min")
    if not pd.DatetimeIndex(times).equals(expected):
        raise ValueError("原始目标必须为 8 月完整真实 5min 时间网格")
    values = pd.to_numeric(power["value"], errors="raise")
    gap = times.dt.normalize().eq(GAP_DAY)
    if not values.isna().equals(gap):
        raise ValueError("期望仅 2026-08-19 整日缺失，不能填补其他位置或覆盖实测")
    if not np.isfinite(values[~gap]).all():
        raise ValueError("原始目标含非有限观测")
    past = pd.Series(values[times < GAP_DAY].to_numpy(), index=pd.DatetimeIndex(times[times < GAP_DAY]))
    filled, audit = fill_past_slot_mean(past)
    target = pd.DataFrame({"time": times, "value": values.copy()})
    target.loc[gap, "value"] = filled
    target.attrs["fill_audit"] = audit
    return target


def prepare_weather(weather_csv: Path) -> pd.DataFrame:
    weather = pd.read_csv(weather_csv)
    weather["ts"] = pd.to_datetime(weather["ts"], errors="raise", format="mixed")
    weather = weather.sort_values("ts").reset_index(drop=True)
    august = weather.loc[weather["ts"].between(AUGUST_START, SEPTEMBER_START, inclusive="left")].reset_index(drop=True)
    if not pd.DatetimeIndex(august["ts"]).equals(pd.date_range(AUGUST_START, periods=744, freq="1h")):
        raise ValueError("共享宽表 8 月切片不是完整 744 小时网格")
    weather = august
    raw_columns = [column for column in RT_COLUMNS if column != "cal_rh"] + ["rt_dt"]
    for column in (*raw_columns, *PRED_COLUMNS):
        if column not in weather.columns:
            raise ValueError(f"天气源缺少列 {column}")
        weather[column] = pd.to_numeric(weather[column], errors="raise")
    # 夜间辐射缺测记 0（20:00-02:00 物理定性）；其余已知列缺测直接拒绝
    night_missing = weather["rt_ssr"].isna() & weather["ts"].dt.hour.isin(NIGHT_HOURS)
    night_filled_times = [time.isoformat() for time in weather.loc[night_missing, "ts"]]
    weather.loc[night_missing, "rt_ssr"] = 0.0
    required_finite = weather[raw_columns].isna().any(axis=1)
    if required_finite.any():
        raise ValueError(f"rt_ 实测列存在非夜间缺测: {weather.loc[required_finite, 'ts'].tolist()}")
    if not np.isfinite(weather[[*raw_columns, *PRED_COLUMNS]]).all(axis=None):
        raise ValueError("天气模型依赖列存在缺失/非有限值")
    weather["cal_rh"] = calc_rh(weather["rt_tt2"], weather["rt_dt"])
    if weather["cal_rh"].isna().any():
        raise ValueError("cal_rh 派生后仍含缺失")
    hourly = weather[["ts", *RT_COLUMNS, *PRED_COLUMNS]].reset_index(drop=True)
    _assert_grid(hourly["ts"], "1h")
    fine = resample_hold(hourly, AUGUST_END)
    _assert_grid(fine["ts"], "5min")
    if fine[RT_COLUMNS].isna().any(axis=None):
        raise ValueError("5min 重采样后模型面 rt_/cal_rh 列含缺失")
    fine.attrs["weather_audit"] = {
        "timeline": "original timestamps; full historical training and test coverage",
        "night_rt_ssr_zero_filled_times": night_filled_times,
        "rules": "Magnus RH with supersaturation clipping; within-hour 5min hold",
    }
    repair_path = weather_csv.with_suffix(".six_features_repair.json")
    if repair_path.is_file():
        repair = json.loads(repair_path.read_text())
        if repair["source_sha256_after"] != hashlib.sha256(weather_csv.read_bytes()).hexdigest():
            raise ValueError("天气源修补记录与源文件哈希不匹配")
        fine.attrs["weather_audit"]["source_repair"] = {
            "path": str(repair_path),
            "sha256": hashlib.sha256(repair_path.read_bytes()).hexdigest(),
            "semantic_status": repair["semantic_status"],
        }
    return fine


def _assert_grid(times: pd.Series, freq: str) -> None:
    index = pd.DatetimeIndex(times)
    if index.duplicated().any() or not index.is_monotonic_increasing:
        raise ValueError("时间戳必须严格递增且无重复")
    expected = pd.date_range(index[0], index[-1], freq=freq)
    if not index.equals(expected):
        raise ValueError(f"时间网格不是连续 {freq}")


def publish(frame: pd.DataFrame, dest: Path, role: str, source: str) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False)
    meta = {
        "file": dest.name,
        "sha256_file": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "rows": len(frame),
        "source": source,
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "builder": "config/aidc_electricity_computility/electricity/2026-08-31/liantong_IT/liantong_august_prepare.py",
        "semantics_version": "liantong_august_original_time_18day_slot_mean_v3",
        "rules": "preserve original timestamps; target gap is explicitly estimated, not observed",
        "processing_audit": dict(frame.attrs),
        "role": role,
        "freq": "5min",
    }
    (dest.parent / (dest.stem + ".meta.json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return {"file": str(dest), "rows": len(frame), "sha256": meta["sha256_file"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power-csv", type=Path, default=DEFAULT_POWER_ABS)
    parser.add_argument("--weather-csv", type=Path, default=DEFAULT_WEATHER_ABS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_ABS)
    args = parser.parse_args()

    target = prepare_target(args.power_csv)
    history = prepare_weather(args.weather_csv)
    if target["time"].iloc[-1] != AUGUST_END or history["ts"].iloc[0] != target["time"].iloc[0]:
        raise ValueError("目标与天气真实时间轴不一致")
    report = [
        publish(target, args.output_dir / "target_power_5min_20260801_20260831.csv", "target", str(args.power_csv)),
        publish(history, args.output_dir / "weather_history_5min_20260801_20260831.csv", "history", str(args.weather_csv)),
    ]
    print(json.dumps({"status": "ok", "outputs": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
