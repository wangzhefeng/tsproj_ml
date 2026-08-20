# -*- coding: utf-8 -*-
"""从当前场景 A/B 15min 负荷派生预测原点可用的因果负荷状态特征。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIO = "aidc_load_15min_daily"
DATA_DIR = PROJECT_ROOT / "dataset" / SCENARIO
SOURCE_FILES = {
    "A": "A_Loads_15min_mean_20251001_20260731.csv",
    "B": "B_Loads_15min_mean_20251001_20260731.csv",
}
OUTPUT_DIR = DATA_DIR / "load_state_features"

LOAD_STATE_FEATURE_COLUMNS = [
    "state_roll_1h_mean",
    "state_roll_1h_std",
    "state_roll_4h_mean",
    "state_roll_4h_std",
    "state_roll_24h_mean",
    "state_roll_24h_std",
    "state_roll_24h_range",
    "state_roll_7d_mean",
    "state_roll_7d_std",
    "state_diff_15min",
    "state_diff_1h",
    "state_diff_24h_pct",
    "state_robust_z_7d",
    "state_weekly_base_dev_pct",
    "state_route_diff_pct",
]


def _load_routes() -> dict[str, pd.DataFrame]:
    routes: dict[str, pd.DataFrame] = {}
    for route, filename in SOURCE_FILES.items():
        frame = pd.read_csv(DATA_DIR / filename, usecols=["time", "value"])
        frame["time"] = pd.to_datetime(frame["time"], errors="raise")
        frame["value"] = pd.to_numeric(frame["value"], errors="raise")
        frame = frame.sort_values("time").drop_duplicates(subset="time", keep="last").reset_index(drop=True)
        if frame["time"].duplicated().any() or not frame["time"].is_monotonic_increasing:
            raise ValueError(f"Route {route} timestamps must be unique and increasing.")
        if len(frame) > 1 and not (frame["time"].diff().dropna() == pd.Timedelta(minutes=15)).all():
            raise ValueError(f"Route {route} must have a continuous 15min timeline.")
        routes[route] = frame
    if not routes["A"]["time"].equals(routes["B"]["time"]):
        raise ValueError("Route A/B timestamps are not aligned.")
    return routes


def _safe_pct(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    scale = denominator.abs().replace(0.0, np.nan)
    return numerator / scale * 100.0


def build_load_state_features(own: pd.DataFrame, peer: pd.DataFrame) -> pd.DataFrame:
    """所有特征仅使用当前及历史点；配置以 end_of_period 声明可用时刻。"""
    y = own["value"]
    output = pd.DataFrame({"time": own["time"]})

    roll_1h = y.rolling(4, min_periods=1)
    roll_4h = y.rolling(16, min_periods=1)
    roll_24h = y.rolling(96, min_periods=1)
    roll_7d = y.rolling(672, min_periods=1)
    output["state_roll_1h_mean"] = roll_1h.mean()
    output["state_roll_1h_std"] = roll_1h.std()
    output["state_roll_4h_mean"] = roll_4h.mean()
    output["state_roll_4h_std"] = roll_4h.std()
    output["state_roll_24h_mean"] = roll_24h.mean()
    output["state_roll_24h_std"] = roll_24h.std()
    output["state_roll_24h_range"] = roll_24h.max() - roll_24h.min()
    output["state_roll_7d_mean"] = roll_7d.mean()
    output["state_roll_7d_std"] = roll_7d.std()

    output["state_diff_15min"] = y.diff(1)
    output["state_diff_1h"] = y.diff(4)
    output["state_diff_24h_pct"] = _safe_pct(y - y.shift(96), y.shift(96))

    rolling_median = y.rolling(672, min_periods=96).median()
    abs_deviation = (y - rolling_median).abs()
    rolling_mad = abs_deviation.rolling(672, min_periods=96).median()
    robust_scale = (1.4826 * rolling_mad).replace(0.0, np.nan)
    output["state_robust_z_7d"] = (y - rolling_median) / robust_scale
    output["state_weekly_base_dev_pct"] = _safe_pct(y - y.shift(672), y.shift(672))
    output["state_route_diff_pct"] = _safe_pct(y - peer["value"], peer["value"])

    output[LOAD_STATE_FEATURE_COLUMNS] = (
        output[LOAD_STATE_FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return output[["time", *LOAD_STATE_FEATURE_COLUMNS]]


def main() -> None:
    routes = _load_routes()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for route, peer_route in (("A", "B"), ("B", "A")):
        state = build_load_state_features(routes[route], routes[peer_route])
        output = OUTPUT_DIR / f"{route}_load_state_history.csv"
        state.to_csv(output, index=False, encoding="utf-8")
        print(
            f"[{route}] rows={len(state)}, features={len(LOAD_STATE_FEATURE_COLUMNS)}, "
            f"range={state['time'].min()} -> {state['time'].max()}"
        )
        print(f"  -> {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
