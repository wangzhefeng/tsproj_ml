# -*- coding: utf-8 -*-
"""从完整事件分析表派生在线安全的预测原点负荷状态特征。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = Path("dataset/aidc_power_month/freq_1day/event_label_features")
OUTPUT_DIR = Path("dataset/aidc_power_month/freq_1day/load_state_features")
SOURCE_FILES = {
    "A": "A_Loads_1day_sum_20251001_20260731_labeled_features.csv",
    "B": "B_Loads_1day_sum_20251001_20260731_labeled_features.csv",
}

LOAD_STATE_FEATURE_COLUMNS = [
    "state_z30_robust",
    "state_z30_ready",
    "state_slope30",
    "state_slope30_ready",
    "state_intraday_std",
    "state_intraday_range",
    "state_intraday_p95_p5_gap",
    "state_intraday_cv",
    "state_intraday_max_abs_step",
    "state_intraday_peak_time_frac",
    "state_intraday_range_pct",
    "state_route_diff_pct",
    "state_last_day_volatile",
    "state_volatile_count_7d",
    "state_volatile_count_30d",
]

_SOURCE_TO_STATE = {
    "feat_z30_robust": "state_z30_robust",
    "feat_slope30": "state_slope30",
    "xf_intraday_std": "state_intraday_std",
    "xf_intraday_range": "state_intraday_range",
    "xf_intraday_p95_p5_gap": "state_intraday_p95_p5_gap",
    "xf_intraday_cv": "state_intraday_cv",
    "xf_intraday_max_abs_step": "state_intraday_max_abs_step",
    "xf_intraday_peak_time_frac": "state_intraday_peak_time_frac",
    "xf_intraday_range_pct": "state_intraday_range_pct",
    "xr_route_diff_pct": "state_route_diff_pct",
}


def build_load_state_features(frame: pd.DataFrame) -> pd.DataFrame:
    """保留预测原点可得状态；不输出目标值和非因果事件标签。"""
    required = ["time", "lbl_volatile_day", *_SOURCE_TO_STATE]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Load-state source missing columns: {missing}")

    output = pd.DataFrame({"time": pd.to_datetime(frame["time"])})
    output["state_z30_ready"] = frame["feat_z30_robust"].notna().astype(int)
    output["state_slope30_ready"] = frame["feat_slope30"].notna().astype(int)
    for source, target in _SOURCE_TO_STATE.items():
        output[target] = pd.to_numeric(frame[source], errors="coerce")

    volatile = pd.Series(
        pd.to_numeric(frame["lbl_volatile_day"], errors="coerce"),
        index=frame.index,
        dtype=float,
    ).fillna(0.0)
    output["state_last_day_volatile"] = volatile
    output["state_volatile_count_7d"] = volatile.rolling(7, min_periods=1).sum()
    output["state_volatile_count_30d"] = volatile.rolling(30, min_periods=1).sum()

    numeric_columns = [column for column in output.columns if column != "time"]
    output[numeric_columns] = (
        output[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    output = output[["time", *LOAD_STATE_FEATURE_COLUMNS]]
    if output["time"].duplicated().any() or not output["time"].is_monotonic_increasing:
        raise ValueError("Load-state timestamps must be unique and increasing.")
    return output


def main() -> None:
    output_dir = PROJECT_ROOT / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    for route, filename in SOURCE_FILES.items():
        source = PROJECT_ROOT / SOURCE_DIR / filename
        frame = pd.read_csv(source, parse_dates=["time"])
        state = build_load_state_features(frame)
        output = output_dir / f"{route}_load_state_history.csv"
        state.to_csv(output, index=False, encoding="utf-8-sig")
        print(
            f"[{route}] rows={len(state)}, features={len(LOAD_STATE_FEATURE_COLUMNS)}, "
            f"range={state['time'].min().date()}~{state['time'].max().date()}"
        )
        print(f"  -> {output}")


if __name__ == "__main__":
    main()
