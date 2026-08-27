# -*- coding: utf-8 -*-
"""合并当前场景 A/B 路 15min 总负荷，供多变量模型测试使用。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIO = "aidc_load_15min_short"
DATA_DIR = PROJECT_ROOT / "dataset" / SCENARIO
SOURCE_FILES = {
    "A_load": "A_Loads_15min_mean_20251001_20260731.csv",
    "B_load": "B_Loads_15min_mean_20251001_20260731.csv",
}
OUTPUT_FILE = DATA_DIR / "forecasting_data" / "AB_Loads_15min_mean_20251001_20260731.csv"


def _load_route(column: str, filename: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA_DIR / filename)
    missing_columns = {"time", "value"} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{filename} missing columns: {sorted(missing_columns)}")
    frame = frame.loc[:, ["time", "value"]].copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    frame[column] = pd.to_numeric(frame.pop("value"), errors="raise")
    frame = frame.sort_values("time").reset_index(drop=True)

    if frame["time"].duplicated().any():
        raise ValueError(f"{column} timestamps must be unique.")
    if len(frame) > 1 and not (
        frame["time"].diff().dropna() == pd.Timedelta(minutes=15)
    ).all():
        raise ValueError(f"{column} must have a continuous 15min timeline.")
    if frame[column].isna().any() or not np.isfinite(frame[column].to_numpy()).all():
        raise ValueError(f"{column} contains missing or non-finite values.")
    return frame


def build_multivariate_loads() -> pd.DataFrame:
    routes = {
        column: _load_route(column, filename)
        for column, filename in SOURCE_FILES.items()
    }
    if not routes["A_load"]["time"].equals(routes["B_load"]["time"]):
        raise ValueError("Route A/B timestamps are not aligned.")

    return routes["A_load"].merge(
        routes["B_load"],
        on="time",
        how="inner",
        validate="one_to_one",
    )


def main() -> None:
    output = build_multivariate_loads()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(
        f"rows={len(output)}, columns={list(output.columns)}, "
        f"range={output['time'].min()} -> {output['time'].max()}"
    )
    print(f"-> {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
