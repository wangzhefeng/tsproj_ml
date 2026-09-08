"""红太阳2025年数据检查与日均聚合；不填补、不修改原始CSV。"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset/hongtaiyang_cesuan"
SOURCES = (("xinnengyuan", "demand_load"), ("guangdianchang", "demand_load"),
           ("guangdianchang", "pv_load"))


def validate_frame(frame: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    if list(frame.columns) != ["time", "value"]:
        raise ValueError("source columns must be exactly time,value")
    result = frame.copy()
    result["time"] = pd.to_datetime(result.time, errors="raise")
    result["value"] = pd.to_numeric(result.value, errors="raise")
    expected = pd.date_range("2025-01-01", "2026-01-01", freq=freq, inclusive="left")
    if not pd.DatetimeIndex(result.time).equals(expected):
        raise ValueError("source must contain the complete ordered 2025 time grid")
    if not np.isfinite(result.value).all() or (result.value < 0).any():
        raise ValueError("source values must be finite and nonnegative")
    return result


def aggregate_daily(frame: pd.DataFrame) -> pd.DataFrame:
    source = validate_frame(frame)
    daily = source.set_index("time").resample("1D").mean().reset_index()
    return validate_frame(daily, "1D")


def prepare(data_root: Path = DATA) -> list[dict]:
    reports = []
    # 全部校验通过后才写派生文件，避免部分坏输入产生一半数据。
    inputs = [(station, target, data_root / station / f"{target}.csv")
              for station, target in SOURCES]
    frames = [validate_frame(pd.read_csv(path)) for _, _, path in inputs]
    for (station, target, path), frame in zip(inputs, frames):
        report = {"source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "source_rows": len(frame), "source_freq": "15min"}
        if target == "demand_load":
            output = data_root / station / "freq_1day/demand_load.csv"
            daily = aggregate_daily(frame)
            output.parent.mkdir(parents=True, exist_ok=True)
            daily.to_csv(output, index=False)
            report.update(output=str(output), output_rows=len(daily), method="mean", target_freq="1D")
            output.with_suffix(".aggregate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        reports.append(report)
    return reports


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2, ensure_ascii=False))
