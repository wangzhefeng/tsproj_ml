from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


AGGREGATION_METHODS = {"mean", "max", "min", "sum", "median"}
FILL_METHODS = {"none", "linear", "seasonal_slot"}


@dataclass(frozen=True)
class AggregationResult:
    data_path: Path
    audit_path: Path
    regenerated: bool
    source_rows: int
    output_rows: int
    inserted_timestamp_count: int
    filled_value_count: int


def _seasonal_slot_fill(series: pd.Series, weeks: int) -> pd.Series:
    """用局部周窗口中相同星期和时刻的观测均值填充缺失点。"""
    missing = series.isna().to_numpy()
    if not missing.any():
        return series

    index = series.index
    day_of_week = index.dayofweek.to_numpy()
    minute_of_day = (index.hour * 60 + index.minute).to_numpy()
    values = series.to_numpy(dtype=float)
    filled = series.copy()

    for position in np.flatnonzero(missing):
        timestamp = index[position]
        start = index.searchsorted(timestamp - pd.Timedelta(weeks=weeks), side="left")
        end = index.searchsorted(timestamp + pd.Timedelta(weeks=weeks), side="right")
        window = values[start:end]
        candidates = (
            (day_of_week[start:end] == day_of_week[position])
            & (minute_of_day[start:end] == minute_of_day[position])
            & ~np.isnan(window)
        )
        if candidates.any():
            filled.iloc[position] = float(window[candidates].mean())
    return filled


def _default_output_path(source_path: Path, target_freq: str, method: str) -> Path:
    safe_freq = "".join(char if char.isalnum() else "-" for char in target_freq).strip("-")
    return source_path.parent / "derived" / f"{source_path.stem}__{safe_freq}_{method}.csv"


def _audit_config(
    *,
    source_path: Path,
    time_col: str,
    target_col: str,
    source_freq: str,
    target_freq: str,
    method: str,
    fill_method: str,
    fill_weeks: int,
) -> dict[str, Any]:
    stat = source_path.stat()
    return {
        "source_path": str(source_path.resolve()),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "time_col": time_col,
        "target_col": target_col,
        "source_freq": source_freq,
        "target_freq": target_freq,
        "method": method,
        "fill_method": fill_method,
        "fill_weeks": int(fill_weeks),
    }


def _can_reuse(output_path: Path, audit_path: Path, expected_config: dict[str, Any]) -> bool:
    if not output_path.exists() or not audit_path.exists():
        return False
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return audit.get("config") == expected_config


def aggregate_csv(
    *,
    source_path: str | Path,
    time_col: str,
    target_col: str,
    source_freq: str,
    target_freq: str,
    method: str = "mean",
    fill_method: str = "none",
    fill_weeks: int = 4,
    output_path: str | Path | None = None,
) -> AggregationResult:
    """把规则化后的单目标时间序列聚合到目标频率，并落盘审计信息。"""
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Aggregation source file not found: {source}")
    if method not in AGGREGATION_METHODS:
        raise ValueError(f"aggregation_method must be one of {sorted(AGGREGATION_METHODS)}")
    if fill_method not in FILL_METHODS:
        raise ValueError(f"aggregation_fill_method must be one of {sorted(FILL_METHODS)}")
    if fill_weeks <= 0:
        raise ValueError("aggregation_fill_weeks must be > 0")

    destination = Path(output_path) if output_path else _default_output_path(source, target_freq, method)
    if destination.resolve() == source.resolve():
        raise ValueError("aggregation_output_path must not overwrite data_path")
    audit_path = destination.with_name(f"{destination.name}.aggregate.json")
    config = _audit_config(
        source_path=source,
        time_col=time_col,
        target_col=target_col,
        source_freq=source_freq,
        target_freq=target_freq,
        method=method,
        fill_method=fill_method,
        fill_weeks=fill_weeks,
    )
    if _can_reuse(destination, audit_path, config):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        return AggregationResult(
            data_path=destination,
            audit_path=audit_path,
            regenerated=False,
            source_rows=int(audit["source_rows"]),
            output_rows=int(audit["output_rows"]),
            inserted_timestamp_count=int(audit["inserted_timestamp_count"]),
            filled_value_count=int(audit["filled_value_count"]),
        )

    frame = pd.read_csv(source)
    missing_columns = [column for column in (time_col, target_col) if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Aggregation columns not found: {missing_columns}")
    source_rows = len(frame)
    frame = frame[[time_col, target_col]].copy()
    frame[time_col] = pd.to_datetime(frame[time_col], errors="raise")
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    if frame[target_col].isna().any():
        raise ValueError(f"Aggregation target '{target_col}' contains non-numeric or missing values")

    series = frame.sort_values(time_col).set_index(time_col)[target_col].resample(source_freq).mean()
    inserted_count = int(series.isna().sum())
    before_fill = inserted_count
    if fill_method == "linear":
        series = series.interpolate(method="time", limit_direction="both")
    elif fill_method == "seasonal_slot":
        series = _seasonal_slot_fill(series, fill_weeks)

    remaining = int(series.isna().sum())
    if remaining:
        raise ValueError(
            f"Aggregation has {remaining} missing source-frequency values after fill_method={fill_method!r}"
        )
    filled_count = before_fill - remaining
    aggregated = getattr(series.resample(target_freq), method)().reset_index(name=target_col)
    if aggregated[target_col].isna().any():
        raise ValueError("Aggregation produced missing output values")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_csv = destination.with_name(f".{destination.name}.tmp")
    temp_audit = audit_path.with_name(f".{audit_path.name}.tmp")
    aggregated.to_csv(temp_csv, index=False)
    audit = {
        "config": config,
        "source_rows": source_rows,
        "output_rows": len(aggregated),
        "inserted_timestamp_count": inserted_count,
        "filled_value_count": filled_count,
        "duplicate_timestamp_count": int(frame[time_col].duplicated().sum()),
        "time_range_start": str(aggregated[time_col].iloc[0]),
        "time_range_end": str(aggregated[time_col].iloc[-1]),
    }
    temp_audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_csv, destination)
    os.replace(temp_audit, audit_path)
    return AggregationResult(
        data_path=destination,
        audit_path=audit_path,
        regenerated=True,
        source_rows=source_rows,
        output_rows=len(aggregated),
        inserted_timestamp_count=inserted_count,
        filled_value_count=filled_count,
    )


def resolve_config_aggregation(cfg) -> AggregationResult | None:
    """按 AppConfig 生成派生文件，并把本次有效 data_path 切换到派生文件。"""
    if not cfg.aggregation_enabled:
        return None
    result = aggregate_csv(
        source_path=cfg.data_path,
        time_col=cfg.time_col,
        target_col=cfg.target_col,
        source_freq=cfg.aggregation_source_freq,
        target_freq=cfg.freq,
        method=cfg.aggregation_method,
        fill_method=cfg.aggregation_fill_method,
        fill_weeks=cfg.aggregation_fill_weeks,
        output_path=cfg.aggregation_output_path,
    )
    cfg.data_path = str(result.data_path)
    return result




# ---------------------------------------------------------------------------
# 独立数据聚合入口：AIDC A/B 路 5min 负荷 -> 小时 / 天 频率派生数据
# 聚合流程全部在本脚本内完成，不经过 run.py 与 AppConfig。
# ---------------------------------------------------------------------------
# 项目根目录（本脚本位于 data_provider/ 下，其父目录即项目根）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "aidc_power"

# 源数据日期区间标识（随原始数据更新而调整）
DATE_RANGE = "20251001_20260728"

# 目标频率 -> 输出文件名中的频率标签
FREQ_LABELS = {"15min": "15min", "h": "1hour", "D": "1day"}


def main() -> None:
    # 聚合任务表：A/B 两路 x 15min/小时/天三种目标频率，统一 mean 聚合 + seasonal_slot 缺失填充
    for route in ("A", "B"):
        source_path = DATASET_DIR / f"{route}_Loads_5min_{DATE_RANGE}.csv"
        for target_freq, freq_label in FREQ_LABELS.items():
            output_path = (
                DATASET_DIR / "derived"
                / f"{route}_Loads_{freq_label}_mean_{DATE_RANGE}.csv"
            )
            result = aggregate_csv(
                source_path=source_path,
                time_col="time",
                target_col="value",
                source_freq="5min",
                target_freq=target_freq,
                method="mean",
                fill_method="seasonal_slot",
                fill_weeks=4,
                output_path=output_path,
            )
            status = "重新生成" if result.regenerated else "复用缓存"
            print(
                f"[{status}] {route} 路 5min -> {target_freq}: "
                f"{result.data_path.name} "
                f"({result.output_rows} 行, 补齐 {result.filled_value_count} 个缺失点)"
            )


if __name__ == "__main__":
    main()
