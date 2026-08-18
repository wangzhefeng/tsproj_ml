# -*- coding: utf-8 -*-
"""按场景配置清洗点位功率并聚合为完整 5min 总功率序列。"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[5]
LOGIC_VERSION = 1


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_process_config(config_path: Path) -> dict[str, Any]:
    """加载并校验单场景数据处理配置。"""
    config_path = Path(config_path).resolve()
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Process config must be a mapping: {config_path}")

    required = {
        "scene",
        "enabled",
        "dataset_dir",
        "start_time",
        "end_time",
        "freq",
        "time_col",
        "value_col",
        "input_glob",
        "exclude_globs",
        "output_file",
        "audit_file",
        "max_fill_gap_slots",
        "outlier",
    }
    missing = sorted(required - loaded.keys())
    if missing:
        raise ValueError(f"Missing process config fields {missing}: {config_path}")

    start = cast(pd.Timestamp, pd.Timestamp(str(loaded["start_time"])))
    end = cast(pd.Timestamp, pd.Timestamp(str(loaded["end_time"])))
    if bool(pd.isna(start)) or bool(pd.isna(end)) or bool(end < start):
        raise ValueError(f"Invalid time range in {config_path}: {start} -> {end}")
    if int(loaded["max_fill_gap_slots"]) < 0:
        raise ValueError("max_fill_gap_slots must be non-negative")
    if not isinstance(loaded["outlier"], dict):
        raise ValueError("outlier must be a mapping")

    config = dict(loaded)
    config["_config_path"] = config_path
    config["_dataset_path"] = _resolve_path(config["dataset_dir"]).resolve()
    config["_start"] = start
    config["_end"] = end
    return config


def discover_input_files(config: dict[str, Any]) -> list[Path]:
    """按配置发现点位 CSV，并排除外生数据和既有处理产物。"""
    dataset_path = Path(config["_dataset_path"])
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    excluded = list(config.get("exclude_globs") or [])
    files = [
        path
        for path in dataset_path.glob(str(config["input_glob"]))
        if path.is_file()
        and not any(fnmatch.fnmatch(path.name, pattern) for pattern in excluded)
    ]
    files.sort(key=lambda path: path.name)
    if not files:
        raise FileNotFoundError(
            f"No point CSV matched {config['input_glob']!r} under {dataset_path}"
        )
    return files


def _gap_stats(missing: pd.Series) -> tuple[int, int]:
    """返回连续缺失段数量和最长段槽数。"""
    values = missing.fillna(False).to_numpy(dtype=bool)
    changes = np.diff(np.concatenate(([False], values, [False])).astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    lengths = ends - starts
    if len(lengths) == 0:
        return 0, 0
    return int(len(lengths)), int(lengths.max())


def _robust_threshold(values: pd.Series, multiplier: float, minimum: float) -> float:
    numeric = pd.Series(pd.to_numeric(values, errors="coerce"), index=values.index, dtype="float64")
    finite = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return float(minimum)
    median = float(finite.median())
    mad = float((finite - median).abs().median())
    return float(max(minimum, median + multiplier * 1.4826 * mad))


def _detect_isolated_spikes(series: pd.Series, outlier_config: dict[str, Any]) -> tuple[pd.Series, float]:
    """只检测两侧稳定、中心显著偏离的孤立跳点。"""
    half_window = int(outlier_config.get("spike_half_window", 2))
    z_threshold = float(outlier_config.get("spike_z_threshold", 8.0))
    mad_multiplier = float(outlier_config.get("spike_abs_diff_mad_multiplier", 8.0))
    min_abs_diff = float(outlier_config.get("spike_min_abs_diff", 0.0))
    if half_window < 1:
        raise ValueError("spike_half_window must be >= 1")

    window = half_window * 2 + 1
    min_periods = max(3, window // 2)
    baseline = series.rolling(window, center=True, min_periods=min_periods).median()
    residual = (series - baseline).abs()

    local_mad = residual.rolling(window, center=True, min_periods=min_periods).median() * 1.4826
    positive_scale = local_mad[(local_mad > 0) & np.isfinite(local_mad)]
    fallback_scale = float(positive_scale.median()) if not positive_scale.empty else 1e-9
    scale = local_mad.where(local_mad > 0, fallback_scale).fillna(fallback_scale)
    z_score = residual / scale

    abs_diffs = series.diff().abs()
    abs_threshold = _robust_threshold(abs_diffs, mad_multiplier, min_abs_diff)
    companion_spread = (series.shift(1) - series.shift(-1)).abs()
    mask = (
        (residual >= abs_threshold)
        & (z_score >= z_threshold)
        & (companion_spread <= abs_threshold)
    ).fillna(False)
    return mask.astype(bool), abs_threshold


def process_point_file(
    path: Path,
    config: dict[str, Any],
    expected_index: pd.DatetimeIndex,
) -> tuple[pd.Series, dict[str, Any]]:
    """处理单个点位：筛选、规则化、异常修复和缺失填充。"""
    frame = pd.read_csv(path)
    time_col = str(config["time_col"])
    value_col = str(config["value_col"])
    missing_columns = [column for column in (time_col, value_col) if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{path} missing required columns: {missing_columns}")

    source_rows = int(len(frame))
    parsed_time = pd.to_datetime(frame[time_col], errors="coerce")
    parsed_value = pd.to_numeric(frame[value_col], errors="coerce")
    invalid_time_count = int(parsed_time.isna().sum())
    invalid_value_count = int(parsed_value.isna().sum())

    normalized = pd.DataFrame({"time": parsed_time, "value": parsed_value})
    valid_time = normalized["time"].notna()
    outside = valid_time & (
        (normalized["time"] < config["_start"]) | (normalized["time"] > config["_end"])
    )
    outside_count = int(outside.sum())
    normalized = normalized.loc[
        valid_time
        & (normalized["time"] >= config["_start"])
        & (normalized["time"] <= config["_end"])
    ].copy()
    if normalized.empty:
        raise ValueError(f"{path} has no rows inside the configured time range")

    duplicate_count = int(normalized["time"].duplicated(keep="last").sum())
    normalized = (
        normalized.drop_duplicates(subset="time", keep="last")
        .sort_values("time")
        .set_index("time")
    )
    series = normalized["value"].reindex(expected_index).astype(float)
    source_missing_count = int(series.isna().sum())

    physical_invalid = series.notna() & ((series < 0) | ~np.isfinite(series))
    physical_invalid_count = int(physical_invalid.sum())
    cleaned = series.mask(physical_invalid)

    spike_mask, spike_abs_diff_threshold = _detect_isolated_spikes(
        cleaned,
        dict(config.get("outlier") or {}),
    )
    spike_count = int(spike_mask.sum())
    cleaned = cleaned.mask(spike_mask)

    missing_before_fill = int(cleaned.isna().sum())
    gap_segment_count, max_gap_slots = _gap_stats(cleaned.isna())
    allowed_gap = int(config["max_fill_gap_slots"])
    if max_gap_slots > allowed_gap:
        raise ValueError(
            f"{path.name}: continuous gap {max_gap_slots} exceeds max_fill_gap_slots {allowed_gap}"
        )
    if cleaned.notna().sum() == 0:
        raise ValueError(f"{path.name}: no valid values remain after cleaning")

    filled = cleaned.interpolate(method="time", limit_direction="both").ffill().bfill()
    if filled.isna().any() or not np.isfinite(filled.to_numpy(dtype=float)).all():
        raise ValueError(f"{path.name}: missing or non-finite values remain after filling")
    if (filled < 0).any():
        raise ValueError(f"{path.name}: negative values remain after filling")

    audit = {
        "source_file": path.name,
        "source_rows": source_rows,
        "invalid_time_count": invalid_time_count,
        "invalid_value_count": invalid_value_count,
        "outside_time_range_count": outside_count,
        "duplicate_timestamp_count": duplicate_count,
        "source_missing_count": source_missing_count,
        "physical_invalid_count": physical_invalid_count,
        "spike_count": spike_count,
        "spike_abs_diff_threshold": spike_abs_diff_threshold,
        "missing_before_fill": missing_before_fill,
        "filled_value_count": missing_before_fill,
        "gap_segment_count": gap_segment_count,
        "max_gap_slots": max_gap_slots,
        "value_min": float(filled.min()),
        "value_max": float(filled.max()),
    }
    filled.name = path.stem
    return filled, audit


def _atomic_write_csv(frame: pd.DataFrame, output_path: Path) -> None:
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    frame.to_csv(temp_path, index=False)
    os.replace(temp_path, output_path)


def _atomic_write_json(payload: dict[str, Any], output_path: Path) -> None:
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, output_path)


def process_scene(config_path: Path) -> dict[str, Any]:
    """处理一个场景并写出 df_power.csv 与审计 JSON。"""
    config = load_process_config(config_path)
    if not bool(config["enabled"]):
        return {
            "scene": str(config["scene"]),
            "status": "skipped",
            "reason": str(config.get("disabled_reason") or "disabled by config"),
            "config": str(config["_config_path"]),
        }

    expected_index = pd.date_range(
        start=config["_start"],
        end=config["_end"],
        freq=str(config["freq"]),
    )
    if len(expected_index) == 0:
        raise ValueError(f"Empty expected time index: {config_path}")

    point_series: list[pd.Series] = []
    point_audits: list[dict[str, Any]] = []
    input_files = discover_input_files(config)
    for input_path in input_files:
        series, audit = process_point_file(input_path, config, expected_index)
        point_series.append(series)
        point_audits.append(audit)

    point_frame = pd.concat(point_series, axis=1)
    aggregate = point_frame.sum(axis=1, min_count=len(point_series))
    if not aggregate.index.equals(expected_index):
        raise ValueError("Aggregate time index does not match the configured complete index")
    aggregate_values = aggregate.to_numpy(dtype=float)
    if not np.isfinite(aggregate_values).all() or (aggregate_values < 0).any():
        raise ValueError("Aggregate contains missing, non-finite, or negative values")

    output = pd.DataFrame(
        {
            "time": expected_index.strftime("%Y-%m-%d %H:%M:%S"),
            "value": aggregate_values,
        }
    )
    dataset_path = Path(config["_dataset_path"])
    output_path = dataset_path / str(config["output_file"])
    audit_path = dataset_path / str(config["audit_file"])
    audit_payload = {
        "logic_version": LOGIC_VERSION,
        "scene": str(config["scene"]),
        "status": "success",
        "config_path": str(config["_config_path"]),
        "dataset_dir": str(dataset_path),
        "start_time": expected_index[0].strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": expected_index[-1].strftime("%Y-%m-%d %H:%M:%S"),
        "freq": str(config["freq"]),
        "expected_rows": int(len(expected_index)),
        "input_files": [path.name for path in input_files],
        "points": point_audits,
        "output": {
            "data_path": str(output_path),
            "rows": int(len(output)),
            "columns": list(output.columns),
            "start_time": output.iloc[0]["time"],
            "end_time": output.iloc[-1]["time"],
            "value_min": float(output["value"].min()),
            "value_max": float(output["value"].max()),
        },
    }

    _atomic_write_csv(output, output_path)
    _atomic_write_json(audit_payload, audit_path)
    return {
        "scene": str(config["scene"]),
        "status": "success",
        "rows": int(len(output)),
        "points": int(len(point_series)),
        "data_path": str(output_path),
        "audit_path": str(audit_path),
    }


def run_config_root(config_root: Path) -> list[dict[str, Any]]:
    """递归执行日期目录下的全部 data_process.yaml。"""
    config_root = Path(config_root).resolve()
    config_paths = sorted(config_root.glob("**/data_process.yaml"))
    if not config_paths:
        raise FileNotFoundError(f"No data_process.yaml found under {config_root}")

    results: list[dict[str, Any]] = []
    for config_path in config_paths:
        try:
            result = process_scene(config_path)
        except Exception as exc:
            result = {
                "scene": str(config_path.parent.relative_to(config_root)),
                "status": "failed",
                "reason": str(exc),
                "config": str(config_path),
            }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="包含各场景 data_process.yaml 的日期配置目录",
    )
    args = parser.parse_args()
    results = run_config_root(args.config_root)
    success_count = sum(item["status"] == "success" for item in results)
    skipped_count = sum(item["status"] == "skipped" for item in results)
    failed = [item for item in results if item["status"] == "failed"]
    print(
        f"SUMMARY success={success_count} skipped={skipped_count} failed={len(failed)}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
