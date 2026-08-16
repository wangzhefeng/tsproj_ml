from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd


AGGREGATION_METHODS = {"mean", "max", "min", "sum", "median"}

# 聚合/填充逻辑版本：修改本文件处理逻辑时 +1，使既有派生缓存自动失效
LOGIC_VERSION = 2

# pandas 频率别名 -> 项目规范写法（与建模侧 default_lags_for_freq 的 1D 约定对齐）
_FREQ_ALIASES = {"D": "1D", "h": "1h", "H": "1h", "min": "1min", "T": "1min"}


def _normalize_freq(freq: str) -> str:
    """把 pandas 频率别名规范化为项目约定写法（如 D -> 1D, h -> 1h）。"""
    return _FREQ_ALIASES.get(freq, freq)


def _freq_to_timedelta(freq: str) -> pd.Timedelta:
    # 月频是非固定频率（28~31 天），to_offset(freq).nanos 会 RAISE。
    # 用 31 天作为上界做 target>=source 校验（月频 target 一定 >= 日频/小时频 source）。
    if freq in ("1ME", "1MS", "ME", "MS"):
        return pd.Timedelta(days=31)
    # cast：pandas stub 把 Timedelta 构造器标成 Timedelta | NaTType，固定频率下不会 NaT
    return cast(pd.Timedelta, pd.Timedelta(nanoseconds=int(pd.tseries.frequencies.to_offset(freq).nanos)))


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


def _linear_fill(series: pd.Series, weeks: int) -> pd.Series:
    """按时间权重线性插值，双向补端点。weeks 参数不用，保持签名一致。

    回测依据（2026-07-29，AIDC A/B 路）：linear 在 1 槽到 288 槽（1 天）
    所有缺口长度上 MAPE 均优于 seasonal_slot（0.4%-1.5% vs ~3.0%）——
    AIDC 负荷日内平坦 + 缓慢趋势，局部水平优于周周期匹配。
    """
    return series.interpolate(method="time", limit_direction="both")


# fill_method -> 填充函数注册表（"none" 不注册，表示不填充）
_FILLERS = {
    "linear": _linear_fill,
    "seasonal_slot": _seasonal_slot_fill,
}

FILL_METHODS = {"none", *_FILLERS}


def _gap_stats(is_missing: np.ndarray) -> dict[str, int]:
    """统计连续缺失段：段数与最长段长度（单位：源频率槽数）。"""
    gaps = np.diff(np.concatenate(([0], is_missing.view(np.int8), [0])))
    starts = np.flatnonzero(gaps == 1)
    ends = np.flatnonzero(gaps == -1)
    lengths = ends - starts
    return {
        "gap_segment_count": int(len(starts)),
        "max_gap_length": int(lengths.max()) if len(lengths) else 0,
    }


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
    unit_scale: float,
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
        "unit_scale": float(unit_scale),
        "logic_version": LOGIC_VERSION,
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
    unit_scale: float = 1.0,
    output_path: str | Path,
) -> AggregationResult:
    """把规则化后的单目标时间序列聚合到目标频率，并落盘审计信息。

    unit_scale：聚合结果的全局缩放系数（默认 1.0）。典型用途——
    源数据是 5min 平均功率（kW），sum 聚合后乘步长 1/12 h 得电量 kWh。
    """
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Aggregation source file not found: {source}")
    if method not in AGGREGATION_METHODS:
        raise ValueError(f"method must be one of {sorted(AGGREGATION_METHODS)}")
    if fill_method not in FILL_METHODS:
        raise ValueError(f"fill_method must be one of {sorted(FILL_METHODS)}")
    if fill_weeks <= 0:
        raise ValueError("fill_weeks must be > 0")
    source_freq = _normalize_freq(source_freq)
    target_freq = _normalize_freq(target_freq)
    if _freq_to_timedelta(target_freq) < _freq_to_timedelta(source_freq):
        raise ValueError(
            f"target_freq ({target_freq}) must not be finer than source_freq ({source_freq})"
        )

    destination = Path(output_path)
    if destination.resolve() == source.resolve():
        raise ValueError("output_path must not overwrite source_path")
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
        unit_scale=unit_scale,
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
    gap_stats = _gap_stats(series.isna().to_numpy())
    before_fill = inserted_count
    filler = _FILLERS.get(fill_method)
    if filler is not None:
        series = filler(series, fill_weeks)

    remaining_index = series.index[series.isna()]
    if len(remaining_index):
        preview = ", ".join(str(ts) for ts in remaining_index[:10])
        raise ValueError(
            f"Aggregation has {len(remaining_index)} missing source-frequency values "
            f"after fill_method={fill_method!r}: {preview}"
            f"{' ...' if len(remaining_index) > 10 else ''}"
        )
    filled_count = before_fill - len(remaining_index)
    aggregated = getattr(series.resample(target_freq), method)().reset_index(name=target_col)
    if unit_scale != 1.0:
        aggregated[target_col] = aggregated[target_col] * unit_scale
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
        **gap_stats,
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


# ---------------------------------------------------------------------------
# 配置驱动的聚合入口
# ---------------------------------------------------------------------------
# 聚合配置 YAML schema 与模型配置完全独立（无 base_config/overrides 字段），
# 避免被批量模型 driver（glob **/*.yaml）误当模型配置加载。
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AggregationSpec:
    """单个聚合任务规格。"""
    source_path: Path
    time_col: str
    target_col: str
    source_freq: str
    target_freq: str
    method: str
    fill_method: str
    fill_weeks: int
    unit_scale: float
    output_path: Path


def _load_aggregation_config(config_path: str | Path) -> dict[str, Any]:
    """加载聚合 YAML 配置。schema 独立于模型配置，无 base_config/overrides。"""
    import yaml

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Aggregation config must be a mapping: {config_path}")
    return raw  # type: ignore[return-value]


def _build_spec(raw: dict[str, Any], config_path: Path) -> AggregationSpec:
    """解析单个聚合任务的 YAML 字段为 AggregationSpec。"""
    missing = [k for k in ("source_path", "time_col", "target_col",
                           "source_freq", "target_freq", "output_path") if k not in raw]
    if missing:
        raise ValueError(f"Aggregation config missing required fields {missing}: {config_path}")
    return AggregationSpec(
        source_path=_resolve_path(raw["source_path"], config_path),
        time_col=raw["time_col"],
        target_col=raw["target_col"],
        source_freq=raw["source_freq"],
        target_freq=raw["target_freq"],
        method=raw.get("method", "mean"),
        fill_method=raw.get("fill_method", "none"),
        fill_weeks=raw.get("fill_weeks", 4),
        unit_scale=float(raw.get("unit_scale", 1.0)),
        output_path=_resolve_path(raw["output_path"], config_path),
    )


def _resolve_path(p: str, config_path: Path) -> Path:
    """路径解析：~ 开头按 home；相对路径按项目根（与模型配置的 data_dir 约定一致）。"""
    pr = Path(p).expanduser()
    return pr if pr.is_absolute() else (PROJECT_ROOT / pr).resolve()


def _run_spec(spec: AggregationSpec, force: bool = False) -> None:
    """执行单个聚合任务并打印结果。"""
    if force:
        spec.output_path.unlink(missing_ok=True)
        spec.output_path.with_name(f"{spec.output_path.name}.aggregate.json").unlink(missing_ok=True)
    result = aggregate_csv(
        source_path=spec.source_path,
        time_col=spec.time_col,
        target_col=spec.target_col,
        source_freq=spec.source_freq,
        target_freq=spec.target_freq,
        method=spec.method,
        fill_method=spec.fill_method,
        fill_weeks=spec.fill_weeks,
        unit_scale=spec.unit_scale,
        output_path=spec.output_path,
    )
    status = "重新生成" if result.regenerated else "复用缓存"
    print(
        f"[{status}] {spec.source_freq} -> {spec.target_freq}: "
        f"{result.data_path.name} "
        f"({result.output_rows} 行, 补齐 {result.filled_value_count} 个缺失点)"
    )


def run_aggregation(
    config_path: str | Path,
    *,
    force: bool = False,
) -> None:
    """加载 YAML 聚合配置并执行。

    配置文件可写单个任务（顶层平铺）或多任务（顶层 tasks: 列表）。
    """
    config_path = Path(config_path).resolve()
    raw = _load_aggregation_config(config_path)
    task_list = raw["tasks"] if "tasks" in raw else [raw]
    for item in task_list:
        spec = _build_spec(item, config_path)
        _run_spec(spec, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description="单目标时序数据频率聚合（配置驱动）")
    parser.add_argument("config", help="聚合配置 YAML 路径")
    parser.add_argument("--force", action="store_true", help="忽略缓存强制重建")
    args = parser.parse_args()
    run_aggregation(args.config, force=args.force)


if __name__ == "__main__":
    main()
