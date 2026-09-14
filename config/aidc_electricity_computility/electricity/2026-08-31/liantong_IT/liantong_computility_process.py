"""联通算力离线准备：实例保留、截面聚合、零语义及逐样本异常审计。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

METRICS = (
    "cpu_util", "gpu_util", "memory_amount", "memory_total", "memory_util",
    "gpu_memory_amount", "gpu_memory_total", "gpu_memory_util", "gpu_power_usage",
)
SAMPLE_COLUMNS = [
    "time", "job_id", "metric", "raw_value", "processed_value", "source_file",
    "source_row", "sample_position", "is_corrected", "epoch_seconds",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_metric(path: Path, source: str, metric: str, grid: pd.DatetimeIndex, sample_writer=None):
    """保留全部实例；利用率为样本统计，不假设历史副本数或跨指标实例对齐。"""
    if source not in {"training", "inference"} or metric not in METRICS:
        raise ValueError("未知来源或指标")
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"uid", "metric", "value"}.issubset(rows.columns):
        raise ValueError(f"缺少 uid/metric/value: {path}")
    if rows.uid.eq("").any() or rows.uid.duplicated().any():
        raise ValueError(f"Job ID 为空或多行重复: {path}")
    expected_metric = f"lepton__aec2__acn__job__{metric}"
    if not rows.metric.eq(expected_metric).all():
        raise ValueError(f"metric 与文件不匹配: {path}")
    parts, repairs = [], []
    for row_number, row in enumerate(rows.itertuples(index=False)):
        raw = str(row.value)
        if raw in {"", "[]"}:
            continue
        try:
            points = json.loads("[" + raw.replace("|", ",") + "]")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"非法序列 {path}:{row_number + 2}") from exc
        if any(not isinstance(p, list) or len(p) != 2 or type(p[0]) is not int
               or isinstance(p[1], (bool, list, dict)) or p[1] is None for p in points):
            raise ValueError(f"非法采样结构 {path}:{row_number + 2}")
        frame = pd.DataFrame(points, columns=["epoch", "value"])
        frame["value"] = pd.to_numeric(frame["value"], errors="raise")
        if not np.isfinite(frame.value).all() or frame.value.lt(0).any():
            raise ValueError(f"负值或非有限值 {path}:{row_number + 2}")
        times = pd.to_datetime(frame.epoch, unit="s", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        if not times.isin(grid).all():
            raise ValueError(f"时间不在指定 5min 网格: {path}:{row_number + 2}")
        frame["time"] = times
        frame["job_id"] = row.uid
        if metric.endswith("_util"):
            for position in frame.index[frame.value.gt(1)]:
                repairs.append({"job_id": row.uid, "row": int(row_number + 2),
                                "position": int(position), "time": str(times.iloc[position]),
                                "original": float(frame.value.iloc[position]), "replacement": 1.0})
            frame["value"] = frame.value.clip(upper=1)
        if sample_writer is not None:
            # 长表不填零、不去重；保留数值原文，处理值仍使用源单位。
            records = zip(times.dt.strftime("%Y-%m-%d %H:%M:%S"), points, frame.value)
            for position, (time, (epoch, original), processed) in enumerate(records):
                sample_writer.writerow([
                    time, row.uid, metric, original, processed, path.name,
                    row_number + 2, position, int(metric.endswith("_util") and float(original) > 1), epoch,
                ])
        parts.append(frame[["time", "job_id", "value"]])
    samples = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["time", "job_id", "value"])
    prefix = f"{source}_{metric}"
    if samples.empty:
        stats = pd.DataFrame(0.0, index=grid, columns=["sum", "mean", "std", "count"])
        jobs = pd.Series(0, index=grid)
    else:
        grouped = samples.groupby("time", sort=True)
        stats = grouped.value.agg(["sum", "mean", "std", "count"]).reindex(grid).fillna(0.0)
        jobs = grouped.job_id.nunique().reindex(grid, fill_value=0)
    features = pd.DataFrame(index=grid)
    if metric.endswith("_util"):
        features[f"{prefix}_sample_mean"] = stats["mean"]
        features[f"{prefix}_sample_std"] = stats["std"]
    else:
        scale, unit = (1000.0, "kw") if metric == "gpu_power_usage" else (1.0, "raw")
        features[f"{prefix}_sum_{unit}"] = stats["sum"] / scale
        features[f"{prefix}_sample_mean_{unit}"] = stats["mean"] / scale
    features[f"{prefix}_job_count"] = jobs.astype("int64")
    quality = pd.DataFrame({f"{prefix}_sample_count": stats["count"].astype("int64"),
                            f"{prefix}_no_record": stats["count"].eq(0).astype("int64")}, index=grid)
    audit = {"file": str(path), "sha256": sha256(path), "source": source, "metric": metric,
             "csv_rows": len(rows), "samples": len(samples), "no_record_steps": int(stats["count"].eq(0).sum()),
             "max_samples_per_job_time": int(samples.groupby(["time", "job_id"]).size().max()) if len(samples) else 0,
             "repairs": repairs}
    return features, quality, audit


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT = ROOT / "dataset/aidc_electricity_computility/electricity/2026-08-31/liantong_IT"


def _run(source_dir: Path, target_csv: Path, output_dir: Path,
         start: str, end: str, overwrite: bool, sample_dir: Path) -> dict:
    """校验全部输入后发布；输出不存在时首次创建，重建需显式授权。"""
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if first.tz is not None or last.tz is not None or first > last:
        raise ValueError("start/end 必须为有序本地墙上时间")
    if first.floor("5min") != first or last.floor("5min") != last:
        raise ValueError("start/end 必须对齐 5min")
    grid = pd.date_range(first, last, freq="5min", name="time")
    suffix = f"5min_{first:%Y%m%d}_{last:%Y%m%d}.csv"
    names = [f"computility_{source}_{suffix}" for source in ("training", "inference", "history", "quality")]
    names += [f"power_computility_{suffix}", "training_job.csv", "inference_job.csv", "computility_processing_audit.json"]
    compute_dir = output_dir / "aidc_comp_liantong_5min"
    destinations = [(output_dir if name.startswith("power_computility_") else compute_dir) / name for name in names]
    if not overwrite and any(path.exists() for path in destinations):
        raise FileExistsError("输出已存在；明确重建请使用 --overwrite")
    inputs = []
    for source in ("training", "inference"):
        files = sorted((source_dir / source).glob("*.csv"))
        selected = []
        for metric in METRICS:
            matches = [p for p in files if p.name.startswith(f"lepton__aec2__acn__job__{metric}_merged_")]
            if len(matches) != 1:
                raise ValueError(f"{source}/{metric} 必须恰好有一个源文件")
            selected.append(matches[0])
            inputs.append((source, metric, matches[0]))
        if set(files) != set(selected):
            raise ValueError(f"{source} 有未识别的 CSV")
    input_paths = [path for _, _, path in inputs] + [target_csv]
    if {p.resolve() for p in destinations} & {p.resolve() for p in input_paths}:
        raise ValueError("输出不得覆盖输入文件")
    hashes = {str(p): sha256(p) for p in input_paths}
    target = pd.read_csv(target_csv, float_precision="round_trip")
    if list(target.columns) != ["time", "value"]:
        raise ValueError("目标必须为 time,value 派生表")
    target["time"] = pd.to_datetime(target.time, errors="raise")
    if not pd.DatetimeIndex(target.time).equals(grid):
        raise ValueError("目标时间轴必须严格等于完整网格，不能重复、缺失或重排")
    target["value"] = pd.to_numeric(target.value, errors="raise")
    if not np.isfinite(target.value).all():
        raise ValueError("目标含缺失或非有限值")
    frames = {"training": [], "inference": []}
    quality_frames, audits = [], []
    for source in frames:
        with (sample_dir / f"{source}_job.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle, lineterminator="\n").writerow(SAMPLE_COLUMNS)
    for source, metric, path in inputs:
        with (sample_dir / f"{source}_job.csv").open("a", newline="", encoding="utf-8") as handle:
            frame, quality, audit = aggregate_metric(
                path, source, metric, grid, sample_writer=csv.writer(handle, lineterminator="\n")
            )
        frames[source].append(frame)
        quality_frames.append(quality)
        audits.append(audit)
        print(f"处理 {source}/{metric}: {audit['samples']} 样本，修正 {len(audit['repairs'])} 条", flush=True)
    training = pd.concat(frames["training"], axis=1)
    inference = pd.concat(frames["inference"], axis=1)
    history = pd.concat([training, inference], axis=1)
    quality = pd.concat(quality_frames, axis=1)
    merged = target.merge(history.reset_index(), on="time", how="left", validate="one_to_one")
    if not merged.value.equals(target.value) or len(merged) != len(target):
        raise ValueError("拼接改变目标")
    products = [training.reset_index(), inference.reset_index(), history.reset_index(), quality.reset_index(), merged]
    for frame in products:
        if not np.isfinite(frame.drop(columns="time").to_numpy(dtype=float)).all():
            raise ValueError("派生特征存在非有限值")
    if hashes != {str(p): sha256(p) for p in input_paths}:
        raise ValueError("处理期间输入发生变化；拒绝发布")
    report = {
        "semantics_version": "liantong_computility_v1",
        "rules": {"source_mapping": {"training": "original job", "inference": "original app"},
                  "same_job_time_values": "distinct instances; retain all; no replica multiplier",
                  "missing": "all absent metrics zero by user confirmation; scoped to this dataset",
                  "util_outliers": "upper clip 1; negative/nonfinite reject",
                  "power": "W to kW", "memory": "original unknown unit",
                  "util_aggregation": "reported-instance sample mean/std; empty and singleton std zero",
                  "temporal_features": "none; compute in forecast compiler as-of",
                  "availability": "observed_past; delivery latency not established",
                  "cross_source_totals": "not generated; physical non-overlap unconfirmed"},
        "inputs": audits, "target": {"file": str(target_csv), "sha256": hashes[str(target_csv)],
                                      "preserved": True, "note": "existing target imputation retained"},
        "builder_sha256": sha256(Path(__file__)),
        "sample_export": {"version": "raw_samples_v1", "units": "original; GPU power W",
                          "order": "metric declaration, source row, sample position",
                          "missing_samples": "not synthesized", "instance_id": "unavailable"},
        "start": str(first), "end": str(last), "rows": len(grid),
        "model_configs_modified": False, "deployment_verified": False,
        "outputs": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    compute_dir.mkdir(parents=True, exist_ok=True)
    for destination, frame in zip(destinations[:5], products):
        frame.to_csv(destination, index=False, date_format="%Y-%m-%d %H:%M:%S")
        report["outputs"].append({"file": str(destination.relative_to(output_dir)), "rows": len(frame),
                                  "columns": list(frame.columns), "sha256": sha256(destination)})
    for source in ("training", "inference"):
        destination = compute_dir / f"{source}_job.csv"
        (sample_dir / destination.name).replace(destination)
        report["outputs"].append({"file": str(destination.relative_to(output_dir)),
                                  "rows": sum(item["samples"] for item in audits if item["source"] == source),
                                  "columns": SAMPLE_COLUMNS, "sha256": sha256(destination)})
    destinations[-1].write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def run(source_dir: Path, target_csv: Path, output_dir: Path,
        start: str = "2026-08-01", end: str = "2026-08-31 23:55:00",
        overwrite: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    # 同文件系统暂存，验证失败自动清理，正式源文件和已有结果保持原样。
    with TemporaryDirectory(prefix=".computility_samples_", dir=output_dir) as temp:
        return _run(source_dir, target_csv, output_dir, start, end, overwrite, Path(temp))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_OUTPUT / "aidc_comp_liantong_5min")
    parser.add_argument("--target-csv", type=Path, default=DEFAULT_OUTPUT / "target_power_5min_20260801_20260831.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default="2026-08-01")
    parser.add_argument("--end", default="2026-08-31 23:55:00")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = run(args.source_dir, args.target_csv, args.output_dir, args.start, args.end, args.overwrite)
    print(json.dumps({"rows": report["rows"], "files": len(report["outputs"]),
                      "repairs": sum(len(item["repairs"]) for item in report["inputs"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
