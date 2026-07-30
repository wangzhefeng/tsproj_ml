# -*- coding: utf-8 -*-

import argparse
import csv
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


TIME_COL = "count_data_time"
POWER_TARGET_COL = "h_total_use"
COMPUTILITY_OUTPUT_NAME = "df_computility.csv"
MERGED_OUTPUT_NAME = "df.csv"
SELECTED_OUTPUT_NAME = "df_selected.csv"
FEATURE_REPORT_OUTPUT_NAME = "feature_selection_report.csv"

ROOM_MAPPING = {
    "A1-01a": "A1_01a",
    "A1-201": "A1_201",
    "A1-IT": "A1_IT",
    "A3-01e": "A3_01e",
}

METRIC_PREFIXES = {
    "job": "lepton__aec2__acn__job__",
    "worker": "lepton__aec2__acn__worker__",
}

UTIL_METRICS = {
    "cpu_util",
    "gpu_memory_util",
    "gpu_util",
    "memory_util",
}
SUM_METRICS = {
    "gpu_memory_amount",
    "gpu_memory_total",
    "gpu_power_usage",
    "memory_amount",
    "memory_total",
}
GPU_UTIL_BUSY_THRESHOLD = 0.3
GPU_POWER_HIGH_THRESHOLD = 200.0
ROLLING_WINDOWS = (3, 12)
FEATURE_MISSING_RATIO_THRESHOLD = 0.30
FEATURE_SCORE_THRESHOLD = 0.08
FEATURE_MIN_KEEP = 16
FEATURE_MAX_KEEP = 64
VOLATILITY_WINDOW = 12
VOLATILITY_MIN_PERIODS = 6

POINT_PATTERN = re.compile(r"\[(\d+),\s*\"?([^\]\",]+)\"?\]")
METRIC_PATTERN = re.compile(r"lepton__aec2__acn__(?:job|worker)__(.+?)_merged_")

csv.field_size_limit(sys.maxsize)

STRUCTURAL_FEATURE_EXACT = {
    "jobs_total_gpu_power_usage_sum",
    "jobs_total_gpu_util_mean",
    "training_share_gpu_power_usage",
    "inference_share_gpu_power_usage",
    "pod_minus_jobs_gpu_power_usage_sum",
    "training_minus_inference_power_sum",
    "training_to_inference_power_ratio",
    "pod_overhead_ratio",
    "training_present",
    "inference_present",
    "pod_present",
    "training_active_jobs",
    "inference_active_jobs",
    "pod_active_count",
    "training_gpu_busy_ratio",
    "inference_gpu_busy_ratio",
    "pod_gpu_busy_ratio",
    "training_high_power_job_ratio",
    "inference_high_power_job_ratio",
    "pod_high_power_ratio",
    "pod_gpu_util_minus_jobs_gpu_util_mean",
    "computility_feature_coverage_count",
}

KEEP_SUFFIXES = (
    "_mean",
    "_sum",
    "_count",
    "_std",
    "_busy_ratio",
    "_high_power_ratio",
    "_diff_1",
    "_roll_mean_3",
    "_roll_mean_12",
)
DROP_SUFFIXES = (
    "_min",
    "_max",
    "_busy_count",
    "_high_power_count",
    "_diff_3",
    "_roll_std_12",
)


def parse_metric_points(raw_value: str) -> List[Tuple[int, float]]:
    if not raw_value or raw_value == "[]":
        return []
    return [(int(ts), float(value)) for ts, value in POINT_PATTERN.findall(raw_value)]


def extract_metric_name(file_path: Path) -> str:
    matched = METRIC_PATTERN.search(file_path.name)
    if not matched:
        raise ValueError(f"Cannot parse metric name from {file_path}")
    return matched.group(1)


@lru_cache(maxsize=None)
def normalize_timestamp(epoch_seconds: int) -> str:
    return (
        pd.Timestamp(epoch_seconds, unit="s", tz="UTC")
        .tz_convert("Asia/Shanghai")
        .tz_localize(None)
        .strftime("%Y-%m-%d %H:%M:%S")
    )


def expand_metric_file(file_path: Path) -> pd.DataFrame:
    metric_name = extract_metric_name(file_path)
    latest_values: Dict[Tuple[str, str], float] = {}
    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            object_id = (
                row.get("uid")
                or row.get("name")
                or row.get("pod_uid")
                or row.get("job_uid")
                or "unknown"
            )
            for epoch_seconds, metric_value in parse_metric_points(row.get("value", "")):
                latest_values[(normalize_timestamp(epoch_seconds), object_id)] = metric_value

    if not latest_values:
        return pd.DataFrame(columns=[TIME_COL])

    frame = pd.DataFrame(
        (
            {TIME_COL: count_data_time, "object_id": object_id, "value": metric_value}
            for (count_data_time, object_id), metric_value in latest_values.items()
        )
    )
    aggregated = frame.groupby(TIME_COL, sort=True)["value"].agg(["min", "max", "mean", "std", "count"])
    if metric_name in SUM_METRICS:
        aggregated["sum"] = frame.groupby(TIME_COL, sort=True)["value"].sum()
    if metric_name == "gpu_util":
        busy_mask = frame["value"] > GPU_UTIL_BUSY_THRESHOLD
        aggregated["busy_count"] = frame.assign(is_busy=busy_mask).groupby(TIME_COL, sort=True)["is_busy"].sum()
        aggregated["busy_ratio"] = aggregated["busy_count"].div(aggregated["count"].where(aggregated["count"] > 0))
    if metric_name == "gpu_power_usage":
        high_power_mask = frame["value"] > GPU_POWER_HIGH_THRESHOLD
        aggregated["high_power_count"] = frame.assign(is_high_power=high_power_mask).groupby(TIME_COL, sort=True)["is_high_power"].sum()
        aggregated["high_power_ratio"] = aggregated["high_power_count"].div(aggregated["count"].where(aggregated["count"] > 0))
    aggregated = aggregated.reset_index()
    aggregated["std"] = aggregated["std"].fillna(0.0)
    return aggregated.rename(columns={stat: f"{metric_name}_{stat}" for stat in aggregated.columns if stat != TIME_COL})


def merge_feature_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    merged: Optional[pd.DataFrame] = None
    for frame in frames:
        if frame.empty:
            continue
        if merged is None:
            merged = frame.copy()
        else:
            merged = merged.merge(frame, on=TIME_COL, how="outer")
    if merged is None:
        return pd.DataFrame(columns=[TIME_COL])
    return merged.sort_values(TIME_COL).reset_index(drop=True)


def build_source_features(source_dir: Path, source_group: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for file_path in sorted(source_dir.glob("*.csv")):
        metric_frame = expand_metric_file(file_path)
        renamed = metric_frame.rename(
            columns={
                column: f"{source_group}_{column}"
                for column in metric_frame.columns
                if column != TIME_COL
            }
        )
        frames.append(renamed)
    return merge_feature_frames(frames)


def build_presence_flag(df: pd.DataFrame, source_group: str) -> pd.Series:
    count_columns = [column for column in df.columns if column.startswith(f"{source_group}_") and column.endswith("_count")]
    if not count_columns:
        return pd.Series(0, index=df.index, dtype="int64")
    return df[count_columns].fillna(0).gt(0).any(axis=1).astype("int64")


def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    training_power = result.get("training_gpu_power_usage_sum", pd.Series(0.0, index=result.index)).fillna(0.0)
    inference_power = result.get("inference_gpu_power_usage_sum", pd.Series(0.0, index=result.index)).fillna(0.0)
    pod_power = result.get("pod_gpu_power_usage_sum", pd.Series(0.0, index=result.index)).fillna(0.0)

    result["jobs_total_gpu_power_usage_sum"] = training_power + inference_power

    gpu_util_mean_columns = [
        column
        for column in ["training_gpu_util_mean", "inference_gpu_util_mean"]
        if column in result.columns
    ]
    if gpu_util_mean_columns:
        result["jobs_total_gpu_util_mean"] = result[gpu_util_mean_columns].mean(axis=1, skipna=True)
    else:
        result["jobs_total_gpu_util_mean"] = pd.NA

    jobs_total_power = result["jobs_total_gpu_power_usage_sum"]
    safe_total = jobs_total_power.where(jobs_total_power > 0)
    result["training_share_gpu_power_usage"] = training_power.div(safe_total)
    result["inference_share_gpu_power_usage"] = inference_power.div(safe_total)
    result["pod_minus_jobs_gpu_power_usage_sum"] = pod_power - jobs_total_power
    result["training_minus_inference_power_sum"] = training_power - inference_power
    result["training_to_inference_power_ratio"] = training_power.div(inference_power.where(inference_power > 0))
    result["pod_overhead_ratio"] = pod_power.div(safe_total)

    result["training_present"] = build_presence_flag(result, "training")
    result["inference_present"] = build_presence_flag(result, "inference")
    result["pod_present"] = build_presence_flag(result, "pod")
    result["training_active_jobs"] = result.get("training_gpu_power_usage_count", pd.Series(0, index=result.index)).fillna(0).astype("int64")
    result["inference_active_jobs"] = result.get("inference_gpu_power_usage_count", pd.Series(0, index=result.index)).fillna(0).astype("int64")
    result["pod_active_count"] = result.get("pod_gpu_power_usage_count", pd.Series(0, index=result.index)).fillna(0).astype("int64")
    result["training_gpu_busy_ratio"] = result.get("training_gpu_util_busy_ratio", pd.Series(pd.NA, index=result.index))
    result["inference_gpu_busy_ratio"] = result.get("inference_gpu_util_busy_ratio", pd.Series(pd.NA, index=result.index))
    result["pod_gpu_busy_ratio"] = result.get("pod_gpu_util_busy_ratio", pd.Series(pd.NA, index=result.index))
    result["training_high_power_job_ratio"] = result.get("training_gpu_power_usage_high_power_ratio", pd.Series(pd.NA, index=result.index))
    result["inference_high_power_job_ratio"] = result.get("inference_gpu_power_usage_high_power_ratio", pd.Series(pd.NA, index=result.index))
    result["pod_high_power_ratio"] = result.get("pod_gpu_power_usage_high_power_ratio", pd.Series(pd.NA, index=result.index))
    result["pod_gpu_util_minus_jobs_gpu_util_mean"] = result.get("pod_gpu_util_mean", pd.Series(pd.NA, index=result.index)) - result["jobs_total_gpu_util_mean"]
    result["computility_feature_coverage_count"] = (
        result["training_present"] + result["inference_present"] + result["pod_present"]
    )
    return result


def add_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    key_series = [
        "training_gpu_power_usage_sum",
        "inference_gpu_power_usage_sum",
        "pod_gpu_power_usage_sum",
        "jobs_total_gpu_power_usage_sum",
        "jobs_total_gpu_util_mean",
        "training_active_jobs",
        "inference_active_jobs",
    ]
    for column in key_series:
        if column not in result.columns:
            continue
        series = pd.to_numeric(result[column], errors="coerce")
        result[f"{column}_diff_1"] = series.diff(1)
        for window in ROLLING_WINDOWS:
            result[f"{column}_roll_mean_{window}"] = series.rolling(window=window, min_periods=1).mean()
        if column.endswith("_sum"):
            result[f"{column}_diff_3"] = series.diff(3)
            result[f"{column}_roll_std_12"] = series.rolling(window=12, min_periods=2).std()
    return result


def build_computility_dataset(room_dir: Path) -> pd.DataFrame:
    source_frames = []
    for source_group in ["training", "inference", "pod"]:
        source_path = room_dir / source_group
        if not source_path.exists():
            continue
        source_frames.append(build_source_features(source_path, source_group))

    merged = merge_feature_frames(source_frames)
    if merged.empty:
        return merged
    merged = add_structural_features(merged)
    merged = add_dynamic_features(merged)
    return merged.sort_values(TIME_COL).reset_index(drop=True)


def merge_power_with_computility(power_path: Path, computility_df: pd.DataFrame) -> pd.DataFrame:
    power_df = pd.read_csv(power_path)
    merged = power_df.merge(computility_df, on=TIME_COL, how="inner")
    return merged.sort_values(TIME_COL).reset_index(drop=True)


def _safe_abs_corr(left: pd.Series, right: pd.Series, method: str = "pearson") -> float:
    pair = pd.concat([pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")], axis=1).dropna()
    if len(pair) < 2:
        return 0.0
    if pair.iloc[:, 0].nunique(dropna=True) <= 1 or pair.iloc[:, 1].nunique(dropna=True) <= 1:
        return 0.0
    value = pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method)
    if pd.isna(value):
        return 0.0
    return float(abs(value))


def _feature_family_drop_reason(feature_name: str) -> Optional[str]:
    if feature_name in STRUCTURAL_FEATURE_EXACT:
        return None
    if feature_name.endswith(DROP_SUFFIXES):
        return "dropped_by_family_rule"
    if feature_name.endswith(KEEP_SUFFIXES):
        return None
    return "dropped_by_family_rule"


def build_feature_stats(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    target_series = pd.to_numeric(df[target_col], errors="coerce")
    target_volatility = target_series.diff().abs().rolling(
        window=VOLATILITY_WINDOW,
        min_periods=VOLATILITY_MIN_PERIODS,
    ).mean()
    records: List[Dict[str, object]] = []

    for feature_name in df.columns:
        if feature_name in {TIME_COL, target_col}:
            continue
        feature_series = pd.to_numeric(df[feature_name], errors="coerce")
        feature_volatility = feature_series.diff().abs().rolling(
            window=VOLATILITY_WINDOW,
            min_periods=VOLATILITY_MIN_PERIODS,
        ).mean()
        records.append(
            {
                "feature_name": feature_name,
                "missing_ratio": float(feature_series.isna().mean()),
                "nunique_non_na": int(feature_series.nunique(dropna=True)),
                "std_non_na": float(feature_series.dropna().std()) if feature_series.notna().any() else 0.0,
                "level_pearson": _safe_abs_corr(feature_series, target_series, method="pearson"),
                "level_spearman": _safe_abs_corr(feature_series, target_series, method="spearman"),
                "vol_pearson": _safe_abs_corr(feature_volatility, target_volatility, method="pearson"),
                "vol_spearman": _safe_abs_corr(feature_volatility, target_volatility, method="spearman"),
            }
        )

    stats_df = pd.DataFrame(records)
    if stats_df.empty:
        return stats_df
    stats_df["final_score"] = (
        0.35 * stats_df["level_pearson"]
        + 0.25 * stats_df["level_spearman"]
        + 0.25 * stats_df["vol_pearson"]
        + 0.15 * stats_df["vol_spearman"]
    )
    stats_df["keep"] = False
    stats_df["drop_reason"] = ""
    return stats_df


def filter_feature_candidates(stats_df: pd.DataFrame) -> pd.DataFrame:
    filtered = stats_df.copy()
    if filtered.empty:
        return filtered

    missing_mask = filtered["missing_ratio"] > FEATURE_MISSING_RATIO_THRESHOLD
    filtered.loc[missing_mask, "drop_reason"] = "dropped_by_missing_ratio"

    constant_mask = (
        filtered["drop_reason"].eq("")
        & ((filtered["nunique_non_na"] <= 1) | (filtered["std_non_na"] == 0))
    )
    filtered.loc[constant_mask, "drop_reason"] = "dropped_by_constant"

    family_mask = filtered["drop_reason"].eq("")
    filtered.loc[family_mask, "drop_reason"] = filtered.loc[family_mask, "feature_name"].map(_feature_family_drop_reason).fillna("")
    return filtered


def select_features_by_score(stats_df: pd.DataFrame) -> List[str]:
    candidates = stats_df[stats_df["drop_reason"].eq("")].copy()
    if candidates.empty:
        return []

    candidates = candidates.sort_values(["final_score", "feature_name"], ascending=[False, True]).reset_index(drop=True)
    threshold_selected = candidates[candidates["final_score"] >= FEATURE_SCORE_THRESHOLD]["feature_name"].tolist()
    if len(threshold_selected) < FEATURE_MIN_KEEP:
        selected = candidates.head(min(FEATURE_MIN_KEEP, len(candidates)))["feature_name"].tolist()
    else:
        selected = threshold_selected[:FEATURE_MAX_KEEP]
    return selected


def write_selected_dataset(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    selected_cols: List[str],
    selected_output_path: Path,
    report_output_path: Path,
) -> None:
    ordered_columns = [TIME_COL, POWER_TARGET_COL] + selected_cols
    df.loc[:, ordered_columns].to_csv(selected_output_path, index=False, encoding="utf-8-sig")

    report_df = stats_df.copy()
    if not report_df.empty:
        selected_set = set(selected_cols)
        report_df["keep"] = report_df["feature_name"].isin(selected_set)
        selected_order = {name: idx for idx, name in enumerate(selected_cols)}
        remaining = report_df["drop_reason"].eq("")
        threshold_fail = remaining & ~report_df["keep"] & (report_df["final_score"] < FEATURE_SCORE_THRESHOLD)
        report_df.loc[threshold_fail, "drop_reason"] = "dropped_by_score_threshold"
        capped = remaining & ~report_df["keep"] & (report_df["final_score"] >= FEATURE_SCORE_THRESHOLD)
        report_df.loc[capped, "drop_reason"] = "dropped_by_score_rank_cap"
        report_df.loc[report_df["keep"], "drop_reason"] = "kept"
        report_df["selected_rank"] = report_df["feature_name"].map(selected_order)
        report_df = report_df.sort_values(
            ["keep", "selected_rank", "final_score", "feature_name"],
            ascending=[False, True, False, True],
            na_position="last",
        ).drop(columns=["selected_rank"])
    report_df.to_csv(report_output_path, index=False, encoding="utf-8-sig")


def process_room(computility_room_dir: Path, demand_room_dir: Path) -> Dict[str, Path]:
    computility_df = build_computility_dataset(computility_room_dir)
    if computility_df.empty:
        raise ValueError(f"No computility points found under {computility_room_dir}")

    demand_room_dir.mkdir(parents=True, exist_ok=True)
    df_computility_path = demand_room_dir / COMPUTILITY_OUTPUT_NAME
    computility_df.to_csv(df_computility_path, index=False, encoding="utf-8-sig")

    power_path = demand_room_dir / "df_power.csv"
    if not power_path.exists():
        raise FileNotFoundError(f"Missing df_power.csv at {power_path}")
    merged_df = merge_power_with_computility(power_path, computility_df)
    merged_path = demand_room_dir / MERGED_OUTPUT_NAME
    merged_df.to_csv(merged_path, index=False, encoding="utf-8-sig")
    stats_df = build_feature_stats(merged_df, POWER_TARGET_COL)
    filtered_stats = filter_feature_candidates(stats_df)
    selected_cols = select_features_by_score(filtered_stats)
    selected_path = demand_room_dir / SELECTED_OUTPUT_NAME
    report_path = demand_room_dir / FEATURE_REPORT_OUTPUT_NAME
    write_selected_dataset(merged_df, filtered_stats, selected_cols, selected_path, report_path)

    return {
        "df_computility": df_computility_path,
        "df": merged_path,
        "df_selected": selected_path,
        "feature_selection_report": report_path,
    }


def process_all_rooms(computility_root: Path, demand_root: Path) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for source_room, target_room in ROOM_MAPPING.items():
        computility_room_dir = computility_root / source_room
        if not computility_room_dir.exists():
            continue
        demand_room_dir = demand_root / target_room
        output_paths = process_room(computility_room_dir, demand_room_dir)
        results.append(
            {
                "room": source_room,
                **output_paths,
            }
        )
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build df_computility.csv and merged df.csv from AIDC computility data.")
    parser.add_argument(
        "--computility-root",
        type=Path,
        default=Path("dataset/aidc_electricity_computility/electricity/算力数据"),
        help="Root directory containing AIDC computility room folders.",
    )
    parser.add_argument(
        "--demand-root",
        type=Path,
        default=Path("dataset/aidc_electricity_computility/electricity/2026-06-11/demand_load"),
        help="Demand-load root directory containing df_power.csv room folders.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    results = process_all_rooms(args.computility_root, args.demand_root)
    for result in results:
        print(
            f"{result['room']}: saved {result['df_computility']} and {result['df']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
