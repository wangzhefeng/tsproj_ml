"""ESS 策略特征 v2 的加载、因果构建、校验与落盘流水线。"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import yaml

from .contracts import (
    CRITICAL_FUTURE_COLUMNS,
    FORBIDDEN_FUTURE_NAMES,
    FORBIDDEN_FUTURE_PATTERNS,
    LAG_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    PLAN_CYCLE_FEATURE_COLUMNS,
    SIMILAR_DAY_FEATURE_COLUMNS,
)
from .joint_clustering import (
    JointClusterArtifact,
    JointClusteringConfig,
    build_joint_lag_features,
    fit_joint_cluster_artifact,
)
from .profiles import summarize_dispatch_profiles
from .similar_day import SimilarDayConfig, estimate_similar_day_template
from .states import OperatingThresholds, encode_actual_operating_state, encode_plan_direction
from .windows import audit_history_timestamps, dispatch_cycle_start, validate_future_timestamps


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "dataset" / "aidc_ess_selfuse_load"
PROFILE_RENAME = {
    "max_ramp_kw": "max_ramp",
}


@dataclass(frozen=True)
class RoutePaths:
    target_path: Path
    endogenous_path: Path
    plan_path: Path


@dataclass(frozen=True)
class StrategyFeatureConfig:
    logic_version: int
    data_start: pd.Timestamp
    as_of_time: pd.Timestamp
    forecast_steps: int
    freq: str
    points_per_day: int
    calendar_start_hour: int
    dispatch_start_hour: int
    thresholds: OperatingThresholds
    similar_day: SimilarDayConfig
    joint_clustering: JointClusteringConfig
    joint_reference_fit_end: pd.Timestamp
    routes: dict[str, RoutePaths]
    data_root: Path
    output_dir: Path


@dataclass(frozen=True)
class RouteBuildResult:
    history: pd.DataFrame
    future: pd.DataFrame
    calendar_day_quality: pd.DataFrame
    dispatch_cycle_summary: pd.DataFrame
    similar_day_matches: pd.DataFrame
    joint_cluster_assignments: pd.DataFrame
    joint_cluster_artifact: JointClusterArtifact
    audit: dict[str, Any]


def _require_mapping(value, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _resolve_path(data_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def load_strategy_config(
    config_path: str | Path,
    data_root: str | Path | None = None,
) -> StrategyFeatureConfig:
    """加载并校验独立 v2 YAML；相对数据路径默认落在 ESS 数据根目录。"""
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    if not isinstance(raw, Mapping):
        raise ValueError("strategy feature config must be a mapping")

    required = {
        "logic_version",
        "data_start",
        "as_of_time",
        "forecast_steps",
        "freq",
        "points_per_day",
        "calendar_day",
        "dispatch_cycle",
        "states",
        "similar_day",
        "joint_clustering",
        "routes",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"missing config fields: {missing}")
    if raw["logic_version"] != 2:
        raise ValueError("logic_version must be 2")
    if raw["freq"] != "5min" or raw["points_per_day"] != 288:
        raise ValueError("Phase 3 supports only freq=5min and points_per_day=288")
    forecast_steps = int(raw["forecast_steps"])
    if forecast_steps != int(raw["points_per_day"]):
        raise ValueError("forecast_steps must equal points_per_day in the natural-day v2 pipeline")

    calendar = _require_mapping(raw["calendar_day"], "calendar_day")
    dispatch = _require_mapping(raw["dispatch_cycle"], "dispatch_cycle")
    if calendar.get("start_hour") != 0:
        raise ValueError("calendar_day.start_hour must be 0")
    if dispatch.get("start_hour") != 22:
        raise ValueError("dispatch_cycle.start_hour must be 22")

    states = _require_mapping(raw["states"], "states")
    thresholds = OperatingThresholds(
        charge_power=float(states["actual_charge_threshold_kw"]),
        discharge_power=float(states["actual_discharge_threshold_kw"]),
    )
    similar = _require_mapping(raw["similar_day"], "similar_day")
    similar_day = SimilarDayConfig(
        lookback_days=int(similar["lookback_days"]),
        k_neighbors=int(similar["k_neighbors"]),
        min_history_days=int(similar["min_history_days"]),
        robust_template_days=int(similar["robust_template_days"]),
        q75=float(similar["novelty_low_quantile"]),
        q95=float(similar["novelty_high_quantile"]),
        curve_weight=float(similar["curve_weight"]),
        duration_energy_weight=float(similar["duration_energy_weight"]),
        transition_weight=float(similar["transition_weight"]),
        power_scale=float(similar["power_scale_kw"]),
        count_scale=float(similar["count_scale"]),
        min_effective_samples=float(similar["min_effective_samples"]),
    )

    joint = _require_mapping(raw["joint_clustering"], "joint_clustering")
    if joint.get("enabled") is not True:
        raise ValueError("joint_clustering.enabled must be true for the P6 pipeline")
    joint_reference_fit_end = pd.Timestamp(joint["reference_fit_end"]).normalize()
    joint_clustering = JointClusteringConfig(
        pca_variance_ratio=float(joint["pca_variance_ratio"]),
        candidate_clusters=tuple(int(value) for value in joint["candidate_clusters"]),
        max_clusters=int(joint["max_clusters"]),
        rare_cluster_min_days=int(joint["rare_cluster_min_days"]),
        random_state=int(joint["random_state"]),
        n_init=int(joint["n_init"]),
    )
    if joint_clustering.max_clusters != 5:
        raise ValueError("joint_clustering.max_clusters must be 5")

    route_config = _require_mapping(raw["routes"], "routes")
    if set(route_config) != {"A", "B"}:
        raise ValueError("routes must contain exactly A and B")
    routes = {}
    for route, values in route_config.items():
        route_values = _require_mapping(values, f"routes.{route}")
        routes[route] = RoutePaths(
            target_path=_resolve_path(root, route_values["target_path"]),
            endogenous_path=_resolve_path(root, route_values["endogenous_path"]),
            plan_path=_resolve_path(root, route_values["plan_path"]),
        )

    data_start = pd.Timestamp(raw["data_start"])
    as_of_time = pd.Timestamp(raw["as_of_time"])
    if data_start != data_start.normalize():
        raise ValueError("data_start must be a natural-day boundary")
    if as_of_time != as_of_time.normalize() + pd.Timedelta(hours=23, minutes=55):
        raise ValueError("as_of_time must be the final 5-minute slot of a natural day")
    if not data_start <= joint_reference_fit_end <= as_of_time.normalize():
        raise ValueError("joint_clustering.reference_fit_end must be within the history range")

    return StrategyFeatureConfig(
        logic_version=2,
        data_start=data_start,
        as_of_time=as_of_time,
        forecast_steps=forecast_steps,
        freq=str(raw["freq"]),
        points_per_day=int(raw["points_per_day"]),
        calendar_start_hour=0,
        dispatch_start_hour=22,
        thresholds=thresholds,
        similar_day=similar_day,
        joint_clustering=joint_clustering,
        joint_reference_fit_end=joint_reference_fit_end,
        routes=routes,
        data_root=root,
        output_dir=root / "forecasting_data" / "strategy_features",
    )


def _load_csv(path: Path, required_columns: list[str], numeric_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = [column for column in required_columns if column not in frame]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    frame = frame.loc[:, required_columns].copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    if frame["time"].isna().any():
        raise ValueError(f"{path} contains NaT timestamps")
    if frame["time"].duplicated().any():
        raise ValueError(f"{path} contains duplicate timestamps")
    timestamps = pd.DatetimeIndex(frame["time"])
    off_grid = (
        (timestamps.minute % 5 != 0)
        | (timestamps.second != 0)
        | (timestamps.microsecond != 0)
        | (timestamps.nanosecond != 0)
    )
    if off_grid.any():
        raise ValueError(f"{path} contains timestamps outside the 5-minute grid")
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{path} column {column} must contain only finite values")
    return frame.sort_values("time").reset_index(drop=True)


def _validate_history_boundary(frame: pd.DataFrame, config: StrategyFeatureConfig, name: str) -> None:
    timestamps = pd.DatetimeIndex(frame["time"])
    if len(timestamps) == 0 or timestamps.min() > config.data_start:
        raise ValueError(f"{name} does not cover data_start")
    if config.as_of_time not in timestamps:
        raise ValueError(f"{name} does not contain as_of_time")


def _missing_times(index: pd.DatetimeIndex, actual: pd.DatetimeIndex) -> list[str]:
    return [timestamp.isoformat() for timestamp in index.difference(actual)]


def _input_metadata(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    timestamps = pd.DatetimeIndex(frame["time"])
    off_grid = (
        (timestamps.minute % 5 != 0)
        | (timestamps.second != 0)
        | (timestamps.microsecond != 0)
        | (timestamps.nanosecond != 0)
    )
    return {
        "path": str(path),
        "mtime": path.stat().st_mtime,
        "rows": int(len(frame)),
        "time_min": timestamps.min().isoformat() if len(frame) else None,
        "time_max": timestamps.max().isoformat() if len(frame) else None,
        "duplicate_timestamps": int(frame["time"].duplicated().sum()),
        "off_grid_timestamps": int(off_grid.sum()),
    }


def _profile_summary(frame: pd.DataFrame, prefix: str, state_prefix: str) -> pd.DataFrame:
    summary = summarize_dispatch_profiles(
        frame,
        time_col="time",
        power_col="power_kw",
        state_prefix=state_prefix,
    )
    renamed = {}
    for column in summary.columns:
        if column == "cycle_start":
            continue
        clean = PROFILE_RENAME.get(column, column)
        renamed[column] = f"{prefix}_{clean}"
    return summary.rename(columns=renamed)


def _build_plan_summary(plan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    plan_states = encode_plan_direction(plan["pcs_plan"])
    profile_input = pd.concat(
        [
            plan[["time"]].reset_index(drop=True),
            plan["pcs_plan"].rename("power_kw").reset_index(drop=True),
            plan_states.reset_index(drop=True),
        ],
        axis=1,
    )
    plan_with_states = pd.concat([plan.reset_index(drop=True), plan_states.reset_index(drop=True)], axis=1)
    summary = _profile_summary(profile_input, "plan_cycle", "plan_direction")
    cycle_counts = (
        plan_with_states.assign(cycle_start=dispatch_cycle_start(plan_with_states["time"]))
        .groupby("cycle_start", sort=True)
        .size()
        .to_frame("plan_cycle_point_count")
        .reset_index()
    )
    summary = summary.merge(cycle_counts, on="cycle_start", how="left")
    summary["plan_cycle_complete"] = summary["plan_cycle_point_count"].eq(288)
    return summary, plan_with_states


def _validate_required_plan_cycles(
    grid: pd.DatetimeIndex,
    plan_summary: pd.DataFrame,
    scope: str,
) -> None:
    required_cycles = pd.DatetimeIndex(dispatch_cycle_start(grid).unique())
    complete = (
        plan_summary.set_index("cycle_start")["plan_cycle_complete"]
        .reindex(required_cycles)
        .astype("boolean")
        .fillna(False)
    )
    if not bool(complete.all()):
        first_incomplete = required_cycles[~complete.to_numpy(dtype=bool)][0]
        raise ValueError(
            f"{scope} plan dispatch cycle {first_incomplete} is incomplete"
        )


def _build_actual_summary(
    endogenous: pd.DataFrame,
    plan_with_states: pd.DataFrame,
    thresholds: OperatingThresholds,
    as_of_time: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual = endogenous.loc[endogenous["time"] <= as_of_time, ["time", "pcs_power"]].copy()
    actual_states = encode_actual_operating_state(actual["pcs_power"], thresholds)
    actual_with_states = pd.concat([actual.reset_index(drop=True), actual_states.reset_index(drop=True)], axis=1)
    profile_input = pd.concat(
        [
            actual[["time"]].reset_index(drop=True),
            actual["pcs_power"].rename("power_kw").reset_index(drop=True),
            actual_states.reset_index(drop=True),
        ],
        axis=1,
    )
    summary = _profile_summary(profile_input, "actual_cycle", "actual_operating")

    agreement = actual_with_states.merge(plan_with_states, on="time", how="inner")
    agreement["cycle_start"] = dispatch_cycle_start(agreement["time"])
    actual_labels = agreement[
        ["actual_operating_charge", "actual_operating_standby", "actual_operating_discharge"]
    ].to_numpy().argmax(axis=1)
    plan_labels = agreement[
        ["plan_direction_charge", "plan_direction_standby", "plan_direction_discharge"]
    ].to_numpy().argmax(axis=1)
    agreement["state_agreement"] = (actual_labels == plan_labels).astype(float)
    agreement_summary = agreement.groupby("cycle_start", sort=True).agg(
        actual_cycle_point_count=("time", "count"),
        actual_cycle_plan_state_agreement=("state_agreement", "mean"),
    ).reset_index()
    summary = summary.merge(agreement_summary, on="cycle_start", how="left")
    summary["actual_cycle_complete"] = summary["actual_cycle_point_count"].eq(288)
    return summary, actual_with_states


def _broadcast_plan_features(grid: pd.DatetimeIndex, plan_summary: pd.DataFrame) -> pd.DataFrame:
    lookup = plan_summary.set_index("cycle_start")
    cycle_keys = dispatch_cycle_start(grid)
    values = lookup.reindex(cycle_keys).reset_index(drop=True)
    output = pd.DataFrame(index=grid)
    for column in PLAN_CYCLE_FEATURE_COLUMNS:
        output[column] = pd.to_numeric(values[column], errors="coerce").fillna(0.0).to_numpy()
    return output


def _last_completed_cycle_start(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return days.normalize() - pd.Timedelta(days=2) + pd.Timedelta(hours=22)


def _build_lag_features(
    grid: pd.DatetimeIndex,
    target_series: pd.Series,
    actual_series: pd.Series,
    actual_states: pd.DataFrame,
    actual_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    source_times = grid - pd.Timedelta(days=1)
    target_lag = target_series.reindex(source_times)
    actual_lag = actual_series.reindex(source_times)
    state_lookup = actual_states.set_index("time")
    state_lag = state_lookup.reindex(source_times)
    summary_lookup = actual_summary.set_index("cycle_start")
    completed_keys = _last_completed_cycle_start(grid)
    completed = summary_lookup.reindex(completed_keys)
    ready = (
        target_lag.notna().to_numpy()
        & actual_lag.notna().to_numpy()
        & state_lag[
            ["actual_operating_charge", "actual_operating_standby", "actual_operating_discharge"]
        ].notna().all(axis=1).to_numpy()
        & completed["actual_cycle_complete"]
        .astype("boolean")
        .fillna(False)
        .to_numpy(dtype=bool)
    )

    output = pd.DataFrame(0.0, index=grid, columns=LAG_FEATURE_COLUMNS)
    output.loc[ready, "ess_lag_288"] = target_lag.to_numpy()[ready]
    output.loc[ready, "pcs_actual_lag_288"] = actual_lag.to_numpy()[ready]
    for state in ("charge", "standby", "discharge"):
        output.loc[ready, f"actual_operating_{state}_lag_288"] = state_lag[
            f"actual_operating_{state}"
        ].to_numpy()[ready]
        output.loc[ready, f"last_completed_cycle_{state}_hours"] = completed[
            f"actual_cycle_{state}_hours"
        ].to_numpy()[ready]
    output.loc[ready, "last_completed_cycle_plan_state_agreement"] = completed[
        "actual_cycle_plan_state_agreement"
    ].to_numpy()[ready]
    output["lag_feature_ready"] = ready.astype(int)
    return output, pd.Series(source_times.where(ready, pd.NaT), index=grid)


def _complete_day_dictionary(series: pd.Series, days: pd.DatetimeIndex) -> dict[pd.Timestamp, np.ndarray]:
    output = {}
    for day in days:
        day_index = pd.date_range(day, periods=288, freq="5min")
        values = series.reindex(day_index).to_numpy(dtype=float)
        if np.isfinite(values).all():
            output[pd.Timestamp(day)] = values
    return output


def _build_similar_features(
    grid: pd.DatetimeIndex,
    target_days: pd.DatetimeIndex,
    plan_days: dict[pd.Timestamp, np.ndarray],
    target_history: dict[pd.Timestamp, np.ndarray],
    config: StrategyFeatureConfig,
    future: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = pd.DataFrame(0.0, index=grid, columns=SIMILAR_DAY_FEATURE_COLUMNS)
    match_rows = []
    for day in target_days:
        result = estimate_similar_day_template(
            day,
            plan_days[pd.Timestamp(day)],
            plan_days,
            target_history,
            config.similar_day,
            history_cutoff_day=config.as_of_time.normalize() if future else day,
        )
        day_mask = grid.normalize() == day
        if result.ready:
            similar_template = (
                result.similar_template
                if result.similar_template is not None
                else result.robust_template
            )
            similar_std = (
                result.similar_std
                if result.similar_std is not None
                else np.zeros(config.points_per_day)
            )
            novelty_score = 0.0
            if result.novelty_distance is not None:
                if result.novelty_q95 is not None and result.novelty_q95 > 1e-12:
                    novelty_score = result.novelty_distance / result.novelty_q95
                elif result.novelty_distance > 1e-12:
                    novelty_score = 1.0
            scalar_values = {
                "plan_nearest_day_distance": result.nearest_distance or 0.0,
                "plan_knn_mean_distance": result.knn_mean_distance or 0.0,
                "plan_novelty_score": novelty_score,
                "plan_is_novel": int(
                    result.novelty_distance is not None
                    and result.novelty_q95 is not None
                    and result.novelty_q95 > 1e-12
                    and result.novelty_distance >= result.novelty_q95
                ),
                "similar_day_effective_samples": result.n_effective,
                "template_gate_weight": result.gate,
                "template_feature_ready": 1,
            }
            for column, value in scalar_values.items():
                output.loc[day_mask, column] = value
            output.loc[day_mask, "plan_similar_day_ess_template"] = similar_template
            output.loc[day_mask, "plan_similar_day_template_std"] = similar_std
            output.loc[day_mask, "robust_recent_ess_template"] = result.robust_template
            output.loc[day_mask, "gated_ess_template"] = result.template

        if result.matches:
            for rank, match in enumerate(result.matches, start=1):
                match_rows.append(
                    {
                        "target_day": day,
                        "rank": rank,
                        "candidate_day": match.day,
                        "distance": match.distance,
                        "method": result.method,
                        "reason": result.reason,
                        "ready": int(result.ready),
                    }
                )
        else:
            match_rows.append(
                {
                    "target_day": day,
                    "rank": 0,
                    "candidate_day": pd.NaT,
                    "distance": np.nan,
                    "method": result.method,
                    "reason": result.reason,
                    "ready": int(result.ready),
                }
            )
    return output, pd.DataFrame(match_rows)


def _build_model_frame(
    grid: pd.DatetimeIndex,
    plan_series: pd.Series,
    plan_summary: pd.DataFrame,
    lag_features: pd.DataFrame,
    similar_features: pd.DataFrame,
    joint_features: pd.DataFrame,
) -> pd.DataFrame:
    plan_values = plan_series.reindex(grid)
    plan_states = encode_plan_direction(plan_values.reset_index(drop=True))
    frame = pd.DataFrame({"time": grid, "pcs_plan": plan_values.to_numpy(dtype=float)})
    frame = pd.concat([frame, plan_states.reset_index(drop=True)], axis=1)
    frame["plan_power_abs"] = frame["pcs_plan"].abs()
    frame = pd.concat(
        [
            frame,
            _broadcast_plan_features(grid, plan_summary).reset_index(drop=True),
            lag_features.reset_index(drop=True),
            similar_features.reset_index(drop=True),
            joint_features.reset_index(drop=True),
        ],
        axis=1,
    )
    frame = frame.loc[:, MODEL_FEATURE_COLUMNS]
    numeric = frame.drop(columns="time")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise AssertionError("model feature output contains non-finite values")
    return frame


def _calendar_quality(
    days: pd.DatetimeIndex,
    target_series: pd.Series,
    actual_series: pd.Series,
    plan_series: pd.Series,
) -> pd.DataFrame:
    rows = []
    for day in days:
        index = pd.date_range(day, periods=288, freq="5min")
        target_count = int(target_series.reindex(index).notna().sum())
        actual_count = int(actual_series.reindex(index).notna().sum())
        plan_count = int(plan_series.reindex(index).notna().sum())
        rows.append(
            {
                "calendar_day": day,
                "target_points": target_count,
                "actual_points": actual_count,
                "plan_points": plan_count,
                "target_complete": target_count == 288,
                "actual_complete": actual_count == 288,
                "plan_complete": plan_count == 288,
            }
        )
    return pd.DataFrame(rows)


def _schema_hash() -> str:
    return hashlib.sha256("\n".join(MODEL_FEATURE_COLUMNS).encode("utf-8")).hexdigest()


def _numeric_ranges(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        column: {
            "min": float(frame[column].min()),
            "max": float(frame[column].max()),
        }
        for column in frame.columns
        if column != "time"
    }


def _assert_future_contract(future: pd.DataFrame) -> None:
    forbidden = [
        column
        for column in future.columns
        if column in FORBIDDEN_FUTURE_NAMES
        or any(pattern.search(column) for pattern in FORBIDDEN_FUTURE_PATTERNS)
    ]
    if forbidden:
        raise AssertionError(f"future contains forbidden columns: {forbidden}")
    critical = future.loc[:, sorted(CRITICAL_FUTURE_COLUMNS)]
    if critical.isna().any().any() or not np.isfinite(critical.to_numpy(dtype=float)).all():
        raise AssertionError("critical future columns must be finite")


def _build_route(
    route: str,
    paths: RoutePaths,
    config: StrategyFeatureConfig,
) -> RouteBuildResult:
    target = _load_csv(paths.target_path, ["time", "value"], ["value"])
    endogenous = _load_csv(
        paths.endogenous_path,
        ["time", "ess_power", "pcs_power"],
        ["ess_power", "pcs_power"],
    )
    plan = _load_csv(paths.plan_path, ["time", "pcs_plan"], ["pcs_plan"])
    _validate_history_boundary(target, config, f"route {route} target")
    _validate_history_boundary(endogenous, config, f"route {route} endogenous")

    history_grid = pd.date_range(config.data_start, config.as_of_time, freq=config.freq)
    future_start = config.as_of_time.normalize() + pd.Timedelta(days=1)
    future_grid = pd.date_range(future_start, periods=config.forecast_steps, freq=config.freq)
    validate_future_timestamps(future_grid)

    plan_series = plan.set_index("time")["pcs_plan"]
    missing_history_plan = history_grid.difference(plan_series.index)
    if len(missing_history_plan):
        raise ValueError(f"history plan is incomplete; first missing {missing_history_plan[0]}")
    missing_future_plan = future_grid.difference(plan_series.index)
    if len(missing_future_plan):
        raise ValueError(f"future plan is incomplete; first missing {missing_future_plan[0]}")

    target_history_frame = target.loc[
        (target["time"] >= config.data_start) & (target["time"] <= config.as_of_time)
    ]
    endogenous_history_frame = endogenous.loc[
        (endogenous["time"] >= config.data_start)
        & (endogenous["time"] <= config.as_of_time)
    ]
    target_series = target_history_frame.set_index("time")["value"]
    actual_series = endogenous_history_frame.set_index("time")["pcs_power"]

    plan_summary, plan_with_states = _build_plan_summary(plan)
    _validate_required_plan_cycles(history_grid, plan_summary, "history")
    _validate_required_plan_cycles(future_grid, plan_summary, "future")
    actual_summary, actual_with_states = _build_actual_summary(
        endogenous_history_frame,
        plan_with_states,
        config.thresholds,
        config.as_of_time,
    )
    dispatch_summary = plan_summary.merge(actual_summary, on="cycle_start", how="outer").sort_values("cycle_start")

    history_lag, history_lag_sources = _build_lag_features(
        history_grid, target_series, actual_series, actual_with_states, actual_summary
    )
    future_lag, future_lag_sources = _build_lag_features(
        future_grid, target_series, actual_series, actual_with_states, actual_summary
    )

    history_days = pd.date_range(history_grid[0].normalize(), history_grid[-1].normalize(), freq="1D")
    future_days = pd.DatetimeIndex(future_grid.normalize().unique())
    all_days = history_days.append(future_days)
    plan_days = _complete_day_dictionary(plan_series, all_days)
    missing_plan_days = [day for day in all_days if day not in plan_days]
    if missing_plan_days:
        scope = "future" if missing_plan_days[0] in future_days else "history"
        raise ValueError(f"{scope} plan day {missing_plan_days[0].date()} is incomplete")
    target_days = _complete_day_dictionary(target_series, history_days)
    actual_days = _complete_day_dictionary(actual_series, history_days)

    joint_artifact = fit_joint_cluster_artifact(
        target_days,
        actual_days,
        plan_days,
        fit_end=config.joint_reference_fit_end,
        config=config.joint_clustering,
    )
    history_joint, history_joint_assignments = build_joint_lag_features(
        history_grid, joint_artifact, target_days, actual_days, plan_days
    )
    future_joint, future_joint_assignments = build_joint_lag_features(
        future_grid, joint_artifact, target_days, actual_days, plan_days
    )
    joint_assignments = pd.concat(
        [history_joint_assignments, future_joint_assignments], ignore_index=True
    )

    history_similar, history_matches = _build_similar_features(
        history_grid, history_days, plan_days, target_days, config, future=False
    )
    future_similar, future_matches = _build_similar_features(
        future_grid, future_days, plan_days, target_days, config, future=True
    )
    history = _build_model_frame(
        history_grid,
        plan_series,
        plan_summary,
        history_lag,
        history_similar,
        history_joint,
    )
    future = _build_model_frame(
        future_grid,
        plan_series,
        plan_summary,
        future_lag,
        future_similar,
        future_joint,
    )
    if list(history.columns) != MODEL_FEATURE_COLUMNS or list(future.columns) != MODEL_FEATURE_COLUMNS:
        raise AssertionError("history/future schema contract mismatch")
    if len(future) != config.forecast_steps:
        raise AssertionError("future row count does not match forecast_steps")
    _assert_future_contract(future)

    history_timestamp_audit = audit_history_timestamps(target_history_frame["time"])
    actual_timestamp_audit = audit_history_timestamps(endogenous_history_frame["time"])
    calendar_quality = _calendar_quality(
        history_days, target_series, actual_series, plan_series
    )
    matches = pd.concat([history_matches, future_matches], ignore_index=True)
    future_source_max = future_lag_sources.dropna().max()
    future_joint_rows = joint_assignments.loc[
        joint_assignments["target_day"].isin(future_days)
        & joint_assignments["ready"].eq(1)
    ]
    future_joint_source_max = future_joint_rows["source_day"].max()
    lag_leakage_pass = bool(
        pd.isna(future_source_max) or pd.Timestamp(future_source_max) <= config.as_of_time
    )
    joint_leakage_pass = bool(
        (
            pd.isna(future_joint_source_max)
            or pd.Timestamp(future_joint_source_max) <= config.as_of_time
        )
        and joint_artifact.fit_end <= config.as_of_time.normalize()
    )
    leakage_pass = lag_leakage_pass and joint_leakage_pass
    audit = {
        "logic_version": config.logic_version,
        "route": route,
        "data_start": config.data_start.isoformat(),
        "as_of_time": config.as_of_time.isoformat(),
        "forecast_steps": config.forecast_steps,
        "history_rows": int(len(history)),
        "future_rows": int(len(future)),
        "inputs": {
            "target": _input_metadata(paths.target_path, target),
            "endogenous": _input_metadata(paths.endogenous_path, endogenous),
            "plan": _input_metadata(paths.plan_path, plan),
        },
        "gaps": {
            "target_missing_timestamps": _missing_times(history_grid, pd.DatetimeIndex(target_series.index)),
            "actual_missing_timestamps": _missing_times(history_grid, pd.DatetimeIndex(actual_series.index)),
            "incomplete_target_days": [day.isoformat() for day in history_timestamp_audit.incomplete_days],
            "incomplete_actual_days": [day.isoformat() for day in actual_timestamp_audit.incomplete_days],
        },
        "excluded_calendar_days": [
            day.isoformat() for day in history_days if day not in target_days
        ],
        "fallback_counts": (
            matches.sort_values(["target_day", "rank"])
            .drop_duplicates("target_day", keep="first")["method"]
            .value_counts(dropna=False)
            .to_dict()
        ),
        "readiness_counts": {
            "history_lag": history["lag_feature_ready"].value_counts().to_dict(),
            "future_lag": future["lag_feature_ready"].value_counts().to_dict(),
            "history_template": history["template_feature_ready"].value_counts().to_dict(),
            "future_template": future["template_feature_ready"].value_counts().to_dict(),
            "history_joint": history["joint_cluster_feature_ready"].value_counts().to_dict(),
            "future_joint": future["joint_cluster_feature_ready"].value_counts().to_dict(),
        },
        "joint_clustering": {
            "fit_start": joint_artifact.fit_start.isoformat(),
            "fit_end": joint_artifact.fit_end.isoformat(),
            "reference_days": len(joint_artifact.reference_days),
            "selected_k": joint_artifact.selected_k,
            "silhouette_scores": joint_artifact.silhouette_scores,
            "cluster_counts": joint_artifact.cluster_counts,
            "rare_clusters": list(joint_artifact.rare_clusters),
            "pca_components": {
                view: int(pca.n_components_)
                for view, pca in joint_artifact.pcas.items()
            },
            "pca_explained_variance": {
                view: float(pca.explained_variance_ratio_.sum())
                for view, pca in joint_artifact.pcas.items()
            },
        },
        "similar_day_distance_quantiles": {
            str(quantile): float(matches["distance"].dropna().quantile(quantile))
            if matches["distance"].notna().any()
            else None
            for quantile in (0.5, 0.75, 0.95)
        },
        "schema_hash": _schema_hash(),
        "history_schema_hash": _schema_hash(),
        "future_schema_hash": _schema_hash(),
        "ranges": {
            "history": _numeric_ranges(history),
            "future": _numeric_ranges(future),
        },
        "coverage_checks": {
            "history_plan_complete": len(missing_history_plan) == 0,
            "future_plan_complete": len(missing_future_plan) == 0,
            "history_plan_dispatch_cycles_complete": True,
            "future_plan_dispatch_cycles_complete": True,
            "future_rows_match_forecast_steps": len(future) == config.forecast_steps,
        },
        "leakage_checks": {
            "future_sources_at_or_before_as_of": leakage_pass,
            "future_lag_source_min": (
                future_lag_sources.dropna().min().isoformat()
                if future_lag_sources.notna().any()
                else None
            ),
            "future_lag_source_max": (
                future_lag_sources.dropna().max().isoformat()
                if future_lag_sources.notna().any()
                else None
            ),
            "future_joint_source_max": (
                pd.Timestamp(future_joint_source_max).isoformat()
                if not pd.isna(future_joint_source_max)
                else None
            ),
            "joint_artifact_fit_end_at_or_before_as_of": (
                joint_artifact.fit_end <= config.as_of_time.normalize()
            ),
            "future_forbidden_column_matches": 0,
            "future_target_or_actual_rows_ignored": True,
        },
    }
    if not leakage_pass:
        raise AssertionError("future lag source exceeds as_of_time")
    return RouteBuildResult(
        history=history,
        future=future,
        calendar_day_quality=calendar_quality,
        dispatch_cycle_summary=dispatch_summary,
        similar_day_matches=matches,
        joint_cluster_assignments=joint_assignments,
        joint_cluster_artifact=joint_artifact,
        audit=audit,
    )


def _output_paths(config: StrategyFeatureConfig, route: str) -> dict[str, Path]:
    audit_dir = config.output_dir / "audit"
    artifact_dir = config.output_dir / "artifacts"
    fit_tag = config.joint_reference_fit_end.strftime("%Y%m%d")
    return {
        "history": config.output_dir / f"model_features_history_{route}.csv",
        "future": config.output_dir / f"model_features_future_{route}.csv",
        "calendar": audit_dir / f"calendar_day_quality_{route}.csv",
        "dispatch": audit_dir / f"dispatch_cycle_summary_{route}.csv",
        "matches": audit_dir / f"similar_day_matches_{route}.csv",
        "joint_assignments": (
            audit_dir / f"joint_cluster_assignments_{route}_fit-{fit_tag}.csv"
        ),
        "joint_artifact": (
            artifact_dir / f"joint_cluster_{route}_fit-{fit_tag}.joblib"
        ),
        "audit": audit_dir / f"feature_build_audit_{route}.json",
    }


def build_strategy_features(
    config_path: str | Path,
    *,
    data_root: str | Path | None = None,
    validate_only: bool = False,
    force: bool = False,
) -> dict[str, RouteBuildResult]:
    """构建 A/B 路 v2 特征；validate-only 完整执行但不创建任何输出。"""
    config = load_strategy_config(config_path, data_root=data_root)
    results = {
        route: _build_route(route, paths, config)
        for route, paths in config.routes.items()
    }
    if validate_only:
        return results

    paths_by_route = {route: _output_paths(config, route) for route in results}
    existing = [path for paths in paths_by_route.values() for path in paths.values() if path.exists()]
    if existing and not force:
        raise FileExistsError(
            f"v2 strategy feature outputs already exist; use --force: {existing[0]}"
        )

    for route, result in results.items():
        paths = paths_by_route[route]
        paths["history"].parent.mkdir(parents=True, exist_ok=True)
        paths["calendar"].parent.mkdir(parents=True, exist_ok=True)
        paths["joint_artifact"].parent.mkdir(parents=True, exist_ok=True)
        result.history.to_csv(paths["history"], index=False)
        result.future.to_csv(paths["future"], index=False)
        result.calendar_day_quality.to_csv(paths["calendar"], index=False)
        result.dispatch_cycle_summary.to_csv(paths["dispatch"], index=False)
        result.similar_day_matches.to_csv(paths["matches"], index=False)
        result.joint_cluster_assignments.to_csv(
            paths["joint_assignments"], index=False
        )
        joblib.dump(result.joint_cluster_artifact, paths["joint_artifact"])
        paths["audit"].write_text(
            json.dumps(result.audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return results
