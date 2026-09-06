"""Strict local batch artifact manifests; validate before committing completion.

Pickles are only loaded from trusted local runtime output, after digest checks.
This is an integrity gate, not a sandbox for untrusted pickle files.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from forecasting_core.artifacts import ForecastModelBundle, QuantileGrid
from model_testing.geometry import TimeGeometry, scheduled_origin_indices
from model_pipeline.run_state import require_completed_state


REQUIRED_ARTIFACTS = frozenset({
    "model", "model_schema", "backtest", "scores", "horizon_scores",
    "forecast", "resolved_config", "result_metadata",
})


def artifact_paths(result: Any) -> dict[str, str]:
    paths = {
        "model": result.model_dir / "model.pkl",
        "model_schema": result.model_dir / "resolved_model.json",
        "backtest": result.test_dir / "cv_plot_df.csv",
        "scores": result.test_dir / "test_scores_df.csv",
        "horizon_scores": result.test_dir / "test_scores_horizon_df.csv",
        "forecast": result.forecast_dir / "prediction.csv",
        "resolved_config": result.forecast_dir / "resolved_config.json",
        "result_metadata": result.test_dir / "result_metadata.json",
    }
    if result.bundle.probabilistic_spec.mode == "quantile":
        paths["probabilistic_scores"] = result.test_dir / "test_scores_probabilistic_df.csv"
    if (result.model_dir / "run_state.json").exists():
        paths["lifecycle"] = result.model_dir / "run_state.json"
    return {key: str(path.resolve()) for key, path in paths.items()}


def artifact_digests(artifacts: Mapping[str, str]) -> dict[str, str]:
    result = {}
    for key, path in artifacts.items():
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        result[key] = digest.hexdigest()
    return result


def validate_time_grid(frame: pd.DataFrame, times: pd.DatetimeIndex, *, rows_per_time: int) -> None:
    observed = pd.to_datetime(frame["time"], errors="raise").value_counts().sort_index()
    if not observed.index.equals(times) or not (observed == rows_per_time).all():
        raise ValueError("batch prediction time grid mismatch")


def validate_artifacts(task: Mapping[str, Any], *, require_digests: bool = True) -> None:
    artifacts = task.get("artifacts")
    if not isinstance(artifacts, Mapping) or not REQUIRED_ARTIFACTS <= artifacts.keys():
        raise ValueError("batch artifacts missing required manifest keys")
    paths = {key: Path(value) for key, value in artifacts.items()}
    if len({path.resolve() for path in paths.values()}) != len(paths):
        raise ValueError("batch artifact paths must be distinct")
    if any(not path.is_file() or path.stat().st_size == 0 for path in paths.values()):
        raise ValueError("batch artifacts missing or empty")
    if require_digests and task.get("artifact_sha256") != artifact_digests(artifacts):
        raise ValueError("batch artifact digest mismatch; completion requires revalidation")
    resolved = json.loads(paths["resolved_config"].read_text())
    fingerprint = task.get("config_fingerprint")
    if not fingerprint or resolved.get("config_fingerprint") != fingerprint:
        raise ValueError("batch artifact config fingerprint mismatch")
    identity = task.get("result_identity")
    if not identity or any(
        path.parent.name != identity and not (
            path.parent.name in {"pretrained_models", "results_test", "results_forecast"}
            and path.parent.parent.name == identity
        ) for path in paths.values()
    ):
        raise ValueError("batch artifact result identity mismatch")
    with paths["model"].open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, ForecastModelBundle) or bundle.schema_version != 2:
        raise ValueError("batch model must be a schema-2 ForecastModelBundle")
    if bundle.config_fingerprint != fingerprint:
        raise ValueError("batch bundle fingerprint mismatch")
    model_schema = json.loads(paths["model_schema"].read_text())
    if model_schema.get("config_fingerprint") != fingerprint:
        raise ValueError("batch model schema fingerprint mismatch")
    metadata = json.loads(paths["result_metadata"].read_text())
    if not isinstance(metadata, dict):
        raise ValueError("batch result metadata must be a mapping")
    runtime = resolved["runtime"]
    if runtime.get("lifecycle_schema_version") == 1:
        if "lifecycle" not in paths:
            raise ValueError("batch artifacts missing run completion state")
        require_completed_state(json.loads(paths["lifecycle"].read_text()), fingerprint)
    holdout = runtime["holdout"]
    windows = holdout["windows"]
    expected_folds = int(resolved["validation"]["fold_count"])
    if len(windows) != expected_folds or expected_folds < 1:
        raise ValueError("batch backtest metadata fold count mismatch")
    forecast = pd.read_csv(paths["forecast"])
    backtest = pd.read_csv(paths["backtest"])
    n_series, horizon, n_targets = bundle.dimensions
    targets = set(bundle.target_order)
    for frame, keys in ((forecast, ["series_id", "time", "target"]),
                        (backtest, ["series_id", "time", "target", "window"])):
        if not set(keys + ["predict_value"]) <= set(frame.columns):
            raise ValueError("batch prediction CSV missing canonical columns")
        if frame.empty or frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError("batch prediction CSV has missing/duplicate canonical keys")
        if set(frame["target"]) != targets or frame["series_id"].nunique() != n_series:
            raise ValueError("batch prediction CSV target/series axes mismatch")
        pd.to_datetime(frame["time"], errors="raise")
        prediction_columns = [col for col in frame if col.startswith("predict_")]
        if not np.isfinite(frame[prediction_columns].to_numpy(dtype=float)).all():
            raise ValueError("batch prediction CSV contains nonfinite predictions")
    if len(forecast) != n_series * horizon * n_targets:
        raise ValueError("batch forecast row count mismatch")
    if "actual_value" not in backtest or not np.isfinite(backtest["actual_value"]).all():
        raise ValueError("batch backtest actuals missing/nonfinite")
    if backtest["window"].nunique() != expected_folds:
        raise ValueError("batch backtest CSV fold count mismatch")
    expected_counts = {}
    offset = pd.tseries.frequencies.to_offset(resolved["problem"]["freq"])
    formal_origin = runtime.get("forecast_origin", resolved["validation"].get("forecast_origin"))
    if formal_origin is not None:
        formal_origin = pd.Timestamp(formal_origin)
        validate_time_grid(
            forecast, pd.date_range(formal_origin + offset, periods=horizon, freq=offset),
            rows_per_time=n_series * n_targets,
        )
    for window in windows:
        window_horizon = horizon
        if holdout.get("mode") == "calendar_month":
            offset = pd.tseries.frequencies.to_offset(resolved["problem"]["freq"])
            times = pd.date_range(window["label_start"],
                                  pd.Timestamp(window["label_end"]) + offset,
                                  freq=offset, inclusive="left")
            window_horizon = len(times)
        expected_counts[int(window["window"])] = n_series * n_targets * window_horizon
        if "label_start" in window and "label_end" in window:
            times = pd.date_range(window["label_start"], window["label_end"], freq=offset)
            validate_time_grid(
                backtest.loc[backtest["window"] == window["window"]], times,
                rows_per_time=n_series * n_targets,
            )
        if resolved["validation"].get("schedule_mode") == "intraday":
            if formal_origin is None:
                raise ValueError("intraday artifacts require resolved forecast_origin")
            window_origin = pd.Timestamp(window["origin"])
            if not scheduled_origin_indices(
                (window_origin,), TimeGeometry(offset=offset, horizon=horizon), formal_origin,
                stride_steps=int(resolved["validation"]["stride_steps"]),
            ):
                raise ValueError("batch intraday backtest schedule mismatch")
    actual_counts = {int(key): int(value) for key, value in backtest.groupby("window").size().items()}
    if expected_counts != actual_counts:
        raise ValueError("batch backtest per-window geometry mismatch")
    for name in ("scores", "horizon_scores"):
        scores = pd.read_csv(paths[name])
        if scores.empty or not {"target", "scope"} <= set(scores.columns):
            raise ValueError(f"batch {name} schema missing target/scope")
    if bundle.probabilistic_spec.mode == "quantile":
        if "probabilistic_scores" not in paths:
            raise ValueError("batch quantile results missing probabilistic scores")
        grid = QuantileGrid(bundle.probabilistic_spec.quantiles)
        columns = [grid.column_name(level) for level in grid.levels]
        if any(not set(columns) <= set(frame.columns) for frame in (forecast, backtest)):
            raise ValueError("batch quantile CSV grid mismatch")
        if pd.read_csv(paths["probabilistic_scores"]).empty:
            raise ValueError("batch probabilistic scores empty")


def artifacts_complete(task: Mapping[str, Any]) -> bool:
    try:
        validate_artifacts(task)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, EOFError, pickle.UnpicklingError):
        return False
    return True
