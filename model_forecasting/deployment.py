"""Prediction from a self-contained canonical strategy bundle.

The deployment caller supplies feature rows compiled in ``bundle.input_schema``
column order.  This module owns all persisted preprocessing, selected-feature
alignment, strategy execution, quantile assembly, and target-space restoration;
it never reads model YAML or training/OOF caches.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from feature_engineering.selection import selected_indices_for_artifact
from forecasting_core.artifacts import ForecastModelBundle, MarginalForecastDistribution
from forecasting_core.specs import ForecastStrategySpec
from forecasting_core.tensors import MarginalQuantileForecastTensor, PointForecastTensor
from model_forecasting.forecaster import repair_marginal_quantile_crossing
from model_training.strategies import (
    CanonicalStrategyArtifact,
    TargetCoordinate,
    get_standard_executor,
)
from probabilistic.training import CanonicalMarginalQuantileArtifact


FeatureProvider = Callable[..., np.ndarray]


def predict_strategy_bundle(
    bundle: ForecastModelBundle,
    raw_design: np.ndarray,
    *,
    forecast_times: pd.DatetimeIndex,
    series_ids: tuple[Any, ...] | None = None,
    raw_feature_provider: FeatureProvider | None = None,
) -> PointForecastTensor | MarginalForecastDistribution:
    """Predict from one schema-2 strategy bundle without config or cache IO."""
    if not isinstance(bundle, ForecastModelBundle) or bundle.schema_version != 2:
        raise TypeError("bundle must be a schema-2 ForecastModelBundle")
    if bundle.strategy_spec is None or bundle.ensemble_spec is not None:
        raise ValueError("predict_strategy_bundle requires a single-model bundle")
    strategy = ForecastStrategySpec(**bundle.strategy_spec)
    artifact = bundle.model
    artifact_schema = _artifact_schema(artifact)
    design, provider = _prepare_features(
        bundle,
        raw_design,
        artifact_schema=artifact_schema,
        raw_feature_provider=raw_feature_provider,
    )
    resolved_series_ids = tuple(series_ids or bundle.series_ids)
    times = pd.DatetimeIndex(forecast_times)
    if isinstance(artifact, CanonicalStrategyArtifact):
        transformed = _predict_point(
            strategy,
            artifact,
            design,
            series_ids=resolved_series_ids,
            forecast_times=times,
            feature_provider=provider,
        )
    elif isinstance(artifact, CanonicalMarginalQuantileArtifact):
        transformed = _predict_quantiles(
            strategy,
            artifact,
            design,
            series_ids=resolved_series_ids,
            forecast_times=times,
            feature_provider=provider,
        )
    else:
        raise TypeError(
            "strategy bundle model must be CanonicalStrategyArtifact or "
            "CanonicalMarginalQuantileArtifact"
        )
    if bundle.target_transform is None:
        return transformed
    if isinstance(transformed, PointForecastTensor):
        return bundle.target_transform.restore_point(transformed)
    return bundle.target_transform.restore_distribution(transformed)


def _artifact_schema(
    artifact: CanonicalStrategyArtifact | CanonicalMarginalQuantileArtifact,
) -> tuple[str, ...]:
    if isinstance(artifact, CanonicalStrategyArtifact):
        return tuple(artifact.feature_schema)
    if isinstance(artifact, CanonicalMarginalQuantileArtifact):
        point = artifact.artifacts_by_level[artifact.point_level]
        return tuple(point.feature_schema)
    raise TypeError("unsupported canonical bundle artifact")


def _prepare_features(
    bundle: ForecastModelBundle,
    raw_design: np.ndarray,
    *,
    artifact_schema: tuple[str, ...],
    raw_feature_provider: FeatureProvider | None,
) -> tuple[np.ndarray, FeatureProvider | None]:
    columns = bundle.input_schema.get("columns")
    if not isinstance(columns, list) or any(not isinstance(value, str) for value in columns):
        raise ValueError("strategy bundle input_schema.columns must be a string list")
    design = np.asarray(raw_design, dtype=float)
    if design.ndim != 2 or design.shape[1] != len(columns):
        raise ValueError(
            "raw deployment design must be two-dimensional and match "
            "bundle.input_schema.columns"
        )
    if bundle.feature_scaler is not None:
        design = np.asarray(bundle.feature_scaler.transform(design), dtype=float)
    indices = selected_indices_for_artifact(tuple(columns), artifact_schema)
    if indices is not None:
        design = design[:, indices]

    if raw_feature_provider is None:
        return design, None

    def provider(call_index, coordinates, dependencies, predicted):
        values = np.asarray(
            raw_feature_provider(call_index, coordinates, dependencies, predicted),
            dtype=float,
        )
        if bundle.feature_scaler is not None:
            values = np.asarray(bundle.feature_scaler.transform(values), dtype=float)
        if indices is not None:
            values = values[:, indices]
        return values

    return design, provider


def _predict_point(
    strategy: ForecastStrategySpec,
    artifact: CanonicalStrategyArtifact,
    design: np.ndarray,
    *,
    series_ids: tuple[Any, ...],
    forecast_times: pd.DatetimeIndex,
    feature_provider: FeatureProvider | None,
) -> PointForecastTensor:
    resolved = strategy.resolve(artifact.H)
    if artifact.target_plan.strategy_name is not strategy.name:
        raise ValueError("bundle strategy does not match strategy artifact")
    executor_type = get_standard_executor(strategy)
    executor = executor_type(strategy, artifact.target_plan, artifact.predictors)
    if resolved.consumes_previous and feature_provider is None:
        raise ValueError("recursive strategy deployment requires raw_feature_provider")
    return executor.predict(
        design,
        series_ids=series_ids,
        forecast_times=forecast_times,
        feature_provider=feature_provider,
    )


def _predict_quantiles(
    strategy: ForecastStrategySpec,
    artifact: CanonicalMarginalQuantileArtifact,
    design: np.ndarray,
    *,
    series_ids: tuple[Any, ...],
    forecast_times: pd.DatetimeIndex,
    feature_provider: FeatureProvider | None,
) -> MarginalForecastDistribution:
    point_artifact = artifact.artifacts_by_level[artifact.point_level]
    has_dependencies = any(point_artifact.target_plan.dependencies)
    if has_dependencies and feature_provider is None:
        raise ValueError("recursive quantile deployment requires raw_feature_provider")
    point = _predict_point(
        strategy,
        point_artifact,
        design,
        series_ids=series_ids,
        forecast_times=forecast_times,
        feature_provider=feature_provider,
    )
    median_predictions = {
        TargetCoordinate(target, step): point.values[:, step - 1, target_index]
        for step in range(1, point.n_steps + 1)
        for target_index, target in enumerate(point.targets)
    }

    def median_path_provider(call_index, coordinates, dependencies, _predicted):
        assert feature_provider is not None
        return feature_provider(
            call_index,
            coordinates,
            dependencies,
            median_predictions,
        )

    tensors = {artifact.point_level: point}
    for level in artifact.levels:
        if level == artifact.point_level:
            continue
        tensors[level] = _predict_point(
            strategy,
            artifact.artifacts_by_level[level],
            design,
            series_ids=series_ids,
            forecast_times=forecast_times,
            feature_provider=(
                median_path_provider if has_dependencies else feature_provider
            ),
        )
    ordered = [tensors[level] for level in artifact.levels]
    quantiles = repair_marginal_quantile_crossing(
        MarginalQuantileForecastTensor(
            values=np.stack([tensor.values for tensor in ordered], axis=-1),
            levels=artifact.levels,
            point_level=artifact.point_level,
            series_ids=ordered[0].series_ids,
            forecast_times=ordered[0].forecast_times,
            targets=ordered[0].targets,
        )
    )
    return MarginalForecastDistribution(
        point=quantiles.point(),
        quantiles=quantiles,
        dependence_model=None,
        metadata={"recursive_propagation": "median_path"},
    )


__all__ = ["FeatureProvider", "predict_strategy_bundle"]
