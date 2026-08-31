"""Deployment prediction from a self-contained Ensemble bundle."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from forecasting_core.artifacts import ForecastModelBundle, MarginalForecastDistribution
from forecasting_core.tensors import MarginalQuantileForecastTensor, PointForecastTensor
from model_ensemble.artifacts import EnsembleArtifact
from model_ensemble.predictor import combine_members
from model_forecasting.deployment import FeatureProvider, predict_strategy_bundle


def predict_ensemble_bundle(
    bundle: ForecastModelBundle,
    raw_designs_by_member: Mapping[str, np.ndarray],
    *,
    forecast_times: pd.DatetimeIndex,
    series_ids: tuple[Any, ...] | None = None,
    raw_feature_providers: Mapping[str, FeatureProvider] | None = None,
) -> PointForecastTensor | MarginalForecastDistribution:
    """Predict only from the saved bundle plus explicit deployment features."""
    if not isinstance(bundle, ForecastModelBundle) or bundle.schema_version != 2:
        raise TypeError("bundle must be a schema-2 ForecastModelBundle")
    if bundle.ensemble_spec is None or bundle.strategy_spec is not None:
        raise ValueError("predict_ensemble_bundle requires an ensemble bundle")
    if not isinstance(bundle.model, dict):
        raise TypeError("ensemble bundle model must be a mapping")
    artifact = bundle.model.get("ensemble_artifact")
    member_bundles = bundle.model.get("member_bundles")
    if not isinstance(artifact, EnsembleArtifact):
        raise TypeError("ensemble bundle is missing EnsembleArtifact")
    if not isinstance(member_bundles, dict):
        raise TypeError("ensemble bundle is missing member_bundles")
    expected = tuple(artifact.member_order)
    if set(raw_designs_by_member) != set(expected):
        raise ValueError("deployment designs must match ensemble member_order")
    if set(member_bundles) != set(expected):
        raise ValueError("saved member bundles must match ensemble member_order")
    if raw_feature_providers is not None and not set(raw_feature_providers) <= set(expected):
        raise ValueError("raw_feature_providers contains an unknown member")

    resolved_series_ids = tuple(series_ids or bundle.series_ids)
    times = pd.DatetimeIndex(forecast_times)
    member_values: dict[str, np.ndarray] = {}
    for name in expected:
        prediction = predict_strategy_bundle(
            member_bundles[name],
            raw_designs_by_member[name],
            forecast_times=times,
            series_ids=resolved_series_ids,
            raw_feature_provider=(
                raw_feature_providers.get(name)
                if raw_feature_providers is not None
                else None
            ),
        )
        if artifact.quantile_levels is None:
            if not isinstance(prediction, PointForecastTensor):
                raise ValueError("point ensemble member returned a quantile distribution")
            member_values[name] = np.asarray(prediction.values, dtype=float)
        else:
            if not isinstance(prediction, MarginalForecastDistribution):
                raise ValueError("quantile ensemble member returned a point tensor")
            if tuple(prediction.quantiles.levels) != tuple(artifact.quantile_levels):
                raise ValueError("member quantile levels differ from EnsembleArtifact")
            member_values[name] = np.asarray(prediction.quantiles.values, dtype=float)

    combined = np.asarray(combine_members(artifact, member_values), dtype=float)
    if artifact.quantile_levels is None:
        return PointForecastTensor(
            values=combined,
            series_ids=resolved_series_ids,
            forecast_times=times,
            targets=artifact.targets,
        )
    point_level = float(bundle.probabilistic_spec.point_quantile)
    quantiles = MarginalQuantileForecastTensor(
        values=combined,
        levels=tuple(artifact.quantile_levels),
        point_level=point_level,
        series_ids=resolved_series_ids,
        forecast_times=times,
        targets=artifact.targets,
    )
    return MarginalForecastDistribution(
        point=quantiles.point(),
        quantiles=quantiles,
        dependence_model=None,
        metadata={"ensemble_method": artifact.method_artifact.method_name},
    )


__all__ = ["predict_ensemble_bundle"]
