"""Canonical fold/final-fit and prediction services."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from forecasting_core.artifacts import MarginalForecastDistribution
from forecasting_core.specs import ForecastConfigSpec, TargetAdapter
from forecasting_core.tensors import PointForecastTensor
from model_forecasting.design import _RegistryDesignBuilder
from model_forecasting.forecaster import (
    CanonicalForecaster,
    CanonicalMarginalQuantileForecaster,
)
from model_forecasting.transforms import CanonicalFeatureScaler, CanonicalTargetTransform
from model_training.estimators import make_model_factory, resolve_model_capabilities
from model_training.trainer import CanonicalTrainer
from probabilistic.training import CanonicalMarginalQuantileTrainer


def _fit_runtime_transforms(
    config: ForecastConfigSpec,
    builder: _RegistryDesignBuilder,
    X_by_call: tuple[np.ndarray, ...],
    Y: np.ndarray,
    origins: tuple[pd.Timestamp, ...],
    sample_series_ids: tuple[Any, ...],
    history_cutoff: pd.Timestamp,
) -> tuple[
    CanonicalFeatureScaler,
    CanonicalTargetTransform,
    tuple[np.ndarray, ...],
    np.ndarray,
]:
    feature_scaler = CanonicalFeatureScaler.from_config(
        config,
        feature_names=builder.feature_schema,
        categorical_names=builder.categorical_schema,
        category_orders={
            column: tuple(
                dict.fromkeys(
                    (
                        identity[index]
                        if isinstance(identity, tuple)
                        else identity
                    )
                    for identity in builder.series_ids
                )
            )
            for index, column in enumerate(config.problem.series_id_cols)
        },
    )
    transformed_X = feature_scaler.fit_transform_calls(X_by_call)
    target_transform = CanonicalTargetTransform.from_config(config)
    target_transform.fit_transform(builder.target_history(history_cutoff))
    transformed_Y = target_transform.transform_training(
        Y,
        origins,
        series_ids=sample_series_ids,
    )
    return feature_scaler, target_transform, transformed_X, transformed_Y


def _forecast_designs_with_scaler(
    builder: _RegistryDesignBuilder,
    origin: pd.Timestamp,
    feature_scaler: CanonicalFeatureScaler,
    target_transform: CanonicalTargetTransform,
):
    raw_designs, raw_provider = builder.forecast_designs(
        origin,
        target_transform=target_transform,
    )

    def provider(call_index, coordinates, dependencies, predicted):
        return feature_scaler.transform(
            raw_provider(call_index, coordinates, dependencies, predicted)
        )

    return (feature_scaler.transform(raw_designs[0]),), provider


def _restore_prediction(
    prediction: PointForecastTensor | MarginalForecastDistribution,
    target_transform: CanonicalTargetTransform,
) -> PointForecastTensor | MarginalForecastDistribution:
    if isinstance(prediction, PointForecastTensor):
        return target_transform.restore_point(prediction)
    if isinstance(prediction, MarginalForecastDistribution):
        return target_transform.restore_distribution(prediction)
    raise TypeError(f"unsupported canonical prediction type: {type(prediction).__name__}")


def _fit_point(
    config: ForecastConfigSpec,
    feature_schema: tuple[str, ...],
    X_by_call: tuple[np.ndarray, ...],
    Y: np.ndarray,
    *,
    n_series: int,
):
    capabilities = resolve_model_capabilities(
        config.estimator.model_type,
        config.estimator.params,
        feature_names=feature_schema,
        probe_native=config.estimator.target_adapter is TargetAdapter.NATIVE,
    )
    trainer = CanonicalTrainer(
        config,
        estimator_factory=make_model_factory(
            config.estimator.model_type,
            config.estimator.params,
            feature_names=feature_schema,
        ),
        capabilities=capabilities,
        feature_schema=feature_schema,
    )
    return trainer, trainer.train(X_by_call, Y, n_series=n_series)


def _fit_quantile(
    config: ForecastConfigSpec,
    feature_schema: tuple[str, ...],
    X_by_call: tuple[np.ndarray, ...],
    Y: np.ndarray,
    *,
    n_series: int,
):
    capabilities = resolve_model_capabilities(
        config.estimator.model_type,
        config.estimator.params,
        feature_names=feature_schema,
        probe_native=config.estimator.target_adapter is TargetAdapter.NATIVE,
    )
    if not capabilities.scalar_quantile:
        raise ValueError(
            f"model_type {config.estimator.model_type!r} does not support scalar quantiles"
        )
    trainer = CanonicalMarginalQuantileTrainer(
        config,
        estimator_factory_for_level=lambda level: make_model_factory(
            config.estimator.model_type,
            config.estimator.params,
            feature_names=feature_schema,
            quantile=level,
        ),
        capabilities=capabilities,
        feature_schema=feature_schema,
    )
    return trainer, trainer.train(X_by_call, Y, n_series=n_series), capabilities


def _predict(
    config: ForecastConfigSpec,
    artifact,
    base_design: np.ndarray,
    provider,
    forecast_times: pd.DatetimeIndex,
    series_ids: tuple[Any, ...],
):
    kwargs = {
        "series_ids": series_ids,
        "forecast_times": forecast_times,
        "feature_provider": provider,
    }
    if str(config.probabilistic.get("mode", "point")) == "point":
        return CanonicalForecaster(config, artifact).predict(base_design, **kwargs)
    return CanonicalMarginalQuantileForecaster(config, artifact).predict(
        base_design,
        **kwargs,
    )
