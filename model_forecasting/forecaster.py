# -*- coding: utf-8 -*-
"""Canonical forecaster（2026-08-29 架构收敛自 models/ModelForecasting.py 迁入，实现逐字保真）。"""

from typing import Any

import numpy as np
import pandas as pd

from forecasting_core.specs import ForecastConfigSpec
from model_training.strategies import (
    CanonicalStrategyArtifact,
    TargetCoordinate,
    get_standard_executor,
)
from forecasting_core.tensors import (
    MarginalQuantileForecastTensor,
    PointForecastTensor,
)
from probabilistic.training import CanonicalMarginalQuantileArtifact
from forecasting_core.artifacts import MarginalForecastDistribution


class CanonicalForecaster:
    """Execute a fitted canonical strategy artifact and preserve ``(N,H,K)``."""

    def __init__(
        self,
        config: ForecastConfigSpec,
        artifact: CanonicalStrategyArtifact,
    ) -> None:
        if not isinstance(config, ForecastConfigSpec):
            raise TypeError("config must be a ForecastConfigSpec")
        if not isinstance(artifact, CanonicalStrategyArtifact):
            raise TypeError("artifact must be a CanonicalStrategyArtifact")
        if artifact.target_plan.strategy_name is not config.strategy.name:
            raise ValueError("artifact strategy does not match canonical config")
        if artifact.H != config.problem.horizon:
            raise ValueError("artifact horizon does not match canonical config")
        if artifact.target_plan.targets != config.problem.targets:
            raise ValueError("artifact targets do not match canonical config")
        self.config = config
        self.artifact = artifact

    def predict(
        self,
        X: np.ndarray,
        *,
        series_ids: tuple[Any, ...],
        forecast_times: pd.DatetimeIndex,
        feature_provider=None,
    ) -> PointForecastTensor:
        design = np.asarray(X, dtype=float)
        if design.ndim != 2:
            raise ValueError("canonical forecast X must be two-dimensional")
        if design.shape[1] != len(self.artifact.feature_schema):
            raise ValueError(
                "canonical forecast feature width does not match artifact schema"
            )
        if design.shape[0] != self.artifact.N:
            raise ValueError(
                f"canonical forecast requires N={self.artifact.N} rows; "
                f"got {design.shape[0]}"
            )
        if len(series_ids) != self.artifact.N:
            raise ValueError(
                f"canonical forecast requires N={self.artifact.N} series IDs"
            )
        times = pd.DatetimeIndex(forecast_times)
        if len(times) != self.artifact.H:
            raise ValueError(
                f"canonical forecast requires H={self.artifact.H} times"
            )
        executor_type = get_standard_executor(self.config.strategy)
        executor = executor_type(
            self.config.strategy,
            self.artifact.target_plan,
            self.artifact.predictors,
        )
        return executor.predict(
            design,
            series_ids=series_ids,
            forecast_times=times,
            feature_provider=feature_provider,
        )


def repair_marginal_quantile_crossing(
    tensor: MarginalQuantileForecastTensor,
) -> MarginalQuantileForecastTensor:
    """Repair each marginal quantile grid around its fixed point level."""
    if not isinstance(tensor, MarginalQuantileForecastTensor):
        raise TypeError("tensor must be a MarginalQuantileForecastTensor")
    values = tensor.values.copy()
    point_index = tensor.levels.index(tensor.point_level)
    anchor = values[..., point_index].copy()
    if point_index:
        lower = np.sort(values[..., :point_index], axis=-1)
        lower = np.minimum(lower, anchor[..., None])
        values[..., :point_index] = lower
    if point_index + 1 < tensor.n_levels:
        upper = np.sort(values[..., point_index + 1 :], axis=-1)
        upper = np.maximum(upper, anchor[..., None])
        values[..., point_index + 1 :] = upper
    values[..., point_index] = anchor
    return MarginalQuantileForecastTensor(
        values=values,
        levels=tensor.levels,
        point_level=tensor.point_level,
        series_ids=tensor.series_ids,
        forecast_times=tensor.forecast_times,
        targets=tensor.targets,
    )


class CanonicalMarginalQuantileForecaster:
    """Forecast every marginal quantile artifact into ``(N,H,K,Q)``."""

    def __init__(
        self,
        config: ForecastConfigSpec,
        artifact: CanonicalMarginalQuantileArtifact,
    ) -> None:
        if not isinstance(config, ForecastConfigSpec):
            raise TypeError("config must be a ForecastConfigSpec")
        if not isinstance(artifact, CanonicalMarginalQuantileArtifact):
            raise TypeError(
                "artifact must be a CanonicalMarginalQuantileArtifact"
            )
        self.config = config
        self.artifact = artifact

    def predict(
        self,
        X: np.ndarray,
        *,
        series_ids: tuple,
        forecast_times: pd.DatetimeIndex,
        feature_provider=None,
    ) -> MarginalForecastDistribution:
        point_artifact = self.artifact.artifacts_by_level[self.artifact.point_level]
        has_recursive_dependencies = any(point_artifact.target_plan.dependencies)
        if has_recursive_dependencies and feature_provider is None:
            raise ValueError(
                "recursive quantile median_path requires a fixed-schema feature_provider"
            )
        point_tensor = CanonicalForecaster(
            self.config,
            point_artifact,
        ).predict(
            X,
            series_ids=series_ids,
            forecast_times=forecast_times,
            feature_provider=feature_provider,
        )
        median_predictions = {
            TargetCoordinate(target, step): point_tensor.values[
                :, step - 1, target_index
            ]
            for step in range(1, point_tensor.n_steps + 1)
            for target_index, target in enumerate(point_tensor.targets)
        }

        def median_path_provider(call_index, coordinates, dependencies, _predicted):
            assert feature_provider is not None
            return feature_provider(
                call_index,
                coordinates,
                dependencies,
                median_predictions,
            )

        tensors_by_level = {self.artifact.point_level: point_tensor}
        for level in self.artifact.levels:
            if level == self.artifact.point_level:
                continue
            tensors_by_level[level] = CanonicalForecaster(
                self.config,
                self.artifact.artifacts_by_level[level],
            ).predict(
                X,
                series_ids=series_ids,
                forecast_times=forecast_times,
                feature_provider=(
                    median_path_provider
                    if has_recursive_dependencies
                    else feature_provider
                ),
            )
        point_tensors = [tensors_by_level[level] for level in self.artifact.levels]
        values = np.stack(
            [tensor.values for tensor in point_tensors],
            axis=-1,
        )
        quantiles = repair_marginal_quantile_crossing(
            MarginalQuantileForecastTensor(
                values=values,
                levels=self.artifact.levels,
                point_level=self.artifact.point_level,
                series_ids=point_tensors[0].series_ids,
                forecast_times=point_tensors[0].forecast_times,
                targets=point_tensors[0].targets,
            )
        )
        return MarginalForecastDistribution(
            point=quantiles.point(),
            quantiles=quantiles,
            dependence_model=None,
            metadata={"recursive_propagation": "median_path"},
        )


__all__ = [
    "CanonicalForecaster",
    "CanonicalMarginalQuantileForecaster",
    "repair_marginal_quantile_crossing",
]
