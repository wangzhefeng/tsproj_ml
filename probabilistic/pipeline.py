# -*- coding: utf-8 -*-
"""概率预测后处理流水线。"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from model_forecasting.forecaster import CanonicalForecaster
from model_forecasting.specs import ForecastConfigSpec
from model_training.strategies import TargetCoordinate
from model_forecasting.tensors import MarginalQuantileForecastTensor
from probabilistic.calibration import (
    attach_cqr_interval_columns,
    calibrate_quantile_band,
)
from probabilistic.postprocessing import (
    _parse_quantile_columns,
    repair_quantile_crossing,
)
from probabilistic.training import CanonicalMarginalQuantileArtifact
from probabilistic.types import MarginalForecastDistribution


def finalize_quantile_forecast(
    frame: pd.DataFrame,
    monotone_enabled: bool,
    conformal_scores: Optional[np.ndarray] = None,
    alpha: float = 0.1,
    min_scores: int = 30,
    allow_interval_shrink: bool = False,
) -> Tuple[pd.DataFrame, Optional[float]]:
    """统一完成 q50 锚定修复，并可选追加独立 CQR interval。"""
    processed = repair_quantile_crossing(frame, enabled=monotone_enabled)
    if conformal_scores is None:
        return processed, None

    quantile_columns = _parse_quantile_columns(processed)
    if len(quantile_columns) < 2:
        raise ValueError("CQR calibration requires at least two quantile columns")
    lower_column = quantile_columns[0][1]
    upper_column = quantile_columns[-1][1]
    lower, upper, correction = calibrate_quantile_band(
        processed[lower_column].to_numpy(),
        processed[upper_column].to_numpy(),
        conformal_scores,
        alpha=alpha,
        min_scores=min_scores,
        allow_interval_shrink=allow_interval_shrink,
    )
    if lower is None or upper is None:
        return processed, None
    result = attach_cqr_interval_columns(
        processed,
        lower=lower,
        upper=upper,
        target_coverage=1.0 - float(alpha),
    )
    return result, correction


def repair_marginal_quantile_crossing(
    tensor: MarginalQuantileForecastTensor,
) -> MarginalQuantileForecastTensor:
    """Repair each ``(series,time,target)`` marginal grid around fixed q50."""
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
