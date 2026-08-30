# -*- coding: utf-8 -*-
"""Canonical forecaster（2026-08-29 架构收敛自 models/ModelForecasting.py 迁入，实现逐字保真）。"""

from typing import Any

import numpy as np
import pandas as pd

from model_forecasting.specs import ForecastConfigSpec
from model_training.strategies import CanonicalStrategyArtifact, get_standard_executor
from model_forecasting.tensors import PointForecastTensor


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


__all__ = ["CanonicalForecaster"]
