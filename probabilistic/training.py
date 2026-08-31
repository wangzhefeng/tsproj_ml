# -*- coding: utf-8 -*-
"""按 quantile grid 编排训练并构造强类型模型 bundle。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from model_training.estimators import EstimatorCapabilities
from model_training.trainer import CanonicalTrainer
from forecasting_core.specs import ForecastConfigSpec
from forecasting_core.probabilistic_spec import ProbabilisticSpec


@dataclass(frozen=True, slots=True)
class CanonicalMarginalQuantileArtifact:
    levels: tuple[float, ...]
    point_level: float
    artifacts_by_level: MappingProxyType
    dependence_model: None = None
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("canonical quantile artifact schema_version must be 2")
        if self.dependence_model is not None:
            raise ValueError("canonical marginal artifact dependence_model must be None")
        if tuple(self.artifacts_by_level) != self.levels:
            raise ValueError("artifacts_by_level keys must exactly match levels")

    def __reduce__(self):
        return (
            type(self),
            (
                self.levels,
                self.point_level,
                dict(self.artifacts_by_level),
                self.dependence_model,
                self.schema_version,
            ),
        )


class CanonicalMarginalQuantileTrainer:
    """Fit one canonical strategy artifact per marginal quantile level."""

    def __init__(
        self,
        config: ForecastConfigSpec,
        *,
        estimator_factory_for_level: Callable[[float], Callable[[], object]],
        capabilities: EstimatorCapabilities,
        feature_schema: Sequence[str],
    ) -> None:
        if not isinstance(config, ForecastConfigSpec):
            raise TypeError("config must be a ForecastConfigSpec")
        if str(config.probabilistic.get("mode", "point")) != "quantile":
            raise ValueError(
                "CanonicalMarginalQuantileTrainer requires probabilistic.mode=quantile"
            )
        if not callable(estimator_factory_for_level):
            raise TypeError("estimator_factory_for_level must be callable")
        raw_levels = config.probabilistic.get("quantiles")
        if not isinstance(raw_levels, (list, tuple)):
            raise TypeError("probabilistic.quantiles must be a sequence")
        levels = tuple(float(level) for level in raw_levels)
        if (
            not levels
            or any(
                not np.isfinite(level) or not 0.0 < level < 1.0
                for level in levels
            )
            or tuple(sorted(set(levels))) != levels
        ):
            raise ValueError(
                "probabilistic.quantiles must be unique, increasing, and inside (0, 1)"
            )
        point_level = float(config.probabilistic.get("point_quantile", 0.5))
        if point_level not in levels:
            raise ValueError(
                "point_quantile must be present in probabilistic.quantiles"
            )
        self.config = config
        self.estimator_factory_for_level = estimator_factory_for_level
        self.capabilities = capabilities
        self.feature_schema = tuple(feature_schema)
        self.levels = levels
        self.point_level = point_level

    def train(
        self,
        X_by_call: Sequence[np.ndarray],
        Y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        n_series: int = 1,
    ) -> CanonicalMarginalQuantileArtifact:
        artifacts = {}
        for level in self.levels:
            estimator_factory = self.estimator_factory_for_level(level)
            if not callable(estimator_factory):
                raise TypeError(
                    "estimator_factory_for_level must return an estimator factory"
                )
            artifacts[level] = CanonicalTrainer(
                self.config,
                estimator_factory=estimator_factory,
                capabilities=self.capabilities,
                feature_schema=self.feature_schema,
            ).train(
                X_by_call,
                Y,
                sample_weight=sample_weight,
                n_series=n_series,
            )
        return CanonicalMarginalQuantileArtifact(
            levels=self.levels,
            point_level=self.point_level,
            artifacts_by_level=MappingProxyType(artifacts),
        )
