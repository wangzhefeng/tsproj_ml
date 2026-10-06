"""Canonical target and feature transformations consumed by the canonical runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from feature_engineering.transforms.stack import TargetTransformPipeline
from feature_engineering.transforms.per_series import PerSeriesTargetTransformPipeline
from feature_engineering.transform_specs import normalize_target_transformations
# CanonicalFeatureScaler 住在同包 scaling.py（2026-09-06 R3 归位，re-export 桥删除）
from forecasting_core.tensors.quantile import MarginalQuantileForecastTensor
from forecasting_core.tensors.point import PointForecastTensor
from forecasting_core.probability.distribution import MarginalForecastDistribution


@dataclass(frozen=True)
class _CanonicalPipelineFactory:
    transformations: dict[str, dict[str, Any]]

    def __call__(self) -> TargetTransformPipeline:
        return TargetTransformPipeline(self.transformations)


class CanonicalTargetTransform:
    """Per-series/per-target target-space state with strict reverse restoration."""

    def __init__(self, transformations: Mapping[str, Any], *, freq: str) -> None:
        self.transformations = normalize_target_transformations(transformations)
        self.freq = str(freq)
        self.fit_window_metadata: dict[str, Any] = {}
        self.is_identity = all(
            self.transformations[name]["method"] == "none"
            for name in ("calendar_normalization", "decomposition", "scaling")
        )
        self._identity_series_ids: tuple[Any, ...] = ()
        self._identity_targets: tuple[str, ...] = ()
        self._pipeline = PerSeriesTargetTransformPipeline(
            _CanonicalPipelineFactory(self.transformations)
        )

    @classmethod
    def from_config(cls, config: Any) -> "CanonicalTargetTransform":
        target = config.features.transformations.get("target", {})
        return cls(target, freq=config.problem.freq)

    @property
    def fitted_keys(self) -> tuple[tuple[Any, str], ...]:
        if self.is_identity:
            return tuple(
                (series_id, target)
                for series_id in self._identity_series_ids
                for target in self._identity_targets
            )
        return self._pipeline.fitted_keys

    @property
    def training_steps(self) -> dict[tuple[Any, str], tuple[str, ...]]:
        if self.is_identity:
            return {key: () for key in self.fitted_keys}
        return self._pipeline.training_steps

    def fit_transform(self, history: PointForecastTensor, *, scaling_times: pd.DatetimeIndex | None = None) -> PointForecastTensor:
        if self.is_identity:
            if not isinstance(history, PointForecastTensor):
                raise TypeError("history must be a PointForecastTensor")
            self.fit_identity(history.series_ids, history.targets)
            return history
        return self._pipeline.fit_transform(history, scaling_times=scaling_times)

    def fit_identity(
        self,
        series_ids: Sequence[Any],
        targets: Sequence[str],
    ) -> None:
        if not self.is_identity:
            raise RuntimeError("fit_identity requires an identity target transform")
        self._identity_series_ids = tuple(series_ids)
        self._identity_targets = tuple(targets)

    def transform_point(self, tensor: PointForecastTensor) -> PointForecastTensor:
        if self.is_identity:
            if not isinstance(tensor, PointForecastTensor):
                raise TypeError("tensor must be a PointForecastTensor")
            self._validate_identity(tensor.series_ids, tensor.targets)
            return tensor
        return self._pipeline.transform_point(tensor)

    def transform_training(
        self,
        values: np.ndarray,
        origins: Sequence[pd.Timestamp],
        *,
        series_ids: Sequence[Any] | None = None,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 3:
            raise ValueError(
                f"training targets must have shape (samples,H,K); got {array.shape}"
            )
        if len(origins) != array.shape[0]:
            raise ValueError("training origins must match the target sample axis")
        normalized_series_ids = (
            ("__local__",) * array.shape[0]
            if series_ids is None
            else tuple(series_ids)
        )
        if len(normalized_series_ids) != array.shape[0]:
            raise ValueError("training series_ids must match the target sample axis")
        if self.is_identity and not self._identity_series_ids:
            raise RuntimeError("fit_transform must run before transform")
        targets = (
            self._identity_targets
            if self.is_identity
            else self._pipeline.targets
        )
        if array.shape[2] != len(targets):
            raise ValueError("training target axis must match fitted target order")
        if self.is_identity:
            return array
        transformed = np.empty_like(array, dtype=float)
        offset = pd.tseries.frequencies.to_offset(self.freq)
        for sample_index, origin in enumerate(origins):
            series_id = normalized_series_ids[sample_index]
            times = pd.DatetimeIndex(
                [pd.Timestamp(origin) + (step + 1) * offset for step in range(array.shape[1])]
            )
            for target_index, target in enumerate(targets):
                transformed[sample_index, :, target_index] = self._pipeline.transform_values(
                    series_id,
                    target,
                    array[sample_index, :, target_index],
                    times,
                )
        return transformed

    def restore_values(
        self,
        series_id: Any,
        target: str,
        values: Any,
        times: Any,
    ) -> np.ndarray:
        if self.is_identity:
            if not self._identity_series_ids:
                raise RuntimeError(
                    "fit_transform must run before transform or restore"
                )
            if (series_id, target) not in self.fitted_keys:
                raise ValueError(
                    f"unknown fitted target transform key: {(series_id, target)!r}"
                )
            array = np.asarray(values, dtype=float)
            if array.ndim == 0:
                array = array.reshape(1)
            if array.ndim != 1:
                raise ValueError(
                    "target restore expects one-dimensional values; "
                    f"got shape={array.shape}"
                )
            time_index = pd.DatetimeIndex(pd.to_datetime(times))
            if len(time_index) != len(array):
                raise ValueError(
                    "target restore length mismatch: "
                    f"values={len(array)}, times={len(time_index)}"
                )
            if time_index.hasnans:
                raise ValueError("target restore times must not contain NaT")
            return array
        return self._pipeline.restore_values(series_id, target, values, times)

    def restore_point(self, tensor: PointForecastTensor) -> PointForecastTensor:
        if self.is_identity:
            if not isinstance(tensor, PointForecastTensor):
                raise TypeError("tensor must be a PointForecastTensor")
            self._validate_identity(tensor.series_ids, tensor.targets)
            return tensor
        return self._pipeline.restore_point(tensor)

    def _validate_identity(
        self,
        series_ids: tuple[Any, ...],
        targets: tuple[str, ...],
    ) -> None:
        if not self._identity_series_ids:
            raise RuntimeError("fit_transform must run before restore")
        if series_ids != self._identity_series_ids or targets != self._identity_targets:
            raise ValueError(
                "restore tensor series/target identity must match the fitted transform state"
            )

    def restore_quantiles(
        self,
        tensor: MarginalQuantileForecastTensor,
    ) -> MarginalQuantileForecastTensor:
        if self.is_identity:
            if not isinstance(tensor, MarginalQuantileForecastTensor):
                raise TypeError("tensor must be a MarginalQuantileForecastTensor")
            self._validate_identity(tensor.series_ids, tensor.targets)
            return tensor
        return self._pipeline.restore_quantiles(tensor)

    def restore_distribution(
        self,
        distribution: MarginalForecastDistribution,
    ) -> MarginalForecastDistribution:
        if self.is_identity:
            self.restore_quantiles(distribution.quantiles)
            return distribution
        quantiles = self.restore_quantiles(distribution.quantiles)
        return MarginalForecastDistribution(
            point=quantiles.point(),
            quantiles=quantiles,
            dependence_model=None,
            metadata=dict(distribution.metadata),
        )
