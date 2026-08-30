"""Runtime contracts between the ensemble package and base-model execution.

`BaseModelRunner` is the structural protocol the ensemble trainer/predictor
rely on; `CanonicalBaseModelRunner` (model_forecasting.runtime) is the reference
implementation. `FusionMethod` is the protocol every `model_ensemble.methods.*`
class implements.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class BaseModelRunner(Protocol):
    """Narrow member interface — mirrors CanonicalBaseModelRunner."""

    config: Any
    series_ids: tuple[Any, ...]
    feature_schema: tuple[str, ...]

    def fit(
        self, train_indices: tuple[int, ...]
    ) -> tuple[Any, Any, tuple[Any, ...], Any, Any]:
        """Fit member transforms + model; returns (scaler, transform, X, Y, artifact)."""
        ...

    def forecast_designs(
        self,
        origin: pd.Timestamp,
        feature_scaler: Any,
        target_transform: Any,
    ) -> tuple[tuple[Any, ...], Any]:
        ...

    def predict(
        self,
        artifact: Any,
        designs: tuple[Any, ...],
        provider: Any,
        forecast_times: pd.DatetimeIndex,
        target_transform: Any,
    ) -> Any:
        """Predict and restore to the original target space."""
        ...

    def actual(
        self, origin_index: int, forecast_times: pd.DatetimeIndex
    ) -> Any:
        ...

    def forecast_times(self, origin: pd.Timestamp) -> pd.DatetimeIndex:
        ...


@runtime_checkable
class FusionMethod(Protocol):
    """Per-target fuser learned on OOF predictions (v4 §3)."""

    name: str

    def fit(
        self,
        oof_predictions: Mapping[str, Any],
        actual: Any,
        targets: tuple[str, ...],
    ) -> Any:
        """Learn per-target parameters; returns a frozen MethodArtifact."""
        ...

    def combine(
        self,
        method_artifact: Any,
        member_predictions: Mapping[str, Any],
        targets: tuple[str, ...],
    ) -> Any:
        """Combine restored member predictions into the ensemble tensor."""
        ...
