"""Runtime contracts between the ensemble package and base-model execution.

`BaseModelRunner` is the structural protocol the ensemble trainer/predictor
rely on; `CanonicalBaseModelRunner` (model_forecasting.runtime) is the reference
implementation. `FusionMethod` is the protocol every `model_ensemble.methods.*`
class implements.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import pandas as pd
from model_testing.validation import OriginTimeline


@runtime_checkable
class BaseModelRunner(OriginTimeline, Protocol):
    """Narrow member interface — mirrors CanonicalBaseModelRunner."""

    config: Any
    origin: pd.Timestamp
    series_ids: tuple[Any, ...]
    feature_schema: tuple[str, ...]
    workload: Any
    resource_budget: Any
    execution_plan: Any

    def fit(
        self,
        train_indices: tuple[int, ...],
        *,
        force_serial: bool = False,
    ) -> tuple[Any, Any, tuple[Any, ...], Any, Any]:
        """Fit member transforms + model; returns (scaler, transform, X, Y, artifact)."""
        ...

    def apply_ensemble_member_plan(self, parent_plan: Any) -> None:
        """Replace internal workers with a parent-budgeted serial child plan."""
        ...

    def runtime_resources_payload(self) -> dict[str, Any]: ...

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

    def target_history(self, origin: pd.Timestamp) -> Any:
        """Full target history tensor as-of origin (for forecast-plot context)."""
        ...

    def final_bundle_inputs(self) -> tuple[Any, Any, tuple[Any, ...], Any]:
        ...

    def fit_final(
        self,
        X_transformed: tuple[Any, ...],
        Y_transformed: Any,
    ) -> tuple[Any, Any, Any]:
        ...

    def build_final_bundle(
        self,
        feature_scaler: Any,
        target_transform: Any,
        trainer: Any,
        artifact: Any,
        capabilities: Any,
    ) -> Any:
        ...


class BaseModelRunnerFactory(Protocol):
    """Construct a member runner with the shared compiled-design cache root."""

    def __call__(
        self,
        config: Any,
        registry: Any,
        origin: pd.Timestamp,
        *,
        compiled_cache_root: str | Path,
        resource_budget: Any | None = None,
    ) -> BaseModelRunner:
        ...


@dataclass(frozen=True, slots=True)
class EnsembleRuntimeServices:
    """Concrete single-model capabilities injected by the application entrypoint."""

    runner_factory: BaseModelRunnerFactory
    persist_bundle: Callable[[Any, str | Path], Any]
    plan_resources: Callable[..., tuple[Any, Any, Any]]
    resolve_budget: Callable[[Any], Any] | None = None


def member_execution_evidence(runner: Any, artifact: Any, transform: Any) -> dict[str, Any]:
    """Optional public runner capability; never fabricate missing historical evidence."""
    provider = getattr(runner, "execution_evidence", None)
    if provider is None:
        return {"status": "unavailable", "reason": "runner_has_no_execution_evidence_provider"}
    if not callable(provider):
        raise TypeError("runner.execution_evidence must be callable")
    payload = provider(artifact, transform)
    if not isinstance(payload, Mapping):
        raise TypeError("runner execution evidence must be a mapping")
    return dict(payload)


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
