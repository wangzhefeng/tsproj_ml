"""OOFPredictionArtifact / MethodArtifact / EnsembleArtifact containers (v4 §8.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class OOFPredictionArtifact:
    """Out-of-fold member predictions on a shared coordinate system.

    ``values_by_member[name]`` is an ``(n_oof_samples, H, K)`` array (point) or
    ``(n_oof_samples, H, K, Q)`` array (marginal quantile), restored to the
    original target space. ``folds`` carries the per-fold metadata summary.
    """

    values_by_member: dict[str, np.ndarray]
    member_order: tuple[str, ...]
    targets: tuple[str, ...]
    horizon: int
    quantile_levels: tuple[float, ...] | None
    oof_fingerprint: str
    folds: tuple[dict[str, Any], ...] = ()
    series_ids: tuple[Any, ...] = ()
    execution_evidence: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if tuple(self.values_by_member) != self.member_order:
            raise ValueError(
                "OOF member order must match values_by_member insertion order"
            )
        shapes = {
            (name, value.shape) for name, value in self.values_by_member.items()
        }
        first_shape = next(iter(shapes))[1]
        for _, shape in shapes:
            if shape != first_shape:
                raise ValueError("OOF member prediction shapes must match")
        expected_depth = 4 if self.quantile_levels else 3
        if len(first_shape) != expected_depth:
            raise ValueError(
                f"OOF arrays must have {expected_depth} dimensions "
                f"(samples, H, K{', Q' if self.quantile_levels else ''})"
            )
        if first_shape[1] != self.horizon:
            raise ValueError("OOF horizon must match the array shape")
        if first_shape[2] != len(self.targets):
            raise ValueError("OOF target count must match the array shape")
        if self.quantile_levels and first_shape[3] != len(self.quantile_levels):
            raise ValueError("OOF quantile level count must match the array shape")

    @property
    def n_samples(self) -> int:
        return next(iter(self.values_by_member.values())).shape[0]


@dataclass(frozen=True, slots=True)
class EqualWeightsArtifact:
    method_name: str = "averaging"


@dataclass(frozen=True, slots=True)
class PerTargetWeightsArtifact:
    """Inverse-error or blend weights, one tuple per target."""

    method_name: str
    weights_by_target: dict[str, tuple[float, ...]]
    metric: str | None = None
    coefficients: bool = False  # True when learned as blend coefficients
    quantile_levels: tuple[float, ...] | None = None
    n_samples_by_target: dict[str, int] = field(default_factory=dict)
    optimizer_success_by_target: dict[str, bool] = field(default_factory=dict)
    optimizer_status_by_target: dict[str, int] = field(default_factory=dict)
    optimizer_message_by_target: dict[str, str] = field(default_factory=dict)
    fallback_reason_by_target: dict[str, str | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PerTargetMetaArtifact:
    """Ridge stacking models, one per target, with saved standardization.

    ``y_mean_by_target`` stores the target mean used to center the Ridge
    regression target (equivalent to an intercept, kept explicit and frozen).
    """

    method_name: str
    models_by_target: dict[str, Any]
    mean_by_target: dict[str, np.ndarray]
    scale_by_target: dict[str, np.ndarray]
    member_order: tuple[str, ...]
    alpha: float = 1.0
    fit_intercept: bool = False
    y_mean_by_target: dict[str, float] = field(default_factory=dict)


MethodArtifact = EqualWeightsArtifact | PerTargetWeightsArtifact | PerTargetMetaArtifact


def method_artifact_audit_payload(artifact: MethodArtifact) -> dict[str, Any]:
    """Return JSON-safe fusion state without serializing estimator objects."""
    if isinstance(artifact, EqualWeightsArtifact):
        return {"method_name": artifact.method_name}
    if isinstance(artifact, PerTargetWeightsArtifact):
        return {
            "method_name": artifact.method_name,
            "weights_by_target": {
                target: list(weights)
                for target, weights in artifact.weights_by_target.items()
            },
            "metric": artifact.metric,
            "coefficients": artifact.coefficients,
            "quantile_levels": (
                list(artifact.quantile_levels)
                if artifact.quantile_levels is not None
                else None
            ),
            "n_samples_by_target": dict(artifact.n_samples_by_target),
            "optimizer_success_by_target": dict(
                artifact.optimizer_success_by_target
            ),
            "optimizer_status_by_target": dict(artifact.optimizer_status_by_target),
            "optimizer_message_by_target": dict(
                artifact.optimizer_message_by_target
            ),
            "fallback_reason_by_target": dict(artifact.fallback_reason_by_target),
        }
    return {
        "method_name": artifact.method_name,
        "member_order": list(artifact.member_order),
        "alpha": artifact.alpha,
        "fit_intercept": artifact.fit_intercept,
        "targets": sorted(artifact.models_by_target),
    }


@dataclass(frozen=True, slots=True)
class EnsembleArtifact:
    """Deployment artifact: method params + fold summary + dimensions (v4 §8.1).

    Member base-model bundles are attached by the runtime (E5) when the
    ForecastModelBundle is assembled; this artifact itself only carries the
    fusion state and lineage summaries.
    """

    method_artifact: MethodArtifact
    member_order: tuple[str, ...]
    targets: tuple[str, ...]
    horizon: int
    quantile_levels: tuple[float, ...] | None
    oof_fingerprint: str
    fold_summary: tuple[dict[str, Any], ...] = ()
    oof_reference: dict[str, Any] = field(default_factory=dict)
    source_lineage: tuple[dict[str, Any], ...] = ()
    config_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if len(set(self.member_order)) != len(self.member_order):
            raise ValueError("member_order must be unique")
        if self.horizon <= 0 or not self.targets:
            raise ValueError("EnsembleArtifact requires positive horizon and targets")
