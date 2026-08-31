"""Ensemble predictor: combine restored member predictions (v4 §8.1).

Deployment path: load the EnsembleArtifact's method state, combine member
predictions on shared (series, time, target[, quantile]) coordinates. No OOF
cache, backtest CSV, or base-model YAML access.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from model_ensemble.artifacts import (
    EnsembleArtifact,
    PerTargetMetaArtifact,
    PerTargetWeightsArtifact,
)
from model_ensemble.methods.stacking import combine_stacking
from model_ensemble.methods.weighted import weight_matrix


def combine_members(
    ens_artifact: EnsembleArtifact,
    member_values: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Combine restored member predictions into the ensemble tensor."""
    if tuple(member_values) != tuple(ens_artifact.member_order):
        raise ValueError(
            "member predictions must be provided exactly in member_order"
        )
    method_artifact = ens_artifact.method_artifact
    if isinstance(method_artifact, PerTargetWeightsArtifact):
        stacked = np.stack(
            [np.asarray(member_values[name], dtype=float) for name in member_values],
            axis=0,
        )
        weights = weight_matrix(
            method_artifact, stacked.shape[3], len(member_values)
        )
        if stacked.ndim == 4:
            return np.einsum("km,mnhk->nhk", weights, stacked)
        if stacked.ndim == 5:
            return np.einsum("km,mnhkq->nhkq", weights, stacked)
        raise ValueError("member predictions must have shape (N,H,K[,Q])")
    if isinstance(method_artifact, PerTargetMetaArtifact):
        return combine_stacking(method_artifact, member_values)
    # EqualWeightsArtifact
    stacked = np.stack(
        [np.asarray(member_values[name], dtype=float) for name in member_values],
        axis=0,
    )
    return stacked.mean(axis=0)


__all__ = ["combine_members"]
