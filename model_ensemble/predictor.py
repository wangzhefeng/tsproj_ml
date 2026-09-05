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
from model_ensemble.methods.averaging import combine_averaging
from model_ensemble.methods.linear_blending import combine_linear_blending
from model_ensemble.methods.weighted import combine_weighted


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
        if method_artifact.method_name == "linear_blending":
            return combine_linear_blending(method_artifact, member_values)
        return combine_weighted(method_artifact, member_values)
    if isinstance(method_artifact, PerTargetMetaArtifact):
        return combine_stacking(method_artifact, member_values)
    # EqualWeightsArtifact
    return combine_averaging(method_artifact, member_values)


__all__ = ["combine_members"]
