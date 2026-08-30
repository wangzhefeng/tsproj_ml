"""stacking: per-target Ridge meta models on OOF predictions (v4 §3, fixed).

Semantics locked by v4:
- flatten (n_samples, H) into one training column per member — one Ridge per
  target, summarised across horizons (no per-horizon models);
- no free intercept: X is standardized AND the target is mean-centered
  (centering is stored in the artifact, so this stays equivalent to an
  intercept without introducing an un-regularized offset column);
- per-target standardization of meta inputs (saved in the artifact);
- Ridge(alpha=1.0), not configurable — tuning belongs to E8 experiments.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import Ridge

from model_ensemble.artifacts import PerTargetMetaArtifact
from model_ensemble.methods.weighted import target_key

METHOD_NAME = "stacking"
ALPHA = 1.0


def fit_stacking(
    oof_values_by_member: Mapping[str, np.ndarray],
    actual: np.ndarray,
) -> PerTargetMetaArtifact:
    names = tuple(oof_values_by_member)
    if len(names) < 2:
        raise ValueError("stacking requires at least two members")
    stacked = np.stack(
        [np.asarray(oof_values_by_member[name], dtype=float) for name in names],
        axis=0,
    )  # (members, n, H, K)
    k = stacked.shape[3]

    models_by_target: dict[str, Any] = {}
    mean_by_target: dict[str, np.ndarray] = {}
    scale_by_target: dict[str, np.ndarray] = {}
    y_mean_by_target: dict[str, float] = {}
    for index in range(k):
        key = target_key(index)
        design = stacked[:, :, :, index].reshape(len(names), -1).T  # (n*H, members)
        target = actual[:, :, index].reshape(-1)
        mean = design.mean(axis=0)
        scale = design.std(axis=0)
        scale = np.where(scale > 0.0, scale, 1.0)
        y_mean = float(target.mean())
        model = Ridge(alpha=ALPHA, fit_intercept=False)
        model.fit((design - mean) / scale, target - y_mean)
        models_by_target[key] = model
        mean_by_target[key] = mean
        scale_by_target[key] = scale
        y_mean_by_target[key] = y_mean
    return PerTargetMetaArtifact(
        method_name=METHOD_NAME,
        models_by_target=models_by_target,
        mean_by_target=mean_by_target,
        scale_by_target=scale_by_target,
        member_order=names,
        alpha=ALPHA,
        fit_intercept=False,
        y_mean_by_target=y_mean_by_target,
    )


def combine_stacking(
    method_artifact: PerTargetMetaArtifact,
    member_values: Mapping[str, np.ndarray],
) -> np.ndarray:
    if method_artifact.member_order != tuple(member_values):
        raise ValueError(
            "stacking member order at predict time must match training order"
        )
    stacked = np.stack(
        [np.asarray(member_values[name], dtype=float) for name in member_values],
        axis=0,
    )
    n, horizon, k = stacked.shape[1], stacked.shape[2], stacked.shape[3]
    output = np.empty_like(stacked[0])
    for index in range(k):
        key = target_key(index)
        design = stacked[:, :, :, index].reshape(len(member_values), -1).T
        normalized = (design - method_artifact.mean_by_target[key]) / (
            method_artifact.scale_by_target[key]
        )
        output[:, :, index] = normalized.dot(
            method_artifact.models_by_target[key].coef_
        ).reshape(n, horizon) + method_artifact.y_mean_by_target[key]
    return output


__all__ = ["ALPHA", "METHOD_NAME", "combine_stacking", "fit_stacking"]
