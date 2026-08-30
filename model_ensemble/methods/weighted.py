"""weighted: per-target inverse-error weights learned on OOF (v4 §3).

Default metric is RMSE (stable); `mae` optional; `mape` must be requested
explicitly and is guarded by a nonzero floor (small denominators explode MAPE
on low-load periods). All quantile levels of one target share one weight set.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from model_ensemble.artifacts import PerTargetWeightsArtifact

METHOD_NAME = "weighted"
SUPPORTED_METRICS = ("rmse", "mae", "mape")
_MAPE_FLOOR = 1e-8


def target_key(index: int) -> str:
    return f"target_{index}"


def _metric_values(
    actual: np.ndarray, prediction: np.ndarray, metric: str
) -> np.ndarray:
    """Per-target error aggregation: (K,) array of error magnitudes."""
    err = prediction - actual
    if metric == "rmse":
        return np.sqrt(np.mean(np.square(err), axis=(0, 1)))
    if metric == "mae":
        return np.mean(np.abs(err), axis=(0, 1))
    if metric == "mape":
        denom = np.maximum(np.abs(actual), _MAPE_FLOOR)
        return np.mean(np.abs(err) / denom, axis=(0, 1))
    raise ValueError(
        f"weighted metric must be one of {SUPPORTED_METRICS}; got {metric!r}"
    )


def fit_weighted(
    oof_values_by_member: Mapping[str, np.ndarray],
    actual: np.ndarray,
    *,
    metric: str = "rmse",
    fallback_weights: Mapping[str, float] | None = None,
) -> PerTargetWeightsArtifact:
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"weighted metric must be one of {SUPPORTED_METRICS}; got {metric!r}"
        )
    names = tuple(oof_values_by_member)
    if len(names) < 2:
        raise ValueError("weighted requires at least two members")
    errors = np.stack(
        [
            _metric_values(actual, oof_values_by_member[name], metric)
            for name in names
        ],
        axis=0,
    )  # (members, K)
    inverse = 1.0 / np.maximum(errors, _MAPE_FLOOR)
    totals = inverse.sum(axis=0)
    fallback = (
        np.full(len(names), 1.0 / len(names))
        if fallback_weights is None
        else np.array([fallback_weights[name] for name in names])
    )
    weights = np.where(totals[None, :] > 0.0, inverse / totals[None, :], fallback[:, None])
    weights = weights / weights.sum(axis=0, keepdims=True)
    return PerTargetWeightsArtifact(
        method_name=METHOD_NAME,
        weights_by_target={
            target_key(index): tuple(float(w) for w in weights[:, index])
            for index in range(errors.shape[1])
        },
        metric=metric,
    )


def weight_matrix(
    method_artifact: PerTargetWeightsArtifact, k: int, member_count: int
) -> np.ndarray:
    matrix = np.array(
        [
            method_artifact.weights_by_target[target_key(index)]
            for index in range(k)
        ]
    )  # (K, members)
    if matrix.shape != (k, member_count):
        raise ValueError(
            "method artifact weights do not match the member/target layout"
        )
    return matrix


def combine_weighted(
    method_artifact: PerTargetWeightsArtifact,
    member_values: Mapping[str, np.ndarray],
) -> np.ndarray:
    names = tuple(member_values)
    stacked = np.stack(
        [np.asarray(member_values[name], dtype=float) for name in names],
        axis=0,
    )
    weights_2d = weight_matrix(method_artifact, stacked.shape[3], len(names))
    if stacked.ndim == 4:
        return np.einsum("km,mnhk->nhk", weights_2d, stacked)
    return np.einsum("km,mnhqk->nhqk", weights_2d, stacked)


__all__ = ["METHOD_NAME", "SUPPORTED_METRICS", "combine_weighted", "fit_weighted"]
