"""weighted: per-target inverse-error weights learned on OOF (v4 §3).

Default metric is RMSE (stable); `mae` optional; `mape` must be requested
explicitly and is guarded by a nonzero floor (small denominators explode MAPE
on low-load periods). All quantile levels of one target share one weight set.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from model_ensemble.artifacts import PerTargetWeightsArtifact

METHOD_NAME = "weighted"
SUPPORTED_METRICS = ("rmse", "mae", "mape")
_MAPE_FLOOR = 1e-8


def target_key(index: int) -> str:
    return f"target_{index}"


def _metric_values(
    actual: np.ndarray, prediction: np.ndarray, metric: str,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Per-target error aggregation: (K,) array of error magnitudes."""
    if prediction.ndim == 4:
        actual = actual[..., None]
    axes = (0, 1, 3) if prediction.ndim == 4 else (0, 1)
    err = prediction - actual
    if sample_weights is not None:
        loss = np.square(err) if metric == "rmse" else np.abs(err)
        if metric == "mape":
            loss = loss / np.maximum(np.abs(actual), _MAPE_FLOOR)
        if prediction.ndim == 4:
            loss = loss.mean(axis=-1)
        mean_loss = np.average(loss.mean(axis=1), axis=0, weights=sample_weights)
        return np.sqrt(mean_loss) if metric == "rmse" else mean_loss
    if metric == "rmse":
        return np.sqrt(np.mean(np.square(err), axis=axes))
    if metric == "mae":
        return np.mean(np.abs(err), axis=axes)
    if metric == "mape":
        denom = np.maximum(np.abs(actual), _MAPE_FLOOR)
        return np.mean(np.abs(err) / denom, axis=axes)
    raise ValueError(
        f"weighted metric must be one of {SUPPORTED_METRICS}; got {metric!r}"
    )


def fit_weighted(
    oof_values_by_member: Mapping[str, np.ndarray],
    actual: np.ndarray,
    *,
    metric: str = "rmse",
    fallback_weights: Mapping[str, float] | None = None,
    sample_weights: np.ndarray | None = None,
) -> PerTargetWeightsArtifact:
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"weighted metric must be one of {SUPPORTED_METRICS}; got {metric!r}"
        )
    names = tuple(oof_values_by_member)
    if len(names) < 2:
        raise ValueError("weighted requires at least two members")
    actual = np.asarray(actual, dtype=float)
    if actual.ndim != 3 or not all(actual.shape) or not np.isfinite(actual).all():
        raise ValueError("weighted actual must be finite with shape (samples,H,K)")
    for prediction in oof_values_by_member.values():
        if prediction.ndim not in (3, 4) or prediction.shape[:3] != actual.shape or not np.isfinite(prediction).all():
            raise ValueError("weighted predictions must be finite and match actual on (samples,H,K)")
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=float)
        if (sample_weights.shape != (actual.shape[0],) or not np.isfinite(sample_weights).all()
                or np.any(sample_weights < 0) or sample_weights.sum() <= 0):
            raise ValueError("sample_weights must be finite, nonnegative and match samples")
    errors = np.stack(
        [
            _metric_values(actual, oof_values_by_member[name], metric, sample_weights)
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
    if stacked.ndim == 5:
        return np.einsum("km,mnhkq->nhkq", weights_2d, stacked)
    raise ValueError("member values must have shape (N,H,K[,Q])")


__all__ = ["METHOD_NAME", "SUPPORTED_METRICS", "combine_weighted", "fit_weighted"]
