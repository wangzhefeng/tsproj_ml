"""linear_blending: per-target nonnegative, sum-to-1, no-intercept least MSE
combination on shared OOF predictions (v4 §3). Nonnegative, sum-to-1, no-intercept least MSE combination on shared
rolling-origin OOF predictions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np
from scipy.optimize import nnls

from model_ensemble.artifacts import PerTargetWeightsArtifact
from model_ensemble.methods.weighted import target_key, weight_matrix

METHOD_NAME = "linear_blending"


def fit_linear_blending(
    oof_values_by_member: Mapping[str, np.ndarray],
    actual: np.ndarray,
    *,
    fallback_weights: Mapping[str, float] | None = None,
) -> PerTargetWeightsArtifact:
    names = tuple(oof_values_by_member)
    if len(names) < 2:
        raise ValueError("linear_blending requires at least two members")
    stacked = np.stack(
        [np.asarray(oof_values_by_member[name], dtype=float) for name in names],
        axis=0,
    )  # (members, n, H, K)
    k = stacked.shape[3]
    if fallback_weights is None:
        fallback = np.full(len(names), 1.0 / len(names))
    else:
        fallback = np.array([fallback_weights[name] for name in names])

    weights_by_target: dict[str, tuple[float, ...]] = {}
    for index in range(k):
        design = stacked[:, :, :, index].reshape(len(names), -1).T  # (n*H, members)
        target = actual[:, :, index].reshape(-1)
        coefficients, _ = nnls(design, target)
        total = float(coefficients.sum())
        if not np.isfinite(total) or total <= 0.0:
            weights = fallback
        else:
            weights = coefficients / total
        weights_by_target[target_key(index)] = tuple(
            float(w) for w in weights
        )
    return PerTargetWeightsArtifact(
        method_name=METHOD_NAME,
        weights_by_target=weights_by_target,
        coefficients=True,
    )


def combine_linear_blending(
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


def fit_nonnegative_stacking_weights(
    member_predictions, actual, *, fallback_weights
):
    """v4 compatibility function: legacy per-target NNLS weights.

    Value-for-value identical to the pre-redesign
    E0 parity fixture contract (function-level golden); new code should
    use `fit_linear_blending`.
    """
    if isinstance(member_predictions, (str, bytes)):
        raise TypeError("member_predictions must be a sequence or mapping")
    if isinstance(member_predictions, Mapping):
        member_predictions = list(member_predictions.values())
    if isinstance(member_predictions, np.ndarray) or not isinstance(
        member_predictions, Sequence
    ):
        raise TypeError("member_predictions must be a sequence")
    predictions = tuple(
        np.asarray(value, dtype=float) for value in member_predictions
    )
    if len(predictions) < 2:
        raise ValueError("stacking requires at least two member predictions")
    reference_shape = predictions[0].shape
    if any(value.shape != reference_shape for value in predictions):
        raise ValueError("stacking member prediction shapes must match")
    target = np.asarray(actual, dtype=float)
    if target.shape != reference_shape:
        raise ValueError("stacking actual shape must match member predictions")
    if not np.isfinite(target).all() or any(
        not np.isfinite(value).all() for value in predictions
    ):
        raise ValueError("stacking inputs must be finite")
    if isinstance(fallback_weights, Mapping):
        fallback_weights = list(fallback_weights.values())
    if isinstance(fallback_weights, (str, bytes)) or not isinstance(
        fallback_weights, Sequence
    ):
        raise TypeError("fallback_weights must be a sequence")
    fallback = tuple(float(weight) for weight in fallback_weights)
    if len(fallback) != len(predictions):
        raise ValueError("fallback_weights must match member count")
    fallback_array = np.asarray(fallback, dtype=float)
    if not np.isfinite(fallback_array).all() or np.any(fallback_array < 0.0):
        raise ValueError("fallback_weights must be finite and nonnegative")
    if not np.isclose(fallback_array.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("fallback_weights must sum to 1")
    design = np.column_stack([value.reshape(-1) for value in predictions])
    coefficients, _ = nnls(design, target.reshape(-1))
    total = float(coefficients.sum())
    if not np.isfinite(total) or total <= 0.0:
        return fallback
    return tuple(float(value) for value in coefficients / total)


__all__ = ["METHOD_NAME", "combine_linear_blending", "fit_linear_blending", "fit_nonnegative_stacking_weights"]
