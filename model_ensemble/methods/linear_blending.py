"""Per-target simplex blending on shared rolling-origin OOF predictions.

Point mode learns normalized nonnegative NNLS coefficients. Quantile mode
minimizes pooled pinball loss with one target-level weight vector shared across
all configured marginal quantiles. Every optimization persists an audit trail.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize, nnls

from model_ensemble.artifacts import PerTargetWeightsArtifact
from model_ensemble.methods.weighted import target_key, weight_matrix

METHOD_NAME = "linear_blending"


def fit_linear_blending(
    oof_values_by_member: Mapping[str, np.ndarray],
    actual: np.ndarray,
    *,
    fallback_weights: Mapping[str, float] | None = None,
    quantile_levels: Sequence[float] | None = None,
) -> PerTargetWeightsArtifact:
    names = tuple(oof_values_by_member)
    if len(names) < 2:
        raise ValueError("linear_blending requires at least two members")
    stacked = np.stack(
        [np.asarray(oof_values_by_member[name], dtype=float) for name in names],
        axis=0,
    )
    target_values = np.asarray(actual, dtype=float)
    if stacked.ndim not in {4, 5}:
        raise ValueError("linear_blending members must have shape (N,H,K[,Q])")
    if target_values.shape != stacked.shape[1:4]:
        raise ValueError(
            "linear_blending actual must match member predictions on (N,H,K)"
        )
    if not np.isfinite(stacked).all() or not np.isfinite(target_values).all():
        raise ValueError("linear_blending inputs must be finite")

    k = stacked.shape[3]
    if fallback_weights is None:
        fallback = np.full(len(names), 1.0 / len(names))
    else:
        fallback = np.array([fallback_weights[name] for name in names], dtype=float)
    if (
        fallback.shape != (len(names),)
        or not np.isfinite(fallback).all()
        or np.any(fallback < 0.0)
        or not np.isclose(fallback.sum(), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError(
            "fallback_weights must be finite, nonnegative, and sum to 1"
        )

    levels: np.ndarray | None = None
    if stacked.ndim == 5:
        if quantile_levels is None:
            raise ValueError(
                "quantile linear_blending requires explicit quantile_levels"
            )
        levels = np.asarray(tuple(quantile_levels), dtype=float)
        if (
            levels.shape != (stacked.shape[4],)
            or not np.isfinite(levels).all()
            or np.any(levels <= 0.0)
            or np.any(levels >= 1.0)
            or np.any(np.diff(levels) <= 0.0)
        ):
            raise ValueError(
                "quantile_levels must be unique, increasing, and inside (0, 1)"
            )
    elif quantile_levels is not None:
        raise ValueError("point linear_blending forbids quantile_levels")

    weights_by_target: dict[str, tuple[float, ...]] = {}
    n_samples_by_target: dict[str, int] = {}
    optimizer_success_by_target: dict[str, bool] = {}
    optimizer_status_by_target: dict[str, int] = {}
    optimizer_message_by_target: dict[str, str] = {}
    fallback_reason_by_target: dict[str, str | None] = {}
    for index in range(k):
        key = target_key(index)
        n_samples_by_target[key] = int(target_values[:, :, index].size)
        if levels is None:
            design = stacked[:, :, :, index].reshape(len(names), -1).T
            target = target_values[:, :, index].reshape(-1)
            coefficients, _ = nnls(design, target)
            total = float(coefficients.sum())
            success = bool(np.isfinite(total) and total > 0.0)
            weights = coefficients / total if success else fallback
            optimizer_success_by_target[key] = success
            optimizer_status_by_target[key] = 0 if success else 1
            optimizer_message_by_target[key] = (
                "nnls coefficients normalized"
                if success
                else "nnls returned degenerate coefficient total"
            )
            fallback_reason_by_target[key] = (
                None if success else "degenerate_nnls_coefficients"
            )
        else:
            member_quantiles = stacked[:, :, :, index, :]
            target = target_values[:, :, index, None]

            def pooled_pinball(weights_vector: np.ndarray) -> float:
                prediction = np.tensordot(
                    weights_vector,
                    member_quantiles,
                    axes=(0, 0),
                )
                error = target - prediction
                loss = np.maximum(
                    levels[None, None, :] * error,
                    (levels[None, None, :] - 1.0) * error,
                )
                return float(np.mean(loss))

            optimized = minimize(
                pooled_pinball,
                fallback,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * len(names),
                constraints={
                    "type": "eq",
                    "fun": lambda values: float(np.sum(values) - 1.0),
                },
                options={"ftol": 1e-12, "maxiter": 1000},
            )
            candidate = np.asarray(optimized.x, dtype=float)
            optimizer_success = bool(getattr(optimized, "success", False))
            optimizer_status = int(getattr(optimized, "status", -1))
            optimizer_message = str(getattr(optimized, "message", ""))
            fallback_reason: str | None = None
            if not optimizer_success:
                fallback_reason = f"optimizer_failed: {optimizer_message}"
            elif candidate.shape != fallback.shape:
                fallback_reason = "invalid_optimizer_shape"
            elif not np.isfinite(candidate).all():
                fallback_reason = "nonfinite_optimizer_weights"
            elif np.any(candidate < -1e-10):
                fallback_reason = "negative_optimizer_weights"
            elif not np.isclose(candidate.sum(), 1.0, atol=1e-8):
                fallback_reason = "optimizer_weights_not_simplex"
            valid_candidate = fallback_reason is None
            weights = candidate if valid_candidate else fallback
            weights = np.maximum(weights, 0.0)
            weights = weights / weights.sum()
            optimizer_success_by_target[key] = bool(
                optimizer_success and valid_candidate
            )
            optimizer_status_by_target[key] = optimizer_status
            optimizer_message_by_target[key] = optimizer_message
            fallback_reason_by_target[key] = fallback_reason

        weights_by_target[key] = tuple(float(w) for w in weights)
    return PerTargetWeightsArtifact(
        method_name=METHOD_NAME,
        weights_by_target=weights_by_target,
        metric="pinball" if levels is not None else "mse",
        coefficients=True,
        quantile_levels=(
            tuple(float(level) for level in levels)
            if levels is not None
            else None
        ),
        n_samples_by_target=n_samples_by_target,
        optimizer_success_by_target=optimizer_success_by_target,
        optimizer_status_by_target=optimizer_status_by_target,
        optimizer_message_by_target=optimizer_message_by_target,
        fallback_reason_by_target=fallback_reason_by_target,
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
    if stacked.ndim == 5:
        return np.einsum("km,mnhkq->nhkq", weights_2d, stacked)
    raise ValueError("member values must have shape (N,H,K[,Q])")


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
