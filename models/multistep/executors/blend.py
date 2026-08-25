# -*- coding: utf-8 -*-
import numpy as np

from models.multistep.contracts import require_direct_output
from models.multistep.executors.direct import DirectExecutor
from models.multistep.executors.recursive import RecursiveExecutor
from models.multistep.runtime import context_horizon, log_execution_summary
from models.multistep.weights import BlendWeights
from probabilistic.types import BlendQuantileModel, ProbabilisticModelBundle


def combine_exact_paths(direct, recursive, weights, horizon: int) -> np.ndarray:
    if isinstance(weights, BlendWeights):
        return weights.combine(direct, recursive, horizon)
    direct_path = require_direct_output(direct, horizon)
    try:
        recursive_path = require_direct_output(recursive, horizon)
    except ValueError as exc:
        raise ValueError(
            f"recursive blend path length mismatch: expected {horizon}, got {np.asarray(recursive).shape}."
        ) from exc
    normalized = np.asarray(weights, dtype=float).reshape(-1)
    if normalized.shape != (2,) or not np.isfinite(normalized).all() or normalized.sum() <= 0:
        raise ValueError("blend weights must contain two finite positive-sum values.")
    normalized = normalized / normalized.sum()
    return normalized[0] * direct_path + normalized[1] * recursive_path


class BlendExecutor:
    def execute(self, context):
        if str(getattr(context.args, "predict_type", "point")).lower() == "quantile":
            return self._execute_quantile(context)
        horizon = context_horizon(context)
        direct_context = context._fork_for_model(context.blend_direct_model)
        recursive_context = context._fork_for_model(context.blend_recursive_model)
        direct_path = DirectExecutor().execute(direct_context)
        recursive_path = RecursiveExecutor().execute(recursive_context)
        context.blend_direct_pred = np.asarray(direct_path, dtype=float)
        context.blend_recursive_pred = np.asarray(recursive_path, dtype=float)
        weights = context._resolve_blend_weights()
        result = combine_exact_paths(
            direct_path,
            recursive_path,
            weights,
            horizon,
        )
        log_execution_summary(
            context,
            "blend",
            model_calls=horizon + 1,
            weight_source=(
                weights.strategy if isinstance(weights, BlendWeights) else "runtime_array"
            ),
        )
        return result

    def _execute_quantile(self, context):
        horizon = context_horizon(context)
        model = context.model
        if not isinstance(model, ProbabilisticModelBundle):
            raise ValueError("blend quantile requires a typed quantile bundle.")
        if not model.is_blend:
            raise ValueError("typed blend forecast requires BlendQuantileModel entries.")
        quantile_models = model.models_by_quantile
        median_level = model.spec.point_quantile
        if not quantile_models:
            raise ValueError("blend quantile bundle has no sub-models.")

        weights = context._resolve_blend_weights()
        quantile_paths = {}
        direct_reference = None
        recursive_reference = None
        for raw_level, pair in quantile_models.items():
            level = float(raw_level)
            if not isinstance(pair, BlendQuantileModel):
                raise ValueError(
                    f"typed blend quantile q={level:g} is not a BlendQuantileModel."
                )
            direct_model, recursive_model = pair.direct, pair.recursive
            direct_context = context._fork_for_model(direct_model)
            recursive_context = context._fork_for_model(recursive_model)
            direct_path = DirectExecutor().execute(direct_context)
            recursive_path = RecursiveExecutor().execute(recursive_context)
            quantile_paths[level] = combine_exact_paths(
                direct_path, recursive_path, weights, horizon
            )
            if abs(level - median_level) < 1e-9:
                direct_reference = direct_path
                recursive_reference = recursive_path
        if median_level not in quantile_paths:
            median_level = min(quantile_paths, key=lambda value: abs(value - median_level))
        context._quantile_outputs = quantile_paths
        context.blend_direct_pred = np.asarray(direct_reference, dtype=float) if direct_reference is not None else None
        context.blend_recursive_pred = np.asarray(recursive_reference, dtype=float) if recursive_reference is not None else None
        log_execution_summary(
            context,
            "blend",
            model_calls=len(quantile_models) * (horizon + 1),
            weight_source=(
                weights.strategy if isinstance(weights, BlendWeights) else "runtime_array"
            ),
        )
        return quantile_paths[median_level]
