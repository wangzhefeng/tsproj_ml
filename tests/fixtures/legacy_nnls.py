"""冻结的旧 NNLS 黄金参照；独立于现役融合实现，仅供测试。"""

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import nnls


def fit_nonnegative_stacking_weights(
    member_predictions, actual, *, fallback_weights
):
    """保留迁移前函数的运算及异常合同，不委托现役实现。"""
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
