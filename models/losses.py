# -*- coding: utf-8 -*-

"""
损失函数与调参 scorer 映射工具。
"""

from typing import Callable, Union

import numpy as np
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)


def huber_loss(y_true, y_pred, delta: float = 1.0) -> float:
    error = np.asarray(y_true) - np.asarray(y_pred)
    abs_error = np.abs(error)
    quadratic = 0.5 * np.square(error)
    linear = delta * (abs_error - 0.5 * delta)
    return float(np.where(abs_error <= delta, quadratic, linear).mean())


def rmse_loss(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_scorer_by_loss_name(loss_name: str, delta: float = 1.0) -> Union[str, Callable]:
    """
    返回可用于 sklearn CV 的 scoring 参数。
    """
    name = (loss_name or "mae").strip().lower()
    if name in {"mae", "l1", "regression_l1"}:
        return "neg_mean_absolute_error"
    if name in {"mse", "l2", "regression"}:
        return "neg_mean_squared_error"
    if name in {"rmse"}:
        return make_scorer(rmse_loss, greater_is_better=False)
    if name in {"mape"}:
        return make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    if name in {"huber", "huber_loss"}:
        return make_scorer(lambda y_true, y_pred: huber_loss(y_true, y_pred, delta=delta), greater_is_better=False)
    return "neg_mean_absolute_error"


def get_default_objective_and_metric(loss_name: str):
    """
    返回 tree 模型常用 objective/metric 映射。
    """
    name = (loss_name or "mae").strip().lower()
    if name in {"mae", "l1", "regression_l1"}:
        return "regression_l1", "mae"
    if name in {"mse", "l2", "regression"}:
        return "regression", "rmse"
    if name in {"huber", "huber_loss"}:
        return "huber", "mae"
    return "regression_l1", "mae"
