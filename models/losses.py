# -*- coding: utf-8 -*-

"""
损失函数与调参 scorer 映射工具。
"""

from typing import Any, Callable, Dict, Union

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


def _normalize_loss_name(loss_name: Any) -> str:
    if isinstance(loss_name, (list, tuple)) and loss_name:
        loss_name = loss_name[0]
    text = str(loss_name or "mae").strip().lower()
    text = text.split(":", 1)[0]
    aliases = {
        "mae": "mae",
        "l1": "mae",
        "regression_l1": "mae",
        "reg:absoluteerror": "mae",
        "meanabsoluteerror": "mae",
        "absolute_error": "mae",
        "absoluteerror": "mae",
        "mse": "mse",
        "l2": "mse",
        "regression": "mse",
        "reg:squarederror": "mse",
        "meansquarederror": "mse",
        "squared_error": "mse",
        "squarederror": "mse",
        "rmse": "rmse",
        "rootmeansquarederror": "rmse",
        "mape": "mape",
        "huber": "huber",
        "huber_loss": "huber",
    }
    return aliases.get(text, text)


def get_loss_name_from_model_params(model_type: str, model_params: Dict[str, Any]) -> str:
    """
    从各模型原生参数中提取统一的损失名称，用于 sklearn scorer 映射。
    """
    params = model_params or {}
    model_key = str(model_type or "").strip().lower()

    raw_loss = None
    if model_key in {"lightgbm", "lgb"}:
        raw_loss = params.get("metric") or params.get("objective")
    elif model_key in {"xgboost", "xgb"}:
        raw_loss = params.get("eval_metric") or params.get("objective")
    elif model_key in {"catboost", "cat"}:
        raw_loss = params.get("eval_metric") or params.get("loss_function")
    elif model_key in {"histgb", "histgradientboosting"}:
        raw_loss = params.get("loss")
    elif model_key in {"ridge", "elasticnet", "enet", "lasso"}:
        # 线性正则回归的训练目标均为 MSE 系
        raw_loss = "mse"
    elif model_key in {"quantileregressor", "qr"}:
        # 线性分位数回归（默认 quantile=0.5 即 MAE 口径）
        raw_loss = "mae"

    return _normalize_loss_name(raw_loss)
