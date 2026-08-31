# -*- coding: utf-8 -*-
"""模型类型到原生 quantile objective 的显式能力映射。"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any, Callable

from forecasting_core.probabilistic_spec import ProbabilisticSpec


_MODEL_ALIASES = {
    "lgb": "lightgbm",
    "lightgbm": "lightgbm",
    "xgb": "xgboost",
    "xgboost": "xgboost",
    "cat": "catboost",
    "catboost": "catboost",
    "histgb": "histgb",
    "histgradientboosting": "histgb",
    "qr": "quantileregressor",
    "quantileregressor": "quantileregressor",
}


def _lightgbm(params: dict[str, Any], quantile: float) -> None:
    params.update(objective="quantile", metric="quantile", alpha=quantile)


def _xgboost(params: dict[str, Any], quantile: float) -> None:
    params.update(
        objective="reg:quantileerror",
        quantile_alpha=quantile,
        eval_metric="quantile",
    )


def _catboost(params: dict[str, Any], quantile: float) -> None:
    params["loss_function"] = f"Quantile:alpha={quantile}"


def _histgb(params: dict[str, Any], quantile: float) -> None:
    params.update(loss="quantile", quantile=quantile)


def _quantile_regressor(params: dict[str, Any], quantile: float) -> None:
    params["quantile"] = quantile


_OBJECTIVE_INJECTORS: dict[str, Callable[[dict[str, Any], float], None]] = {
    "lightgbm": _lightgbm,
    "xgboost": _xgboost,
    "catboost": _catboost,
    "histgb": _histgb,
    "quantileregressor": _quantile_regressor,
}


def normalize_quantile_model_type(model_type: str) -> str:
    """把受支持别名归一化；未知名称原样小写以生成明确错误。"""
    token = str(model_type).strip().lower()
    return _MODEL_ALIASES.get(token, token)


def supports_quantile_objective(model_type: str) -> bool:
    """模型是否具有本项目已实现的原生 quantile objective adapter。"""
    return normalize_quantile_model_type(model_type) in _OBJECTIVE_INJECTORS


def inject_quantile_objective(
    model_type: str,
    params: Mapping[str, Any],
    quantile: float,
) -> dict[str, Any]:
    """深拷贝参数并注入 quantile objective；不支持时 fail-fast。"""
    level = float(quantile)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("quantile must be finite and inside (0, 1)")
    canonical_type = normalize_quantile_model_type(model_type)
    injector = _OBJECTIVE_INJECTORS.get(canonical_type)
    if injector is None:
        raise ValueError(
            f"model_type={model_type!r} does not support native quantile prediction; "
            f"supported={sorted(_OBJECTIVE_INJECTORS)}"
        )
    result = copy.deepcopy(dict(params))
    injector(result, level)
    return result


def validate_quantile_model_support(
    model_type: str,
    spec: ProbabilisticSpec,
) -> None:
    """在模型构造边界拒绝把普通回归器伪装成 quantile 模型。"""
    if spec.mode != "quantile":
        return
    if not supports_quantile_objective(model_type):
        raise ValueError(
            f"model_type={model_type!r} does not support native quantile prediction; "
            f"supported={sorted(_OBJECTIVE_INJECTORS)}"
        )
