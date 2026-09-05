# -*- coding: utf-8 -*-
"""模型类型到原生 quantile objective 的显式能力映射。"""

from __future__ import annotations

from forecasting_core.probabilistic_spec import ProbabilisticSpec
from model_training.estimators.capabilities import resolve_model_capabilities


def supports_quantile_objective(model_type: str) -> bool:
    """复用训练层由模型目录派生的能力，不维护第二份别名/注入表。"""
    try:
        return resolve_model_capabilities(model_type).scalar_quantile
    except KeyError:
        return False


def validate_quantile_model_support(
    model_type: str,
    spec: ProbabilisticSpec,
) -> None:
    """在模型构造边界拒绝把普通回归器伪装成 quantile 模型。"""
    if spec.mode != "quantile":
        return
    if not supports_quantile_objective(model_type):
        raise ValueError(
            f"model_type={model_type!r} does not support native quantile prediction"
        )
