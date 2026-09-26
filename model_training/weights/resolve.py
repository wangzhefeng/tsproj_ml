"""训练样本权重的配置接线：spec → origins → 权重数组。

消费点为编排层的拟合入口（model_pipeline/fold_fit.py）。本模块只做
「配置翻译 + 前置校验」，不感知模型与特征；权重算法本体在 temporal.py。
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from models.catalog import MODEL_CATALOG
from model_training.weights.temporal import temporal_sample_weight


def training_sample_weight_spec(config) -> Mapping[str, Any] | None:
    """读取 validation.training.sample_weight 声明；未声明返回 None。"""
    training = config.validation.get("training")
    if not isinstance(training, Mapping):
        return None
    return training.get("sample_weight")


def resolve_training_sample_weight(
    config,
    origins: tuple[pd.Timestamp, ...],
    *,
    history_cutoff: pd.Timestamp,
    sample_weight_capable: bool,
) -> np.ndarray | None:
    """把 YAML 声明翻译为训练权重数组（fit 入口的唯一接线函数）。

    Args:
        config: ForecastConfigSpec（读取 validation.training.sample_weight）。
        origins: 本折训练样本的监督原点（与 X/Y 行轴对齐）。
        history_cutoff: 训练历史截止（回测几何的标签可见性边界）。
        sample_weight_capable: 目标模型的 catalog 能力位；声明了加权
            但模型不支持时在此前置 RAISE，而不是沉到 adapter 深处。

    Returns:
        与 origins 等长的权重数组；未声明加权时返回 None。
    """
    spec = training_sample_weight_spec(config)
    if spec is None:
        return None
    if not sample_weight_capable:
        raise ValueError(
            f"model_type {config.estimator.model_type!r} declares "
            "sample_weight=false in catalog; validation.training.sample_weight "
            "cannot be enabled for this estimator"
        )
    anchor_mode = str(spec.get("anchor", "cutoff"))
    anchor = history_cutoff if anchor_mode == "cutoff" else max(origins)
    return temporal_sample_weight(origins, anchor, spec)
