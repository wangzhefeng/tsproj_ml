# -*- coding: utf-8 -*-
"""Composer 组件：把 residual 预测与结构分量合回原始目标空间。

当前仅支持 additive；multiplicative 等为预留位（fail-fast）。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from decomposition.types import ComponentForecast


class AdditiveComposer:
    """加性组合：restored = residual_pred + deterministic_component。"""

    def __init__(self, params: dict[str, Any] | None = None):
        pass

    def compose(
        self,
        residual_pred: np.ndarray,
        component_forecast: ComponentForecast,
    ) -> np.ndarray:
        deterministic = np.zeros(len(residual_pred), dtype=float)
        for series in component_forecast.components.values():
            deterministic += series
        return np.asarray(residual_pred, dtype=float) + deterministic

    def restore_quantiles(
        self,
        quantile_map: dict[float, np.ndarray],
        component_forecast: ComponentForecast,
    ) -> dict[float, np.ndarray]:
        if not quantile_map:
            return {}
        n = len(next(iter(quantile_map.values())))
        deterministic = np.zeros(n, dtype=float)
        for series in component_forecast.components.values():
            deterministic += np.asarray(series, dtype=float)[:n]
        return {q: np.asarray(pred, dtype=float) + deterministic for q, pred in quantile_map.items()}
