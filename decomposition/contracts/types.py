# -*- coding: utf-8 -*-
"""分解模块的共享数据类型。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass
class ComponentFrame:
    """真实历史分量；目标与结构分量分开存储，不混入外推拟合状态。"""

    times: pd.DatetimeIndex
    target: np.ndarray
    trend: np.ndarray
    seasonal: dict[int, np.ndarray]

    @property
    def seasonal_total(self) -> np.ndarray:
        if not self.seasonal:
            return np.zeros(len(self.times), dtype=float)
        return np.column_stack(list(self.seasonal.values())).sum(axis=1)

    @property
    def deterministic(self) -> np.ndarray:
        return self.trend + self.seasonal_total

    @property
    def residual(self) -> np.ndarray:
        return self.target - self.deterministic

    @property
    def components(self) -> dict[str, np.ndarray]:
        return {"trend": self.trend, **{f"seasonal_{p}": v for p, v in self.seasonal.items()}}


@dataclass
class ComponentForecast:
    """未来时点的结构化分量预测。"""

    times: pd.DatetimeIndex
    components: dict[str, np.ndarray]
