# -*- coding: utf-8 -*-
"""历史分解与分量外推接口；时间轴验证由 time_axis 负责。"""
from __future__ import annotations

from abc import ABC, abstractmethod


import numpy as np
import pandas as pd

from decomposition.contracts.types import ComponentFrame


class ComponentExtractor(ABC):
    """在可见训练窗口内把 y 拆成结构化分量。"""

    @abstractmethod
    def fit_transform(
        self, history: pd.DataFrame, time_col: str = "time", target_col: str = "y"
    ) -> ComponentFrame:
        ...


class ComponentForecaster(ABC):
    """把历史结构化分量外推到未来时点。"""

    @abstractmethod
    def fit(self, frame: ComponentFrame) -> None:
        ...

    @abstractmethod
    def predict(self, future_times: pd.DatetimeIndex) -> np.ndarray | None:
        """返回未来分量数组；不产出分量的 forecaster 返回 None。"""
        ...
