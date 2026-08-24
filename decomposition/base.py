# -*- coding: utf-8 -*-
"""分解组件抽象基类与共享时间轴工具。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from decomposition.types import TARGET_KEY, ComponentFrame


def to_datetime_index(values: Any) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values))


class TimeAxisMixin:
    """捕获并校验规则等间隔时间轴（与旧 TargetDecomposer 逐位一致）。"""

    origin_ns: int | None = None
    last_time_ns: int | None = None
    step_ns: int | None = None
    n_obs: int = 0

    def capture_time_axis(self, times: pd.DatetimeIndex) -> None:
        if len(times) < 2:
            raise ValueError("Target decomposition requires at least two observations.")
        diffs = np.diff(times.asi8)
        if np.any(diffs <= 0):
            raise ValueError("Target decomposition requires strictly increasing timestamps.")
        step_ns = int(np.median(diffs))
        if np.any(diffs != step_ns):
            raise ValueError("Target decomposition requires a regular time index.")
        self.origin_ns = int(times.asi8[0])
        self.last_time_ns = int(times.asi8[-1])
        self.step_ns = step_ns
        self.n_obs = len(times)


class ComponentExtractor(TimeAxisMixin, ABC):
    """在可见训练窗口内把 y 拆成结构化分量。"""

    is_fitted: bool = False

    @abstractmethod
    def fit(self, history: pd.DataFrame, time_col: str, target_col: str) -> "ComponentExtractor":
        ...

    def transform(self, history: pd.DataFrame, time_col: str, target_col: str) -> ComponentFrame:
        times = to_datetime_index(history[time_col])
        y = np.asarray(pd.to_numeric(history[target_col], errors="coerce"), dtype=float)
        components = self.components()
        return ComponentFrame(times=times, components={**components, TARGET_KEY: y})

    def fit_transform(
        self, history: pd.DataFrame, time_col: str = "time", target_col: str = "y"
    ) -> ComponentFrame:
        self.fit(history, time_col=time_col, target_col=target_col)
        return self.transform(history, time_col=time_col, target_col=target_col)

    @abstractmethod
    def components(self) -> dict[str, np.ndarray]:
        """返回历史区间各结构化分量的 numpy 数组。"""
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
