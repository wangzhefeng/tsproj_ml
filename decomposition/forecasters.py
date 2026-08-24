# -*- coding: utf-8 -*-
"""Forecaster 组件：把结构化分量外推到未来时点。

- PolyTrendForecastForecaster：polynomial / damped 趋势外推（旧 _forecast_trend 语义）
- PhaseTemplateForecaster：按最近若干周期的相位均值模板外推（旧 _forecast_seasonal 语义）
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from decomposition.base import ComponentForecaster
from decomposition.types import ComponentFrame


class PolyTrendForecastForecaster(ComponentForecaster):
    """基于多项式系数 + 可选 damped 近期斜率的趋势外推。"""

    def __init__(self, params: dict[str, Any]):
        self.mode = str(params.get("mode", "polynomial")).lower()
        if self.mode not in {"polynomial", "damped"}:
            raise ValueError("decomposition_trend_forecast must be 'polynomial' or 'damped'.")
        self.damping = float(params.get("damping", 0.98))
        if not 0.0 < self.damping <= 1.0:
            raise ValueError("decomposition_damping must be in (0, 1].")
        self.trend_coefficients: np.ndarray | None = None
        self.last_time_ns: int | None = None
        self.last_trend: float = 0.0
        self.last_trend_slope_per_day: float = 0.0
        # polyval 时间原点（pipeline.fit 阶段从 extractor.origin_ns 桥接）
        self.trend_coefficients_context_ns: int = 0

    def fit(self, frame: ComponentFrame) -> None:
        # 趋势系数/近期斜率在 extractor.fit 阶段已拟合；此处从 extractor 状态桥接。
        # 由 pipeline 负责注入（见 pipeline.py），fit 只做时间轴捕获。
        self.last_time_ns = int(frame.times.asi8[-1]) if len(frame.times) else None

    def attach(
        self,
        trend_coefficients: np.ndarray,
        last_time_ns: int,
        last_trend: float,
        last_trend_slope_per_day: float,
    ) -> None:
        self.trend_coefficients = trend_coefficients
        self.last_time_ns = last_time_ns
        self.last_trend = last_trend
        self.last_trend_slope_per_day = last_trend_slope_per_day

    def predict(self, future_times: pd.DatetimeIndex) -> np.ndarray | None:
        if self.trend_coefficients is None:
            return np.zeros(len(future_times), dtype=float)
        origin_ns = self.trend_coefficients_context_ns
        x = (future_times.asi8 - origin_ns) / 86400.0 / 1e9
        trend = np.polyval(self.trend_coefficients, x)
        if self.mode == "polynomial":
            return np.asarray(trend, dtype=float)
        assert self.last_time_ns is not None
        days_ahead = np.maximum(0.0, (future_times.asi8 - self.last_time_ns) / 86400.0 / 1e9)
        phi = self.damping
        if phi == 1.0:
            increments = days_ahead
        else:
            increments = phi * (1.0 - np.power(phi, days_ahead)) / (1.0 - phi)
        future = self.last_trend + self.last_trend_slope_per_day * increments
        return np.where(days_ahead > 0.0, future, trend)


class PhaseTemplateForecaster(ComponentForecaster):
    """按最近若干周期的相位均值模板外推季节分量。"""

    def __init__(self, params: dict[str, Any]):
        self.templates: list[tuple[int, np.ndarray]] = []
        self.n_obs: int = 0
        self.last_time_ns: int | None = None
        self.step_ns: int | None = None

    def fit(self, frame: ComponentFrame) -> None:
        # 模板在 extractor.fit 阶段已构造；由 pipeline 注入。
        self.last_time_ns = int(frame.times.asi8[-1]) if len(frame.times) else None

    def attach(
        self,
        templates: list[tuple[int, np.ndarray]],
        n_obs: int,
        last_time_ns: int,
        step_ns: int,
    ) -> None:
        self.templates = templates
        self.n_obs = n_obs
        self.last_time_ns = last_time_ns
        self.step_ns = step_ns

    def predict(self, future_times: pd.DatetimeIndex) -> np.ndarray | None:
        if not self.templates:
            return np.zeros(len(future_times), dtype=float)
        assert self.last_time_ns is not None and self.step_ns is not None
        positions = self.n_obs - 1 + np.rint(
            (future_times.asi8 - self.last_time_ns) / self.step_ns
        ).astype(int)
        seasonal = np.zeros(len(future_times), dtype=float)
        for period, template in self.templates:
            seasonal += template[np.mod(positions, period)]
        return seasonal
