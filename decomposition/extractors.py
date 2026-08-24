# -*- coding: utf-8 -*-
"""Extractor 组件：在可见训练窗口内把 y 拆成结构化分量。

语义与旧 TargetDecomposer.fit 逐位一致，包括 Phase 0 的两处修复：
- linear+damped 近期斜率从观测 y 的 lookback 段估计（偏差 #1）；
- MSTL 通过 stl_kwargs 透传 robust。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from decomposition.base import ComponentExtractor, to_datetime_index


def _t_index(times: pd.DatetimeIndex, origin_ns: int) -> np.ndarray:
    return (times.asi8 - origin_ns) / 86400.0 / 1e9


class NoneExtractor(ComponentExtractor):
    """不分解；components 为空。"""

    def fit(self, history: pd.DataFrame, time_col: str, target_col: str) -> "NoneExtractor":
        self.is_fitted = True
        return self

    def components(self) -> dict[str, np.ndarray]:
        return {}


class PolyTrendExtractor(ComponentExtractor):
    """一次/二次多项式趋势抽取（旧 linear 分支）。"""

    def __init__(self, params: dict[str, Any]):
        self.degree = int(params.get("degree", 1))
        if self.degree not in {1, 2}:
            raise ValueError("decomposition_trend_degree must be 1 or 2.")
        # damped 外推参数由 trend forecaster 消费，但近期斜率必须在此处从观测 y
        # 的 lookback 段估计——linear 的趋势分量是全局多项式拟合线，对它取任何
        # 窗口重拟合斜率都相同（Phase 0 偏差 #1）。
        self.trend_forecast_mode = str(params.get("mode", "polynomial")).lower()
        self.lookback = int(params.get("lookback", 28))
        self.origin_ns: int | None = None
        self.trend_coefficients: np.ndarray | None = None
        self.last_trend: float = 0.0
        self.last_trend_slope_per_day: float = 0.0
        super().__init__()

    def _resolve_lookback(self, n_obs: int) -> int:
        return min(n_obs, max(3, self.lookback))

    def fit(self, history: pd.DataFrame, time_col: str, target_col: str) -> "PolyTrendExtractor":
        times = to_datetime_index(history[time_col])
        y = np.asarray(pd.to_numeric(history[target_col], errors="coerce"), dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("Target decomposition does not accept NaN or infinite target values.")
        self.capture_time_axis(times)
        self.origin_ns = int(times.asi8[0])

        x = _t_index(times, self.origin_ns)
        self.trend_coefficients = np.polyfit(x, y, self.degree)
        trend = np.polyval(self.trend_coefficients, x)
        self.last_trend = float(trend[-1])
        # damped 近期斜率从观测 y 的 lookback 段估计（Phase 0 修复）
        if self.trend_forecast_mode == "damped":
            lookback = self._resolve_lookback(len(y))
            recent_x = x[-lookback:]
            self.last_trend_slope_per_day = float(np.polyfit(recent_x, y[-lookback:], 1)[0])
        else:
            self.last_trend_slope_per_day = float(np.polyfit(x, trend, 1)[0])
        self.is_fitted = True
        return self

    def components(self) -> dict[str, np.ndarray]:
        assert self.origin_ns is not None and self.n_obs > 0
        x = (np.arange(self.n_obs) * self.step_ns) / 86400.0 / 1e9
        trend = np.polyval(self.trend_coefficients, x)
        return {"trend": trend}


class _SeasonalExtractorBase(ComponentExtractor):
    """STL/MSTL 共用逻辑：分解 + 趋势多项式外推拟合 + 相位模板。"""

    def __init__(self, params: dict[str, Any], n_per_day_fallback: int = 0):
        self.periods_configured = [int(p) for p in (params.get("periods") or [])]
        self.robust = bool(params.get("robust", True))
        self.n_per_day_fallback = int(n_per_day_fallback or 0)
        # 趋势外推参数（forecaster 阶段消费）
        self.trend_degree = int(params.get("degree", 1))
        self.trend_forecast_mode = str(params.get("mode", "polynomial")).lower()
        self.damping = float(params.get("damping", 0.98))
        self.lookback = int(params.get("lookback", 28))
        self.seasonal_cycles = int(params.get("cycles", 4))

        self.method_name = ""
        self.origin_ns: int | None = None
        self.exact_deterministic: np.ndarray | None = None
        self.trend_coefficients: np.ndarray | None = None
        self.last_trend: float = 0.0
        self.last_trend_slope_per_day: float = 0.0
        self.seasonal_templates: list[tuple[int, np.ndarray]] = []
        super().__init__()

    def _resolve_periods(self, n_obs: int) -> list[int]:
        configured = list(self.periods_configured)
        if not configured:
            n_per_day = self.n_per_day_fallback
            if n_per_day >= 2:
                configured = [n_per_day]
        periods = [int(p) for p in configured]
        if not periods or any(p < 2 for p in periods):
            raise ValueError(
                f"decomposition_method={self.method_name} requires periods >= 2."
            )
        if any(2 * p > n_obs for p in periods):
            raise ValueError(
                f"periods={periods} require at least two full cycles in {n_obs} observations."
            )
        return periods

    def _fit_trend_forecast(self, trend: np.ndarray, times: pd.DatetimeIndex) -> None:
        assert self.origin_ns is not None
        x = _t_index(times, self.origin_ns)
        degree = self.trend_degree
        if degree not in {1, 2}:
            raise ValueError("decomposition_trend_degree must be 1 or 2.")
        self.trend_coefficients = np.polyfit(x, trend, degree)
        self.last_trend = float(trend[-1])
        lookback = min(len(trend), max(3, self.lookback))
        recent_x = x[-lookback:]
        recent_trend = trend[-lookback:]
        self.last_trend_slope_per_day = float(np.polyfit(recent_x, recent_trend, 1)[0])

    def _build_seasonal_templates(self, seasonal: np.ndarray, periods: list[int]) -> None:
        seasonal_2d = np.asarray(seasonal, dtype=float)
        if seasonal_2d.ndim == 1:
            seasonal_2d = seasonal_2d[:, None]
        if seasonal_2d.shape[1] != len(periods):
            raise ValueError("Seasonal component count does not match decomposition_periods.")
        cycles = max(1, self.seasonal_cycles)
        self.seasonal_templates = []
        positions = np.arange(self.n_obs)
        for idx, period in enumerate(periods):
            component = seasonal_2d[:, idx]
            start = max(0, self.n_obs - cycles * period)
            template = np.empty(period, dtype=float)
            for phase in range(period):
                mask = (positions >= start) & (positions % period == phase)
                values = component[mask]
                template[phase] = float(np.mean(values)) if len(values) else 0.0
            self.seasonal_templates.append((period, template))


class STLExtractor(_SeasonalExtractorBase):
    """单周期 STL 分解。"""

    def __init__(self, params: dict[str, Any], n_per_day_fallback: int = 0):
        super().__init__(params, n_per_day_fallback)
        self.method_name = "stl"

    def fit(self, history: pd.DataFrame, time_col: str, target_col: str) -> "STLExtractor":
        from statsmodels.tsa.seasonal import STL

        times = to_datetime_index(history[time_col])
        y = np.asarray(pd.to_numeric(history[target_col], errors="coerce"), dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("Target decomposition does not accept NaN or infinite target values.")
        self.capture_time_axis(times)
        self.origin_ns = int(times.asi8[0])

        periods = self._resolve_periods(len(y))
        result = STL(y, period=periods[0], robust=self.robust).fit()
        trend = np.asarray(result.trend, dtype=float)
        seasonal = np.asarray(result.seasonal, dtype=float)[:, None]

        self._fit_trend_forecast(trend, times)
        self._build_seasonal_templates(seasonal, periods)
        # 历史 exact 确定分量 = STL 原始输出（与旧实现逐位一致）
        self.exact_deterministic = trend + seasonal.sum(axis=1)
        self.is_fitted = True
        return self

    def components(self) -> dict[str, np.ndarray]:
        assert self.origin_ns is not None and self.n_obs > 0
        x = (np.arange(self.n_obs) * self.step_ns) / 86400.0 / 1e9
        trend = np.polyval(self.trend_coefficients, x)
        seasonal_total = np.zeros(self.n_obs, dtype=float)
        positions = np.arange(self.n_obs)
        for period, template in self.seasonal_templates:
            seasonal_total += template[np.mod(positions, period)]
        # 历史 exact 分量必须用 STL/MSTL 原始输出（fit 阶段缓存），
        # 不能用 polyfit 重构值——两者存在拟合误差，会导致历史 transform
        # 与旧实现不等价。
        assert self.exact_deterministic is not None
        return {
            "trend": trend,
            "seasonal": seasonal_total,
            "__exact_deterministic__": self.exact_deterministic,
        }


class MSTLExtractor(STLExtractor):
    """多周期 MSTL 分解（robust 经 stl_kwargs 透传，Phase 0 修复）。"""

    def __init__(self, params: dict[str, Any], n_per_day_fallback: int = 0):
        super().__init__(params, n_per_day_fallback)
        self.method_name = "mstl"

    def fit(self, history: pd.DataFrame, time_col: str, target_col: str) -> "MSTLExtractor":
        from statsmodels.tsa.seasonal import MSTL

        times = to_datetime_index(history[time_col])
        y = np.asarray(pd.to_numeric(history[target_col], errors="coerce"), dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("Target decomposition does not accept NaN or infinite target values.")
        self.capture_time_axis(times)
        self.origin_ns = int(times.asi8[0])

        periods = self._resolve_periods(len(y))
        result = MSTL(y, periods=tuple(periods), stl_kwargs={"robust": self.robust}).fit()
        trend = np.asarray(result.trend, dtype=float)
        seasonal = np.asarray(result.seasonal, dtype=float)
        if seasonal.ndim == 1:
            seasonal = seasonal[:, None]

        self._fit_trend_forecast(trend, times)
        self._build_seasonal_templates(seasonal, periods)
        # 历史 exact 确定分量 = MSTL 原始输出
        self.exact_deterministic = trend + seasonal.sum(axis=1)
        self.is_fitted = True
        return self
