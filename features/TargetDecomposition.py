# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


_SUPPORTED_METHODS = {"none", "linear", "stl", "mstl"}


def resolve_decomposition_method(args) -> str:
    """解析目标分解方法。"""
    method = str(getattr(args, "decomposition_method", "none") or "none").lower()
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported decomposition_method={method}. "
            f"Supported: {sorted(_SUPPORTED_METHODS)}"
        )
    return method


class TargetDecomposer:
    """在可见训练窗口内分解目标，并将分量预测还原到原始电平。"""

    def __init__(self, args=None, log_prefix: str = "[TargetDecomposer]", verbose: bool = False):
        self.args = args
        self.log_prefix = log_prefix
        self.verbose = verbose
        self.method = resolve_decomposition_method(args)
        self.enabled = self.method != "none"
        self.is_fitted = False
        self.time_col = "time"
        self.target_col = "y"
        self.origin_ns: Optional[int] = None
        self.last_time_ns: Optional[int] = None
        self.step_ns: Optional[int] = None
        self.n_obs = 0
        self.trend_coefficients: Optional[np.ndarray] = None
        self.history_component: Optional[pd.Series] = None
        self.seasonal_templates: list[tuple[int, np.ndarray]] = []
        self.last_trend = 0.0
        self.last_trend_slope_per_day = 0.0

    @staticmethod
    def _datetime_index(values: Any) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(pd.to_datetime(values))

    def _t_idx(self, times: Any) -> np.ndarray:
        if self.origin_ns is None:
            raise RuntimeError("TargetDecomposer has not been fitted yet.")
        ts_ns = self._datetime_index(times).asi8
        return (ts_ns - self.origin_ns) / 86400.0 / 1e9

    def _resolve_periods(self) -> list[int]:
        configured = list(getattr(self.args, "decomposition_periods", []) or [])
        if not configured:
            n_per_day = int(getattr(self.args, "n_per_day", 0) or 0)
            if n_per_day >= 2:
                configured = [n_per_day]
        periods = [int(period) for period in configured]
        if not periods or any(period < 2 for period in periods):
            raise ValueError(
                f"decomposition_method={self.method} requires decomposition_periods >= 2."
            )
        if self.method == "stl" and len(periods) != 1:
            raise ValueError("decomposition_method=stl requires exactly one period.")
        if self.method == "mstl" and len(periods) < 2:
            raise ValueError("decomposition_method=mstl requires at least two periods.")
        return periods

    def _capture_time_axis(self, times: pd.DatetimeIndex) -> None:
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

    def _resolve_trend_lookback(self, n_obs: int) -> int:
        lookback = int(getattr(self.args, "decomposition_trend_lookback", 28) or 28)
        return min(n_obs, max(3, lookback))

    def _fit_trend_forecast(self, trend: np.ndarray, times: pd.DatetimeIndex) -> None:
        degree = int(getattr(self.args, "decomposition_trend_degree", 1) or 1)
        if degree not in {1, 2}:
            raise ValueError("decomposition_trend_degree must be 1 or 2.")
        x = self._t_idx(times)
        self.trend_coefficients = np.polyfit(x, trend, degree)
        self.last_trend = float(trend[-1])
        # 近期斜率在平滑后的趋势分量上按 lookback 估计
        lookback = self._resolve_trend_lookback(len(trend))
        recent_x = x[-lookback:]
        recent_trend = trend[-lookback:]
        self.last_trend_slope_per_day = float(np.polyfit(recent_x, recent_trend, 1)[0])

    def _build_seasonal_templates(self, seasonal: np.ndarray, periods: list[int]) -> None:
        seasonal_2d = np.asarray(seasonal, dtype=float)
        if seasonal_2d.ndim == 1:
            seasonal_2d = seasonal_2d[:, None]
        if seasonal_2d.shape[1] != len(periods):
            raise ValueError("Seasonal component count does not match decomposition_periods.")

        cycles = max(1, int(getattr(self.args, "decomposition_seasonal_cycles", 4) or 4))
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

    def fit(self, df: pd.DataFrame, time_col: str = "time", target_col: str = "y") -> "TargetDecomposer":
        self.time_col = time_col
        self.target_col = target_col
        if not self.enabled:
            return self
        if time_col not in df.columns or target_col not in df.columns:
            raise ValueError(f"Target decomposition requires columns: {time_col}, {target_col}.")

        times = self._datetime_index(df[time_col])
        y = np.asarray(pd.to_numeric(df[target_col], errors="coerce"), dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("Target decomposition does not accept NaN or infinite target values.")
        self._capture_time_axis(times)

        if self.method == "linear":
            degree = int(getattr(self.args, "decomposition_trend_degree", 1) or 1)
            if degree not in {1, 2}:
                raise ValueError("decomposition_trend_degree must be 1 or 2.")
            x = self._t_idx(times)
            self.trend_coefficients = np.polyfit(x, y, degree)
            trend = np.polyval(self.trend_coefficients, x)
            seasonal = np.zeros((len(y), 0), dtype=float)
            periods: list[int] = []
            # 与 STL/MSTL 分支共用趋势外推拟合；damped 近期斜率直接从观测 y 的
            # lookback 段估计——linear 的趋势分量本身是全局多项式，对它重拟合
            # 任何窗口斜率都相同，lookback 不会生效。
            self._fit_trend_forecast(trend, times)
            lookback = self._resolve_trend_lookback(len(y))
            recent_x = x[-lookback:]
            self.last_trend_slope_per_day = float(np.polyfit(recent_x, y[-lookback:], 1)[0])
        else:
            periods = self._resolve_periods()
            if any(2 * period > len(y) for period in periods):
                raise ValueError(
                    f"decomposition_periods={periods} require at least two full cycles "
                    f"in {len(y)} observations."
                )
            robust = bool(getattr(self.args, "decomposition_robust", True))
            if self.method == "stl":
                from statsmodels.tsa.seasonal import STL

                result = STL(y, period=periods[0], robust=robust).fit()
                trend = np.asarray(result.trend, dtype=float)
                seasonal = np.asarray(result.seasonal, dtype=float)[:, None]
            else:
                from statsmodels.tsa.seasonal import MSTL

                result = MSTL(
                    y,
                    periods=tuple(periods),
                    stl_kwargs={"robust": bool(getattr(self.args, "decomposition_robust", True))},
                ).fit()
                trend = np.asarray(result.trend, dtype=float)
                seasonal = np.asarray(result.seasonal, dtype=float)
                if seasonal.ndim == 1:
                    seasonal = seasonal[:, None]
            self._fit_trend_forecast(trend, times)
            self._build_seasonal_templates(seasonal, periods)

        component = trend + (seasonal.sum(axis=1) if seasonal.size else 0.0)
        self.history_component = pd.Series(component, index=times.asi8, dtype=float)
        self.is_fitted = True
        return self

    def _forecast_trend(self, times: pd.DatetimeIndex) -> np.ndarray:
        if self.trend_coefficients is None:
            return np.zeros(len(times), dtype=float)
        trend = np.polyval(self.trend_coefficients, self._t_idx(times))
        method = str(
            getattr(self.args, "decomposition_trend_forecast", "polynomial") or "polynomial"
        ).lower()
        if method == "polynomial":
            return np.asarray(trend, dtype=float)
        if method != "damped":
            raise ValueError(
                "decomposition_trend_forecast must be 'polynomial' or 'damped'."
            )
        if self.last_time_ns is None:
            return np.asarray(trend, dtype=float)
        phi = float(getattr(self.args, "decomposition_damping", 0.98) or 0.98)
        if not 0.0 < phi <= 1.0:
            raise ValueError("decomposition_damping must be in (0, 1].")
        days_ahead = np.maximum(0.0, (times.asi8 - self.last_time_ns) / 86400.0 / 1e9)
        if phi == 1.0:
            increments = days_ahead
        else:
            increments = phi * (1.0 - np.power(phi, days_ahead)) / (1.0 - phi)
        future = self.last_trend + self.last_trend_slope_per_day * increments
        return np.where(days_ahead > 0.0, future, trend)

    def _forecast_seasonal(self, times: pd.DatetimeIndex) -> np.ndarray:
        if not self.seasonal_templates:
            return np.zeros(len(times), dtype=float)
        if self.last_time_ns is None or self.step_ns is None:
            raise RuntimeError("TargetDecomposer has not captured its time axis.")
        positions = self.n_obs - 1 + np.rint(
            (times.asi8 - self.last_time_ns) / self.step_ns
        ).astype(int)
        seasonal = np.zeros(len(times), dtype=float)
        for period, template in self.seasonal_templates:
            seasonal += template[np.mod(positions, period)]
        return seasonal

    def component(self, times: Any) -> np.ndarray:
        if not self.enabled or not self.is_fitted:
            return np.zeros(len(times), dtype=float)
        index = self._datetime_index(times)
        component = self._forecast_trend(index) + self._forecast_seasonal(index)
        if self.history_component is not None:
            exact = self.history_component.reindex(index.asi8)
            known = exact.notna().to_numpy()
            component[known] = exact.to_numpy(dtype=float)[known]
        return component

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled or not self.is_fitted:
            return df.copy()
        out = df.copy()
        out[self.target_col] = (
            out[self.target_col].to_numpy(dtype=float) - self.component(out[self.time_col])
        )
        return out

    def fit_transform(
        self,
        df: pd.DataFrame,
        time_col: str = "time",
        target_col: str = "y",
    ) -> pd.DataFrame:
        return self.fit(df, time_col=time_col, target_col=target_col).transform(df)

    def restore(self, values, times: Any) -> np.ndarray:
        values_arr = np.asarray(values)
        if not self.enabled or not self.is_fitted:
            return values_arr
        return np.asarray(values, dtype=float) + self.component(times)
