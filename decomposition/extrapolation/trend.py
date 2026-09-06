"""多项式/阻尼趋势拟合与外推。"""
import numpy as np
from decomposition.contracts.interfaces import ComponentForecaster
from decomposition.contracts.time_axis import elapsed_days


class TrendForecaster(ComponentForecaster):
    def __init__(self, *, degree: int, mode: str, damping: float, lookback: int, source: str):
        self.degree = degree
        self.mode = mode
        self.damping = damping
        self.lookback = lookback
        self.source = source
        self.is_fitted = False

    def fit(self, frame):
        self.is_fitted = False
        self.origin_ns = int(frame.times.asi8[0])
        self.last_time_ns = int(frame.times.asi8[-1])
        x = elapsed_days(frame.times, self.origin_ns)
        values = frame.target if self.source == "target" else frame.trend
        self.coefficients = np.polyfit(x, values, self.degree)
        self.last_trend = float(frame.trend[-1])
        lookback = min(len(values), max(3, self.lookback))
        self.slope = float(np.polyfit(x[-lookback:], values[-lookback:], 1)[0])
        self.is_fitted = True

    def predict(self, future_times):
        if not self.is_fitted:
            raise RuntimeError("Trend forecaster must be fitted before predict.")
        trend = np.polyval(self.coefficients, elapsed_days(future_times, self.origin_ns))
        if self.mode == "polynomial":
            return np.asarray(trend, dtype=float)
        days_ahead = np.maximum(0.0, elapsed_days(future_times, self.last_time_ns))
        phi = self.damping
        increments = days_ahead if phi == 1.0 else phi * (1.0 - np.power(phi, days_ahead)) / (1.0 - phi)
        future = self.last_trend + self.slope * increments
        return np.where(days_ahead > 0.0, future, trend)
