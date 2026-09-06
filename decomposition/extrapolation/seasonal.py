"""按最近若干完整相位样本拟合季节模板。"""
import numpy as np
from decomposition.contracts.interfaces import ComponentForecaster
from decomposition.contracts.time_axis import validate_regular_times


class PhaseTemplateForecaster(ComponentForecaster):
    def __init__(self, cycles: int):
        self.cycles = cycles
        self.is_fitted = False

    def fit(self, frame):
        self.is_fitted = False
        self.step_ns = validate_regular_times(frame.times)
        self.last_time_ns = int(frame.times.asi8[-1])
        self.n_obs = len(frame.times)
        positions = np.arange(self.n_obs)
        self.templates = []
        for period, component in frame.seasonal.items():
            start = max(0, self.n_obs - self.cycles * period)
            template = np.empty(period, dtype=float)
            for phase in range(period):
                values = component[(positions >= start) & (positions % period == phase)]
                if not len(values):
                    raise ValueError("Seasonal template requires every phase to be observed.")
                template[phase] = float(np.mean(values))
            self.templates.append((period, template))
        self.is_fitted = True

    def predict(self, future_times):
        if not self.is_fitted:
            raise RuntimeError("Seasonal forecaster must be fitted before predict.")
        positions = self.n_obs - 1 + np.rint((future_times.asi8 - self.last_time_ns) / self.step_ns).astype(int)
        seasonal = np.zeros(len(future_times), dtype=float)
        for period, template in self.templates:
            seasonal += template[np.mod(positions, period)]
        return seasonal
