"""多项式历史趋势提取。"""
import numpy as np
from decomposition.contracts.interfaces import ComponentExtractor
from decomposition.contracts.time_axis import elapsed_days, read_history
from decomposition.contracts.types import ComponentFrame


class PolyTrendExtractor(ComponentExtractor):
    def __init__(self, degree: int):
        self.degree = degree

    def fit_transform(self, history, time_col="time", target_col="y"):
        times, y = read_history(history, time_col, target_col)
        x = elapsed_days(times, int(times.asi8[0]))
        trend = np.polyval(np.polyfit(x, y, self.degree), x)
        return ComponentFrame(times, y, trend, {})
