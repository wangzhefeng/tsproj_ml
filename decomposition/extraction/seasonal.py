"""STL/MSTL 原始历史分量，不包含趋势外推或相位模板。"""
import numpy as np
from statsmodels.tsa.seasonal import MSTL, STL
from decomposition.contracts.interfaces import ComponentExtractor
from decomposition.contracts.time_axis import read_history
from decomposition.contracts.types import ComponentFrame


class STLExtractor(ComponentExtractor):
    def __init__(self, periods: tuple[int, ...], robust: bool):
        self.periods = periods
        self.robust = robust

    def decompose(self, y):
        return STL(y, period=self.periods[0], robust=self.robust).fit()

    def fit_transform(self, history, time_col="time", target_col="y"):
        times, y = read_history(history, time_col, target_col)
        if any(2 * period > len(y) for period in self.periods):
            raise ValueError(f"periods={self.periods} require at least two full cycles in {len(y)} observations.")
        result = self.decompose(y)
        seasonal = np.asarray(result.seasonal, dtype=float).reshape(len(y), -1)
        if seasonal.shape[1] != len(self.periods):
            raise ValueError("Seasonal component count does not match decomposition periods.")
        return ComponentFrame(times, y, np.asarray(result.trend, dtype=float),
                              {period: seasonal[:, i] for i, period in enumerate(self.periods)})


class MSTLExtractor(STLExtractor):
    def decompose(self, y):
        # statsmodels 会剔除 period >= n/2 的周期；明确拒绝而不是静默降级。
        if any(2 * period >= len(y) for period in self.periods):
            raise ValueError("MSTL periods require more than two full cycles.")
        return MSTL(y, periods=self.periods, stl_kwargs={"robust": self.robust}).fit()
