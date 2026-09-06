"""分量外推参数的数值合同；参照直接来自 statsmodels 和显式公式。"""
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL, STL

from decomposition import build_pipeline, resolve_decomposition_spec


class SeasonalParameterTest(unittest.TestCase):
    def test_nondefault_parameters_reach_fitted_forecast(self):
        x = np.arange(480, dtype=float)
        times = pd.date_range("2026-01-01", periods=len(x), freq="h")
        y = 100 + 0.003 * x ** 2 + (3 + 0.02 * x) * np.sin(2 * np.pi * x / 24)
        y += np.sin(2 * np.pi * x / 72)
        frame = pd.DataFrame({"time": times, "y": y})
        future = pd.date_range(times[-1] + pd.Timedelta(hours=1), periods=48, freq="h")
        for method, periods in (("stl", [24]), ("mstl", [24, 72])):
            result = (STL(y, period=24, robust=True) if method == "stl" else
                      MSTL(y, periods=periods, stl_kwargs={"robust": True})).fit()
            trend = np.asarray(result.trend)
            seasonal = np.asarray(result.seasonal).reshape(len(y), -1)
            for overrides in ({"trend_degree": 2}, {"seasonal_cycles": 1},
                              {"trend_forecast": "damped", "trend_lookback": 5}):
                with self.subTest(method=method, overrides=overrides):
                    raw = {"method": method, "periods": periods, **overrides}
                    pipe = build_pipeline(resolve_decomposition_spec(SimpleNamespace(decomposition=raw))).fit(frame)
                    days = (times.asi8 - times.asi8[0]) / 86400.0 / 1e9
                    future_days = (future.asi8 - times.asi8[0]) / 86400.0 / 1e9
                    expected = np.polyval(np.polyfit(days, trend, raw.get("trend_degree", 1)), future_days)
                    if raw.get("trend_forecast") == "damped":
                        lookback = raw["trend_lookback"]
                        slope = np.polyfit(days[-lookback:], trend[-lookback:], 1)[0]
                        ahead = (future.asi8 - times.asi8[-1]) / 86400.0 / 1e9
                        expected = trend[-1] + slope * 0.98 * (1 - 0.98 ** ahead) / (1 - 0.98)
                    positions = np.arange(len(y))
                    for index, period in enumerate(periods):
                        start = max(0, len(y) - raw.get("seasonal_cycles", 4) * period)
                        template = np.array([seasonal[(positions >= start) & (positions % period == phase), index].mean()
                                             for phase in range(period)])
                        expected += template[np.arange(len(y), len(y) + len(future)) % period]
                    np.testing.assert_allclose(pipe.restore(np.zeros(len(future)), future), expected, rtol=1e-12, atol=1e-10)
                    residual = pipe.transform(frame)["y"].to_numpy()
                    np.testing.assert_allclose(residual, y - (trend + seasonal.sum(axis=1)), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
