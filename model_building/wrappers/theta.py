"""原生 Theta：显式有序历史输入，不提供监督回归 fit/predict 接口。

经典 Theta 方法（Assimakopoulos & Nikolopoulos 2000；Hyndman & Billah 2003
证明 theta=2 且带漂移的形态等价于带漂移简单指数平滑）。statsmodels 实现
只暴露单一 theta 线组合，本封装按 ETS 同款严格合同消费：
规则时间网格、末端恰为 as_of、有限值、未知参数 RAISE；估计失败直接
RAISE，不静默降级。``method`` 透传季节性 Oswal/检测显著性检验方式。
"""
from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
from statsmodels.tsa.forecasting.theta import ThetaModel as StatsmodelsTheta

_SEASONAL_METHODS = ("auto", "additive", "multiplicative", "none")


class ThetaModel:
    """单一 theta 线组合；加性季节去季节化，失败证据不隐去。"""

    DEFAULT_PARAMS = {
        "seasonal_periods": 288,
        "deseasonalize": True,
        "use_test": True,
        "method": "auto",
        "difference": False,
        "theta": 2.0,
        "use_mle": False,
    }

    def __init__(self, model_params=None, *, log_prefix="Theta", log_params=False):
        supplied = {} if model_params is None else model_params
        if not isinstance(supplied, Mapping):
            raise TypeError("Theta parameters must be a mapping")
        unknown = set(supplied) - set(self.DEFAULT_PARAMS)
        if unknown:
            raise ValueError(f"unknown Theta parameters: {sorted(unknown)}")
        params = copy.deepcopy(self.DEFAULT_PARAMS)
        params.update(supplied)
        period = params["seasonal_periods"]
        if isinstance(period, bool) or not isinstance(period, int) or period < 2:
            raise ValueError("Theta seasonal_periods must be an integer >= 2")
        for name in ("deseasonalize", "use_test", "difference", "use_mle"):
            if not isinstance(params[name], bool):
                raise ValueError(f"Theta {name} must be a bool")
        method = str(params["method"]).lower()
        if method not in _SEASONAL_METHODS:
            raise ValueError(f"unknown Theta method: {params['method']!r}")
        params["method"] = method
        theta = params["theta"]
        if isinstance(theta, bool) or not isinstance(theta, (int, float)):
            raise TypeError("Theta theta must be a real number")
        theta = float(theta)
        if not np.isfinite(theta):
            raise ValueError("Theta theta must be finite")
        if not theta >= 1.0:
            raise ValueError("Theta theta must be >= 1 (theta < 1 是去趋势线，不用于预测)")
        params["theta"] = theta
        self.params = params
        self.result = None
        self._evidence = None

    def get_params(self):
        return copy.deepcopy(self.params)

    def fit_history(self, history: pd.Series, *, as_of: pd.Timestamp, freq: str):
        self.result = None
        self._evidence = None
        if not isinstance(history, pd.Series) or not isinstance(history.index, pd.DatetimeIndex):
            raise TypeError("Theta requires a one-dimensional Series with DatetimeIndex")
        times = history.index
        if times.empty or times.hasnans or not times.is_unique or not times.is_monotonic_increasing:
            raise ValueError("Theta history must have unique ordered timestamps")
        if times[-1] != pd.Timestamp(as_of):
            raise ValueError("Theta history must end exactly at as_of; future history is forbidden")
        expected = pd.date_range(end=as_of, periods=len(times), freq=freq)
        if not times.equals(expected):
            raise ValueError("Theta requires a complete regular history grid")
        values = history.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Theta history must be finite")
        period = self.params["seasonal_periods"]
        if self.params["deseasonalize"]:
            # 季节性检验按周期两倍数据起步（statsmodels 内部 multiplicative 分解要求正值）
            required = max(2 * period, 2 * period + 2)
            if len(values) < required:
                raise ValueError(f"Theta history requires at least {required} raw points")
            if self.params["method"] == "multiplicative" and np.any(values <= 0):
                raise ValueError("Theta multiplicative seasonal decomposition requires positive history")
        elif len(values) < 3:
            raise ValueError("Theta history requires at least 3 raw points")
        series = pd.Series(values.copy(), index=expected)
        record = {
            "seasonal_periods": period,
            "deseasonalize": self.params["deseasonalize"],
            "use_test": self.params["use_test"],
            "method": self.params["method"],
            "difference": self.params["difference"],
            "theta": self.params["theta"],
            "use_mle": self.params["use_mle"],
            "status": "failed",
        }
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator = StatsmodelsTheta(
                    series,
                    period=period if self.params["deseasonalize"] else None,
                    deseasonalize=self.params["deseasonalize"],
                    use_test=self.params["use_test"],
                    method=self.params["method"],
                    difference=self.params["difference"],
                )
                fitted = estimator.fit(use_mle=self.params["use_mle"], disp=False)
                # 以 theta 参数实际出预测，验证可预测性（不在 fit 期偷看未来）
                probe = np.asarray(fitted.forecast(1, theta=self.params["theta"]), dtype=float)
            record["warnings"] = [str(w.message) for w in caught]
            if probe.shape != (1,) or not np.isfinite(probe).all():
                record["reason"] = "nonfinite one-step probe forecast"
            else:
                record.update(status="converged")
                record["parameters"] = {
                    name: float(value) for name, value in fitted.params.items()
                }
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            record["reason"] = f"{type(exc).__name__}: {exc}"
            self._evidence = {
                "history_count": len(series), "history_start": times[0].isoformat(),
                "history_end": times[-1].isoformat(), "freq": freq,
                "candidates": [record],
            }
            raise ValueError(f"Theta fit failed: {record['reason']}") from exc
        self.result = fitted
        self._evidence = {
            "history_count": len(series), "history_start": times[0].isoformat(),
            "history_end": times[-1].isoformat(), "freq": freq,
            "seasonal_periods": period,
            "selected": "theta",
            "candidates": [record],
        }
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if self.result is None:
            raise ValueError("Theta is not fitted")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("Theta forecast steps must be a positive integer")
        prediction = np.asarray(
            self.result.forecast(steps, theta=self.params["theta"]), dtype=float
        )
        if prediction.shape != (steps,) or not np.isfinite(prediction).all():
            raise ValueError("Theta returned invalid predictions")
        return prediction

    def execution_evidence(self):
        if self._evidence is None:
            raise ValueError("Theta has no fit evidence")
        return copy.deepcopy(self._evidence)
