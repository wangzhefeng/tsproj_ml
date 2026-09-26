"""原生 Naive 基线族：显式有序历史输入，不提供监督回归 fit/predict 接口。

三种模式（M3/M4 竞赛标准基线，纯 numpy 闭式计算）：
- naive           ặ_{n+h} = y_n（最后观测值平推）
- drift           ặ_{n+h} = y_n + h·(y_n - y_1)/(n-1)（随机游走加漂移）
- seasonal_naive  ặ_{n+h} = y_{n+h-m·k}（m 为季节周期，k=⌈h/m⌉）

与 ETS 相同的严格输入合同：规则时间网格、末端恰为 as_of、有限值、
未知参数 RAISE；fit_history / forecast / execution_evidence 三件套
与 ETSModel 逐字段对齐，供 pipeline 注册表统一分发。
"""
from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np
import pandas as pd

_MODES = ("naive", "drift", "seasonal_naive")


class NaiveModel:
    """闭式基线；无估计参数，不存在不收敛问题。"""

    DEFAULT_PARAMS = {
        "mode": "naive",
        "seasonal_periods": 288,
    }

    def __init__(self, model_params=None, *, log_prefix="Naive", log_params=False):
        supplied = {} if model_params is None else model_params
        if not isinstance(supplied, Mapping):
            raise TypeError("Naive parameters must be a mapping")
        unknown = set(supplied) - set(self.DEFAULT_PARAMS)
        if unknown:
            raise ValueError(f"unknown Naive parameters: {sorted(unknown)}")
        params = copy.deepcopy(self.DEFAULT_PARAMS)
        params.update(supplied)
        mode = str(params["mode"]).lower()
        if mode not in _MODES:
            raise ValueError(f"unknown Naive mode: {params['mode']!r}; expected one of {_MODES}")
        period = params["seasonal_periods"]
        if isinstance(period, bool) or not isinstance(period, int) or period < 2:
            raise ValueError("Naive seasonal_periods must be an integer >= 2")
        params["mode"] = mode
        self.params = params
        self._history = None
        self._evidence = None

    def get_params(self):
        return copy.deepcopy(self.params)

    def fit_history(self, history: pd.Series, *, as_of: pd.Timestamp, freq: str):
        self._history = None
        self._evidence = None
        if not isinstance(history, pd.Series) or not isinstance(history.index, pd.DatetimeIndex):
            raise TypeError("Naive requires a one-dimensional Series with DatetimeIndex")
        times = history.index
        if times.empty or times.hasnans or not times.is_unique or not times.is_monotonic_increasing:
            raise ValueError("Naive history must have unique ordered timestamps")
        if times[-1] != pd.Timestamp(as_of):
            raise ValueError("Naive history must end exactly at as_of; future history is forbidden")
        expected = pd.date_range(end=as_of, periods=len(times), freq=freq)
        if not times.equals(expected):
            raise ValueError("Naive requires a complete regular history grid")
        values = history.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Naive history must be finite")
        if len(values) < 2:
            raise ValueError("Naive history requires at least 2 raw points")
        if self.params["mode"] == "seasonal_naive" and len(values) < self.params["seasonal_periods"]:
            raise ValueError(
                f"seasonal_naive requires history length >= seasonal_periods "
                f"({self.params['seasonal_periods']})"
            )
        self._history = values.copy()
        self._evidence = {
            "history_count": len(values), "history_start": times[0].isoformat(),
            "history_end": times[-1].isoformat(), "freq": freq,
            "mode": self.params["mode"],
            "seasonal_periods": self.params["seasonal_periods"],
        }
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if self._history is None:
            raise ValueError("Naive is not fitted")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("Naive forecast steps must be a positive integer")
        values = self._history
        horizons = np.arange(1, steps + 1, dtype=float)
        mode = self.params["mode"]
        if mode == "naive":
            prediction = np.full(steps, values[-1], dtype=float)
        elif mode == "drift":
            slope = (values[-1] - values[0]) / (len(values) - 1)
            prediction = values[-1] + horizons * slope
        else:
            period = self.params["seasonal_periods"]
            # h 的季节索引：h - m·ceil(h/m)，与 statsmodels seasonal naive 一致
            seasonal_index = (horizons - period * np.ceil(horizons / period)).astype(int)
            prediction = values[seasonal_index - 1]
        if prediction.shape != (steps,) or not np.isfinite(prediction).all():
            raise ValueError("Naive returned invalid predictions")
        return prediction

    def execution_evidence(self):
        if self._evidence is None:
            raise ValueError("Naive has no fit evidence")
        return copy.deepcopy(self._evidence)
