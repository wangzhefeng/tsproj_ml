"""原生 ETS：显式有序历史输入，不提供监督回归 fit/predict 接口。"""
from __future__ import annotations

import copy
from collections.abc import Mapping
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel as StatsmodelsETS


class ETSModel:
    """有限加性候选、原生 BIC/AICc 选择；失败证据不隐去。"""

    DEFAULT_PARAMS = {
        "seasonal_periods": 288,
        "candidates": ["ANA", "AAA", "AAdA"],
        "selection": "bic",
        "maxiter": 300,
    }
    CANDIDATES = {
        "ANN": (None, False, None),
        "ANA": (None, False, "add"),
        "AAA": ("add", False, "add"),
        "AAdA": ("add", True, "add"),
    }

    def __init__(self, model_params=None, *, log_prefix="ETS", log_params=False):
        supplied = {} if model_params is None else model_params
        if not isinstance(supplied, Mapping):
            raise TypeError("ETS parameters must be a mapping")
        unknown = set(supplied) - set(self.DEFAULT_PARAMS)
        if unknown:
            raise ValueError(f"unknown ETS parameters: {sorted(unknown)}")
        params = copy.deepcopy(self.DEFAULT_PARAMS)
        params.update(supplied)
        for name in ("seasonal_periods", "maxiter"):
            value = params[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"ETS {name} must be an integer >= 2")
        if params["maxiter"] > 1000:
            raise ValueError("ETS maxiter must be <= 1000")
        candidates = params["candidates"]
        if not isinstance(candidates, (list, tuple)) or not candidates:
            raise ValueError("ETS candidates must be a nonempty sequence")
        if any(not isinstance(c, str) or c not in self.CANDIDATES for c in candidates):
            raise ValueError("unknown ETS candidate")
        if len(set(candidates)) != len(candidates):
            raise ValueError("duplicate ETS candidates")
        if params["selection"] not in ("bic", "aicc"):
            raise ValueError("ETS selection must be bic or aicc")
        params["candidates"] = list(candidates)
        self.params = params
        self.result = None
        self._evidence = None

    def get_params(self):
        return copy.deepcopy(self.params)

    def fit_history(self, history: pd.Series, *, as_of: pd.Timestamp, freq: str):
        self.result = None
        self._evidence = None
        if not isinstance(history, pd.Series) or not isinstance(history.index, pd.DatetimeIndex):
            raise TypeError("ETS requires a one-dimensional Series with DatetimeIndex")
        times = history.index
        if times.empty or times.hasnans or not times.is_unique or not times.is_monotonic_increasing:
            raise ValueError("ETS history must have unique ordered timestamps")
        if times[-1] != pd.Timestamp(as_of):
            raise ValueError("ETS history must end exactly at as_of; future history is forbidden")
        expected = pd.date_range(end=as_of, periods=len(times), freq=freq)
        if not times.equals(expected):
            raise ValueError("ETS requires a complete regular history grid")
        values = history.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("ETS history must be finite")
        period = self.params["seasonal_periods"]
        required = max(2 * period, 10 + 2 * (period // 2)) if any(
            self.CANDIDATES[c][2] is not None for c in self.params["candidates"]
        ) else 10
        if len(values) < required:
            raise ValueError(f"ETS history requires at least {required} raw points")
        # 只估计平滑参数，不把 288 个初始季节状态变成数值优化参数。
        # heuristic 初始化仍完全消费当前 as-of 历史，不来自窗口外数据。
        series = pd.Series(values.copy(), index=expected)
        records = []
        best = None
        best_score = float("inf")
        selected = None
        for name in self.params["candidates"]:
            trend, damped, seasonal = self.CANDIDATES[name]
            record = {"name": name, "error": "add", "trend": trend,
                      "damped_trend": damped, "seasonal": seasonal,
                      "initialization_method": "heuristic", "status": "failed"}
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fitted = StatsmodelsETS(
                        series, error="add", trend=trend, damped_trend=damped,
                        seasonal=seasonal,
                        seasonal_periods=period if seasonal else None,
                        initialization_method="heuristic",
                    ).fit(maxiter=self.params["maxiter"], disp=False)
                record["warnings"] = [str(w.message) for w in caught]
                score = float(getattr(fitted, self.params["selection"]))
                record["score"] = score if np.isfinite(score) else None
                converged = bool(fitted.mle_retvals.get("converged", False))
                record["converged"] = converged
                record["parameters"] = dict(zip(fitted.model.param_names, map(float, fitted.params)))
                if not converged or not np.isfinite(score):
                    record["reason"] = "nonconvergence or nonfinite selection score"
                else:
                    record.update(status="converged", score=score)
                    if score < best_score:
                        best, best_score, selected = fitted, score, name
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                record["reason"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
        self.result = best
        self._evidence = {
            "history_count": len(series), "history_start": times[0].isoformat(),
            "history_end": times[-1].isoformat(), "freq": freq,
            "selection": self.params["selection"], "seasonal_periods": period,
            "selected": selected, "candidates": records,
        }
        if best is None:
            raise ValueError(f"all ETS candidates failed: {records}")
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if self.result is None:
            raise ValueError("ETS is not fitted")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("ETS forecast steps must be a positive integer")
        prediction = np.asarray(self.result.forecast(steps), dtype=float)
        if prediction.shape != (steps,) or not np.isfinite(prediction).all():
            raise ValueError("ETS returned invalid predictions")
        return prediction

    def execution_evidence(self):
        if self._evidence is None:
            raise ValueError("ETS has no fit evidence")
        return copy.deepcopy(self._evidence)
