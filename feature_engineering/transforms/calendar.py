"""目标变换职责拆分；旧持久化类路径不提供兼容别名。"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd


class CalendarDayTargetNormalizer:
    """按时间戳所在月份天数归一化月总量。

    ``per_calendar_day`` 模式把月总量除以当月天数后送入模型；输出按目标月
    天数乘回原始月总量空间。该变换无拟合状态，不接触测试标签。
    """

    SUPPORTED = {"none", "per_calendar_day"}

    def __init__(self, method: str):
        self.method = method
        if self.method not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported target_calendar_normalization='{self.method}'; "
                f"expected one of {sorted(self.SUPPORTED)}."
            )

    @property
    def enabled(self) -> bool:
        return self.method == "per_calendar_day"

    @staticmethod
    def _factors(times) -> np.ndarray:
        ts = pd.Series(pd.to_datetime(times))
        return ts.dt.days_in_month.to_numpy(dtype=float)

    def transform_history(self, df: pd.DataFrame, time_col: str, target_col: str) -> pd.DataFrame:
        result = df.copy()
        if not self.enabled:
            return result
        if time_col not in result.columns or target_col not in result.columns:
            raise ValueError(
                f"Calendar-day target normalization requires columns '{time_col}' and '{target_col}'."
            )
        factors = self._factors(result[time_col])
        result[target_col] = pd.to_numeric(result[target_col], errors="coerce") / factors
        return result

    def restore_predictions(self, predictions, times) -> np.ndarray:
        values = np.asarray(predictions, dtype=float).reshape(-1)
        if not self.enabled:
            return values
        factors = self._factors(times)
        if len(values) != len(factors):
            raise ValueError(
                f"Prediction length ({len(values)}) does not match timestamp length ({len(factors)}) "
                "for calendar-day restoration."
            )
        return values * factors
