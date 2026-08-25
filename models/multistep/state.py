# -*- coding: utf-8 -*-
"""递归预测调用私有状态。"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd


@dataclass
class RecursiveState:
    history: pd.DataFrame
    max_length: int

    @classmethod
    def from_history(cls, history: pd.DataFrame, max_length: int) -> "RecursiveState":
        max_length = max(1, int(max_length))
        return cls(history=history.iloc[-max_length:].copy(deep=True).reset_index(drop=True), max_length=max_length)

    def append(self, row: pd.DataFrame) -> None:
        if row is None or row.empty:
            raise ValueError("recursive state append requires one row.")
        aligned = row.iloc[-1:].copy().reindex(columns=self.history.columns)
        if aligned.isna().any(axis=None):
            missing = aligned.columns[aligned.isna().any()].tolist()
            raise ValueError(f"recursive state row missing values for columns: {missing}.")
        self.history = pd.concat([self.history, aligned], ignore_index=True).iloc[-self.max_length:].reset_index(drop=True)


class CacheProtocol(Protocol):
    """递归特征缓存的最小执行器契约。"""

    def build_step_input(self, step: int) -> pd.DataFrame: ...

    def append_endogenous(self, feature: str, value: float) -> None: ...


@dataclass
class RecursiveFeatureCache:
    """MSMR/MSMDR 共享的调用私有缓存；不写回 Forecaster 公共历史。"""

    df_future_exog: pd.DataFrame
    exogenous_features: tuple[str, ...]
    predictor_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    target_output_features: tuple[str, ...]
    lag_feature_names: frozenset[str]
    lag_state: dict[str, deque]
    lags: tuple[int, ...]
    schema_build_count: int = 1

    def __getitem__(self, key: str) -> Any:
        """短期兼容旧执行器的字典访问；Phase 6 清理调用点。"""
        return getattr(self, key)

    @staticmethod
    def _read_lag_value(buffer: deque, lag: int) -> float:
        values = list(buffer)
        if not values:
            raise ValueError("recursive lag state is empty.")
        if int(lag) < 1:
            raise ValueError("recursive lag must be >= 1.")
        value = values[-lag] if lag <= len(values) else values[0]
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("recursive lag state contains a non-finite value.")
        return result

    def build_step_input(self, step: int) -> pd.DataFrame:
        index = int(step)
        if index < 0 or index >= len(self.df_future_exog):
            raise IndexError(f"recursive feature cache has no future step {index}.")
        row_data: dict[str, Any] = {}
        if not self.df_future_exog.empty:
            future_row = self.df_future_exog.iloc[index]
            for feature in self.exogenous_features:
                if feature not in future_row.index:
                    raise ValueError(
                        f"recursive feature cache missing exogenous feature '{feature}'."
                    )
                row_data[feature] = future_row[feature]
        for feature, buffer in self.lag_state.items():
            for lag in self.lags:
                lag_feature = f"{feature}_lag_{lag}"
                if lag_feature in self.lag_feature_names:
                    row_data[lag_feature] = self._read_lag_value(buffer, lag)
        frame = pd.DataFrame([row_data]).reindex(columns=self.predictor_features)
        missing = frame.columns[frame.isna().any()].tolist()
        if missing:
            raise ValueError(
                f"recursive feature cache could not construct features: {missing}."
            )
        return frame

    def append_endogenous(self, feature: str, value: float) -> None:
        if feature not in self.lag_state:
            raise KeyError(f"recursive feature cache has no endogenous feature '{feature}'.")
        resolved = float(value)
        if not np.isfinite(resolved):
            raise ValueError(
                f"recursive feature cache cannot append non-finite value for '{feature}'."
            )
        self.lag_state[feature].append(resolved)
