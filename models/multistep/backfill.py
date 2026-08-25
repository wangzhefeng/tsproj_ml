# -*- coding: utf-8 -*-
"""多变量未来内生变量的严格提供者链。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FutureValue:
    value: float
    source: str


class EndogenousFutureProvider:
    """按特征与 horizon step 返回有限值和来源标签。"""

    def value_at(self, feature: str, step: int) -> FutureValue:
        raise NotImplementedError


class PersistenceProvider(EndogenousFutureProvider):
    def __init__(self, history: pd.DataFrame, features: Sequence[str]):
        self._values: dict[str, float] = {}
        missing = [feature for feature in features if feature not in history.columns]
        if missing:
            raise ValueError(f"Persistence backfill missing history columns: {missing}.")
        for feature in features:
            values = pd.Series(
                pd.to_numeric(history[feature], errors="coerce"),
                index=history.index,
            ).dropna()
            if values.empty or not np.isfinite(float(values.iloc[-1])):
                raise ValueError(
                    f"Persistence backfill requires a finite last value for '{feature}'."
                )
            self._values[feature] = float(values.iloc[-1])

    def value_at(self, feature: str, step: int) -> FutureValue:
        if feature not in self._values:
            raise KeyError(f"Persistence backfill has no feature '{feature}'.")
        if int(step) < 0:
            raise ValueError("Backfill step must be >= 0.")
        return FutureValue(self._values[feature], "persistence")


class AuxiliaryProvider(EndogenousFutureProvider):
    def __init__(
        self,
        trajectories: Mapping[str, Sequence[float]],
        features: Sequence[str],
        horizon: int,
    ):
        self._values: dict[str, np.ndarray] = {}
        for feature in features:
            if feature not in trajectories:
                raise ValueError(f"Auxiliary trajectory missing feature '{feature}'.")
            values = np.asarray(trajectories[feature], dtype=float).reshape(-1)
            if values.shape != (int(horizon),):
                raise ValueError(
                    f"Auxiliary trajectory '{feature}' length mismatch: "
                    f"expected {horizon}, got {len(values)}."
                )
            if not np.isfinite(values).all():
                raise ValueError(f"Auxiliary trajectory '{feature}' contains non-finite values.")
            self._values[feature] = values

    def value_at(self, feature: str, step: int) -> FutureValue:
        if feature not in self._values:
            raise KeyError(f"Auxiliary backfill has no feature '{feature}'.")
        index = int(step)
        if index < 0 or index >= len(self._values[feature]):
            raise IndexError(
                f"Auxiliary trajectory '{feature}' has no step {index}."
            )
        return FutureValue(float(self._values[feature][index]), "auxiliary")


class KnownFutureProvider(EndogenousFutureProvider):
    """优先读取显式未来值，缺失时调用已配置的严格 provider。"""

    def __init__(
        self,
        future: pd.DataFrame,
        fallback: EndogenousFutureProvider,
        features: Sequence[str],
        horizon: int,
    ):
        if len(future) != int(horizon):
            raise ValueError(
                f"Future endogenous frame length mismatch: expected {horizon}, got {len(future)}."
            )
        self._future = future.reset_index(drop=True)
        self._fallback = fallback
        self._features = set(features)

    def value_at(self, feature: str, step: int) -> FutureValue:
        if feature not in self._features:
            raise KeyError(f"Unknown endogenous feature '{feature}'.")
        index = int(step)
        if index < 0 or index >= len(self._future):
            raise IndexError(f"Future endogenous frame has no step {index}.")
        if feature in self._future.columns:
            raw_value = self._future.at[index, feature]
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = float("nan")
            if np.isfinite(value):
                return FutureValue(value, "known_future")
        return self._fallback.value_at(feature, index)


def build_endogenous_future_provider(context) -> EndogenousFutureProvider | None:
    features = [
        feature
        for feature in context.endogenous_features
        if feature != context.target_feature
    ]
    if not features:
        return None
    strategy = str(
        getattr(context.args, "endogenous_backfill_strategy", "persistence")
        or "persistence"
    ).lower()
    if strategy == "persistence":
        fallback: EndogenousFutureProvider = PersistenceProvider(
            context.df_history_for_lags,
            features,
        )
    elif strategy == "auxiliary":
        if context.aux_trajectories is None:
            raise ValueError(
                "endogenous_backfill_strategy=auxiliary requires fitted auxiliary trajectories."
            )
        fallback = AuxiliaryProvider(context.aux_trajectories, features, context.horizon)
    else:
        raise ValueError(f"Unsupported endogenous_backfill_strategy='{strategy}'.")
    return KnownFutureProvider(context.df_future, fallback, features, context.horizon)
