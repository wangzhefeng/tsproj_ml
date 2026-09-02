# -*- coding: utf-8 -*-
"""Canonical 特征缩放器（2026-09-01 自 model_forecasting/transforms.py 归位）。

特征缩放是特征工程的实现环节，归位到 feature_engineering/ 阶段包；
纯机械搬移，行为与 fingerprint 零变化。旧导入路径经 re-export 保持兼容。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from feature_engineering.transform_specs import normalize_feature_scaling


class CanonicalFeatureScaler:
    """Fit-once feature preprocessing for canonical runtime matrices."""

    def __init__(
        self,
        spec: Mapping[str, Any],
        *,
        feature_names: Sequence[str],
        categorical_names: Sequence[str] = (),
        training_scope: str = "local",
        category_orders: Mapping[str, Sequence[Any]] | None = None,
    ) -> None:
        normalized = normalize_feature_scaling(spec)
        self.method = normalized["method"]
        self.grouped = normalized["grouped"]
        self.encode_categorical = normalized["encode_categorical"]
        self.feature_names = tuple(str(name) for name in feature_names)
        self.categorical_names = tuple(str(name) for name in categorical_names)
        unknown_categorical = set(self.categorical_names) - set(self.feature_names)
        if unknown_categorical:
            raise ValueError(
                f"categorical features are absent from feature schema: {sorted(unknown_categorical)}"
            )
        self.training_scope = str(training_scope)
        self.category_orders = {
            str(name): tuple(values)
            for name, values in dict(category_orders or {}).items()
        }
        self.category_mappings: dict[str, dict[Any, int]] = {}
        self.feature_groups = self._identify_groups()
        self.scalers: dict[str, Any] = {}
        self.is_fitted = False

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        feature_names: Sequence[str],
        categorical_names: Sequence[str] = (),
        category_orders: Mapping[str, Sequence[Any]] | None = None,
    ) -> "CanonicalFeatureScaler":
        return cls(
            config.features.transformations.get("feature_scaling", {}),
            feature_names=feature_names,
            categorical_names=categorical_names,
            training_scope=config.problem.training_scope,
            category_orders=category_orders,
        )

    def _identify_groups(self) -> dict[str, tuple[str, ...]]:
        categorical = set(self.categorical_names)
        lag = tuple(
            name for name in self.feature_names if name not in categorical and "_lag_" in name
        )
        datetime = tuple(
            name for name in self.feature_names if name not in categorical and name.startswith("dt_")
        )
        weather_keywords = ("temp", "humidity", "wind", "rain", "pressure", "weather")
        weather = tuple(
            name
            for name in self.feature_names
            if name not in categorical
            and name not in lag
            and name not in datetime
            and any(keyword in name.lower() for keyword in weather_keywords)
        )
        assigned = set(lag) | set(datetime) | set(weather) | categorical
        other = tuple(name for name in self.feature_names if name not in assigned)
        return {
            "lag": lag,
            "datetime": datetime,
            "weather": weather,
            "other": other,
            "categorical": tuple(
                name for name in self.feature_names if name in categorical
            ),
        }

    def _frame(self, values: Any) -> pd.DataFrame:
        if isinstance(values, pd.DataFrame):
            frame = values.copy()
        else:
            array = np.asarray(values)
            if array.ndim != 2:
                raise ValueError(f"feature matrix must be two-dimensional; got {array.shape}")
            if array.shape[1] != len(self.feature_names):
                raise ValueError("feature matrix width must match feature schema")
            frame = pd.DataFrame(array, columns=self.feature_names)
        missing = [name for name in self.feature_names if name not in frame.columns]
        extra = [name for name in frame.columns if name not in self.feature_names]
        if missing or extra:
            raise ValueError(
                f"feature schema mismatch: missing={missing}, extra={extra}"
            )
        return frame.loc[:, self.feature_names].copy()

    @staticmethod
    def _new_scaler(method: str):
        if method == "standard":
            return StandardScaler()
        if method == "minmax":
            return MinMaxScaler()
        if method == "robust":
            return RobustScaler()
        raise ValueError(f"unsupported feature scaling method: {method}")

    def _encode_fit(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for name in self.categorical_names:
            if self.encode_categorical:
                categories = list(
                    self.category_orders.get(name, tuple(pd.unique(result[name])))
                )
                observed = set(pd.unique(result[name]))
                unknown = [value for value in observed if value not in categories]
                if unknown:
                    raise ValueError(
                        f"unknown categorical values for feature {name!r}: {unknown}"
                    )
                mapping = {value: index for index, value in enumerate(categories)}
                self.category_mappings[name] = mapping
                result[name] = result[name].map(mapping)
        return result

    def _encode_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for name in self.categorical_names:
            if not self.encode_categorical:
                continue
            mapping = self.category_mappings[name]
            unknown = [value for value in pd.unique(result[name]) if value not in mapping]
            if unknown:
                raise ValueError(
                    f"unknown categorical values for feature {name!r}: {unknown}"
                )
            result[name] = result[name].map(mapping)
        return result

    def _numeric_groups(self) -> dict[str, tuple[str, ...]]:
        if self.grouped:
            return {
                name: columns
                for name, columns in self.feature_groups.items()
                if name != "categorical" and columns
            }
        numeric = tuple(
            name for name in self.feature_names if name not in self.categorical_names
        )
        return {"all": numeric} if numeric else {}

    def fit_transform(self, values: Any) -> np.ndarray:
        frame = self._encode_fit(self._frame(values))
        self.scalers = {}
        if self.method != "none":
            for group_name, columns in self._numeric_groups().items():
                scaler = self._new_scaler(self.method)
                frame.loc[:, list(columns)] = scaler.fit_transform(
                    frame.loc[:, list(columns)].astype(float)
                )
                self.scalers[group_name] = scaler
        self.is_fitted = True
        return self._as_float_array(frame)

    def fit_transform_calls(
        self,
        calls: Sequence[Any],
    ) -> tuple[np.ndarray, ...]:
        frames = tuple(self._frame(call) for call in calls)
        lengths = tuple(len(frame) for frame in frames)
        combined = pd.concat(frames, ignore_index=True)
        transformed = self.fit_transform(combined)
        result = []
        start = 0
        for length in lengths:
            result.append(transformed[start : start + length])
            start += length
        return tuple(result)

    def transform(self, values: Any) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("feature scaler must be fitted before transform")
        frame = self._encode_transform(self._frame(values))
        if self.method != "none":
            for group_name, columns in self._numeric_groups().items():
                frame.loc[:, list(columns)] = self.scalers[group_name].transform(
                    frame.loc[:, list(columns)].astype(float)
                )
        return self._as_float_array(frame)

    @staticmethod
    def _as_float_array(frame: pd.DataFrame) -> np.ndarray:
        try:
            array = frame.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "canonical features must be numeric or enable categorical encoding"
            ) from exc
        if not np.isfinite(array).all():
            raise ValueError("canonical transformed features must be finite")
        return array


__all__ = ["CanonicalFeatureScaler"]