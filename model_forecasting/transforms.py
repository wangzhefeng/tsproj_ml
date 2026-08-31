"""Canonical target and feature transformations consumed by the canonical runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.base import clone

from decomposition import DecompositionPipeline, resolve_decomposition_spec
from forecasting_core.tensors import (
    MarginalQuantileForecastTensor,
    PointForecastTensor,
)
from forecasting_core.artifacts import MarginalForecastDistribution


_FEATURE_SCALING_METHODS = frozenset({"none", "minmax", "standard", "robust"})
_CALENDAR_METHODS = frozenset({"none", "per_calendar_day"})
_DECOMPOSITION_METHODS = frozenset(
    {"none", "linear", "quadratic", "damped", "stl", "mstl"}
)
_TARGET_SCALING_METHODS = _FEATURE_SCALING_METHODS


def normalize_feature_scaling(value: Any) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError("transformations.feature_scaling must be a mapping")
    unknown = set(value) - {"method", "grouped", "encode_categorical"}
    if unknown:
        raise ValueError(
            f"unknown transformations.feature_scaling keys: {sorted(unknown)}"
        )
    method = str(value.get("method", "none") or "none").lower()
    if method not in _FEATURE_SCALING_METHODS:
        raise ValueError(
            f"unsupported feature scaling method {method!r}; "
            f"expected one of {sorted(_FEATURE_SCALING_METHODS)}"
        )
    grouped = value.get("grouped", False)
    encode_categorical = value.get("encode_categorical", False)
    if not isinstance(grouped, bool):
        raise TypeError("transformations.feature_scaling.grouped must be bool")
    if not isinstance(encode_categorical, bool):
        raise TypeError(
            "transformations.feature_scaling.encode_categorical must be bool"
        )
    return {
        "method": method,
        "grouped": grouped,
        "encode_categorical": encode_categorical,
    }


def normalize_target_transformations(value: Any) -> dict[str, dict[str, Any]]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError("transformations.target must be a mapping")
    unknown = set(value) - {"calendar_normalization", "decomposition", "scaling"}
    if unknown:
        raise ValueError(f"unknown transformations.target keys: {sorted(unknown)}")

    calendar = value.get("calendar_normalization", {})
    if not isinstance(calendar, Mapping):
        raise TypeError("transformations.target.calendar_normalization must be a mapping")
    calendar_unknown = set(calendar) - {"method"}
    if calendar_unknown:
        raise ValueError(
            "unknown transformations.target.calendar_normalization keys: "
            f"{sorted(calendar_unknown)}"
        )
    calendar_method = str(calendar.get("method", "none") or "none").lower()
    if calendar_method not in _CALENDAR_METHODS:
        raise ValueError(
            f"unsupported calendar normalization method {calendar_method!r}; "
            f"expected one of {sorted(_CALENDAR_METHODS)}"
        )

    decomposition = value.get("decomposition", {})
    if not isinstance(decomposition, Mapping):
        raise TypeError("transformations.target.decomposition must be a mapping")
    decomposition_payload = dict(decomposition)
    decomposition_method = str(
        decomposition_payload.get("method", "none") or "none"
    ).lower()
    if decomposition_method not in _DECOMPOSITION_METHODS:
        raise ValueError(
            f"unsupported decomposition method {decomposition_method!r}; "
            f"expected one of {sorted(_DECOMPOSITION_METHODS)}"
        )
    if decomposition_method == "quadratic":
        configured_degree = int(decomposition_payload.get("trend_degree", 2))
        if configured_degree != 2:
            raise ValueError("quadratic decomposition requires trend_degree=2")
        decomposition_payload["method"] = "linear"
        decomposition_payload["trend_degree"] = 2
    elif decomposition_method == "damped":
        configured_forecast = str(
            decomposition_payload.get("trend_forecast", "damped") or "damped"
        ).lower()
        if configured_forecast != "damped":
            raise ValueError("damped decomposition requires trend_forecast='damped'")
        decomposition_payload["method"] = "linear"
        decomposition_payload["trend_forecast"] = "damped"
    else:
        decomposition_payload["method"] = decomposition_method
    resolve_decomposition_spec(
        SimpleNamespace(decomposition=dict(decomposition_payload))
    )

    scaling = value.get("scaling", {})
    if not isinstance(scaling, Mapping):
        raise TypeError("transformations.target.scaling must be a mapping")
    scaling_unknown = set(scaling) - {"method", "inverse"}
    if scaling_unknown:
        raise ValueError(
            f"unknown transformations.target.scaling keys: {sorted(scaling_unknown)}"
        )
    scaling_method = str(scaling.get("method", "none") or "none").lower()
    if scaling_method not in _TARGET_SCALING_METHODS:
        raise ValueError(
            f"unsupported target scaling method {scaling_method!r}; "
            f"expected one of {sorted(_TARGET_SCALING_METHODS)}"
        )
    inverse = scaling.get("inverse", False)
    if not isinstance(inverse, bool):
        raise TypeError("transformations.target.scaling.inverse must be bool")
    return {
        "calendar_normalization": {"method": calendar_method},
        "decomposition": decomposition_payload,
        "scaling": {"method": scaling_method, "inverse": inverse},
    }


@dataclass(frozen=True)
class _CanonicalPipelineFactory:
    transformations: dict[str, dict[str, Any]]

    def __call__(self) -> TargetTransformPipeline:
        target = self.transformations
        scaling = target["scaling"]
        args = SimpleNamespace(
            target_calendar_normalization=target["calendar_normalization"]["method"],
            decomposition=dict(target["decomposition"]),
            scale_target=scaling["method"] != "none",
            inverse_target=bool(scaling["inverse"]),
            target_scaler_type=scaling["method"],
        )
        return TargetTransformPipeline.from_args(args)


class CanonicalTargetTransform:
    """Per-series/per-target target-space state with strict reverse restoration."""

    def __init__(self, transformations: Mapping[str, Any], *, freq: str) -> None:
        self.transformations = normalize_target_transformations(transformations)
        self.freq = str(freq)
        self._pipeline = PerSeriesTargetTransformPipeline(
            _CanonicalPipelineFactory(self.transformations)
        )

    @classmethod
    def from_config(cls, config: Any) -> "CanonicalTargetTransform":
        target = config.features.transformations.get("target", {})
        return cls(target, freq=config.problem.freq)

    @property
    def fitted_keys(self) -> tuple[tuple[Any, str], ...]:
        return self._pipeline.fitted_keys

    @property
    def training_steps(self) -> dict[tuple[Any, str], tuple[str, ...]]:
        return self._pipeline.training_steps

    def fit_transform(self, history: PointForecastTensor) -> PointForecastTensor:
        return self._pipeline.fit_transform(history)

    def transform_point(self, tensor: PointForecastTensor) -> PointForecastTensor:
        return self._pipeline.transform_point(tensor)

    def transform_training(
        self,
        values: np.ndarray,
        origins: Sequence[pd.Timestamp],
        *,
        series_ids: Sequence[Any] | None = None,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 3:
            raise ValueError(
                f"training targets must have shape (samples,H,K); got {array.shape}"
            )
        if len(origins) != array.shape[0]:
            raise ValueError("training origins must match the target sample axis")
        normalized_series_ids = (
            ("__local__",) * array.shape[0]
            if series_ids is None
            else tuple(series_ids)
        )
        if len(normalized_series_ids) != array.shape[0]:
            raise ValueError("training series_ids must match the target sample axis")
        targets = self._pipeline.targets
        if array.shape[2] != len(targets):
            raise ValueError("training target axis must match fitted target order")
        transformed = np.empty_like(array, dtype=float)
        offset = pd.tseries.frequencies.to_offset(self.freq)
        for sample_index, origin in enumerate(origins):
            series_id = normalized_series_ids[sample_index]
            times = pd.DatetimeIndex(
                [pd.Timestamp(origin) + (step + 1) * offset for step in range(array.shape[1])]
            )
            for target_index, target in enumerate(targets):
                transformed[sample_index, :, target_index] = self._pipeline.transform_values(
                    series_id,
                    target,
                    array[sample_index, :, target_index],
                    times,
                )
        return transformed

    def restore_values(
        self,
        series_id: Any,
        target: str,
        values: Any,
        times: Any,
    ) -> np.ndarray:
        return self._pipeline.restore_values(series_id, target, values, times)

    def restore_point(self, tensor: PointForecastTensor) -> PointForecastTensor:
        return self._pipeline.restore_point(tensor)

    def restore_quantiles(
        self,
        tensor: MarginalQuantileForecastTensor,
    ) -> MarginalQuantileForecastTensor:
        return self._pipeline.restore_quantiles(tensor)

    def restore_distribution(
        self,
        distribution: MarginalForecastDistribution,
    ) -> MarginalForecastDistribution:
        quantiles = self.restore_quantiles(distribution.quantiles)
        return MarginalForecastDistribution(
            point=quantiles.point(),
            quantiles=quantiles,
            dependence_model=None,
            metadata=dict(distribution.metadata),
        )


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


__all__ = [
    "CanonicalFeatureScaler",
    "CanonicalTargetTransform",
    "normalize_feature_scaling",
    "normalize_target_transformations",
]


# ==============================================================================
# 目标变换（2026-08-29 架构收敛 P4/D4 自 features/TargetTransformation.py 并入）
# ==============================================================================

def resolve_scale_features_enabled(args) -> bool:
    """优先使用新字段，兼容旧字段 `scale`。"""
    return bool(getattr(args, "scale_features", getattr(args, "scale", False)))

def resolve_scale_target_enabled(args) -> bool:
    """优先使用新字段，兼容旧字段 `scale`。"""
    return bool(getattr(args, "scale_target", getattr(args, "scale", False)))

def resolve_inverse_target_enabled(args) -> bool:
    """优先使用新字段，兼容旧字段 `inverse`。"""
    return bool(getattr(args, "inverse_target", getattr(args, "inverse", False)))

def resolve_feature_scaler_type(args) -> str:
    """获取预测特征 X 的缩放方法。"""
    return str(getattr(args, "feature_scaler_type", "standard")).lower()

def resolve_target_scaler_type(args) -> str:
    """获取目标变量 Y 的缩放方法。"""
    return str(getattr(args, "target_scaler_type", "standard")).lower()

class TargetScaler:
    """
    目标变量缩放器。

    与 FeatureScaler 分离，专门处理训练目标 Y 的缩放与逆变换。
    """

    def __init__(self, args, scaler_type="standard", log_prefix: str="[TargetScaler]", verbose: bool = False):
        self.args = args
        self.log_prefix = log_prefix
        self.verbose = verbose
        self.enabled = resolve_scale_target_enabled(self.args)
        self.scaler_type = str(scaler_type).lower()
        self.column_transformers = {}
        self.column_names = []
        self.is_fitted = False

    def _resolve_columns(self, columns: Optional[List[str]] = None) -> List[str]:
        if columns is not None:
            return list(columns)
        return list(self.column_names)

    def _validate_columns(self, columns: List[str]):
        if not self.column_names:
            raise ValueError(f"{self.log_prefix} Target scaler has not been fitted yet.")
        missing = [col for col in columns if col not in self.column_names]
        if missing:
            raise ValueError(f"{self.log_prefix} Unknown target columns for scaling: {missing}")

    def _create_column_transformer(self):
        if self.scaler_type == "none":
            return None
        if self.scaler_type == "standard":
            return StandardScaler()
        if self.scaler_type == "minmax":
            return MinMaxScaler()
        if self.scaler_type == "robust":
            return RobustScaler()
        if self.scaler_type in {"yeo-johnson", "yeojohnson"}:
            return PowerTransformer(method="yeo-johnson", standardize=True)
        if self.scaler_type == "log1p":
            return "log1p"
        raise ValueError(
            f"{self.log_prefix} Unsupported target_scaler_type={self.scaler_type}. "
            f"Supported: none, standard, minmax, log1p, robust, yeo-johnson."
        )

    def _fit_transform_column(self, values: np.ndarray, column_name: str) -> np.ndarray:
        transformer = self._create_column_transformer()
        self.column_transformers[column_name] = transformer

        if transformer is None:
            return values
        if transformer == "log1p":
            if np.any(values < 0):
                raise ValueError(f"{self.log_prefix} log1p target scaling requires non-negative values, but column '{column_name}' contains negatives.")
            return np.log1p(values)

        return transformer.fit_transform(values)

    def _apply_column_transform(self, values: np.ndarray, column_name: str, inverse: bool) -> np.ndarray:
        transformer = self.column_transformers.get(column_name)
        if transformer is None:
            return values
        if transformer == "log1p":
            if not inverse and np.any(values < 0):
                raise ValueError(f"{self.log_prefix} log1p target scaling requires non-negative values, but column '{column_name}' contains negatives.")
            return np.expm1(values) if inverse else np.log1p(values)
        return transformer.inverse_transform(values) if inverse else transformer.transform(values)

    @staticmethod
    def _ensure_2d_array(y, columns: List[str]):
        original_type = "array"
        original_shape = np.asarray(y).shape
        original_index = None
        original_columns = columns

        if isinstance(y, pd.DataFrame):
            original_type = "dataframe"
            original_index = y.index
            original_columns = y.columns.tolist()
            arr = y.values
        elif isinstance(y, pd.Series):
            original_type = "series"
            original_index = y.index
            original_columns = [y.name if y.name is not None else columns[0]]
            arr = y.to_frame().values
        else:
            arr = np.asarray(y)
            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
            elif arr.ndim == 1:
                if len(columns) > 1:
                    arr = arr.reshape(1, -1)
                else:
                    arr = arr.reshape(-1, 1)

        return arr.astype(float), original_type, original_shape, original_index, original_columns

    @staticmethod
    def _restore_type(arr: np.ndarray, original_type: str, original_shape, original_index, original_columns):
        if original_type == "dataframe":
            return pd.DataFrame(arr, index=original_index, columns=original_columns)
        if original_type == "series":
            return pd.Series(arr.reshape(-1), index=original_index, name=original_columns[0])

        if len(original_shape) == 0:
            return np.asarray(arr).reshape(())
        if len(original_shape) == 1:
            return np.asarray(arr).reshape(-1)
        return np.asarray(arr)

    def fit_transform(self, y):
        if isinstance(y, pd.DataFrame):
            self.column_names = y.columns.tolist()
        elif isinstance(y, pd.Series):
            self.column_names = [y.name if y.name is not None else "target"]
        else:
            arr = np.asarray(y)
            width = arr.shape[1] if arr.ndim == 2 else 1
            self.column_names = [f"target_{i}" for i in range(width)]

        if not self.enabled:
            return y.copy() if hasattr(y, "copy") else np.asarray(y).copy()

        arr, original_type, original_shape, original_index, original_columns = self._ensure_2d_array(
            y,
            self.column_names,
        )
        transformed = np.zeros_like(arr, dtype=float)
        self.column_transformers = {}
        for idx, column_name in enumerate(self.column_names):
            transformed[:, [idx]] = self._fit_transform_column(arr[:, [idx]], column_name)
        self.is_fitted = True

        if self.verbose:
            logger.info(f"{self.log_prefix} Fitted target scaler ({self.scaler_type}) on columns: {self.column_names}")

        return self._restore_type(transformed, original_type, original_shape, original_index, original_columns)

    def transform(self, y, columns: Optional[List[str]] = None):
        if not self.enabled:
            return y.copy() if hasattr(y, "copy") else np.asarray(y).copy()

        resolved_columns = self._resolve_columns(columns)
        self._validate_columns(resolved_columns)
        arr, original_type, original_shape, original_index, original_columns = self._ensure_2d_array(
            y,
            resolved_columns,
        )
        transformed = np.zeros_like(arr, dtype=float)
        for idx, column_name in enumerate(resolved_columns):
            transformed[:, [idx]] = self._apply_column_transform(arr[:, [idx]], column_name, inverse=False)

        return self._restore_type(transformed, original_type, original_shape, original_index, original_columns)

    def inverse_transform(self, y, columns: Optional[List[str]] = None):
        if not self.enabled:
            return y.copy() if hasattr(y, "copy") else np.asarray(y).copy()

        resolved_columns = self._resolve_columns(columns)
        self._validate_columns(resolved_columns)
        arr, original_type, original_shape, original_index, original_columns = self._ensure_2d_array(
            y,
            resolved_columns,
        )
        restored = np.zeros_like(arr, dtype=float)
        for idx, column_name in enumerate(resolved_columns):
            restored[:, [idx]] = self._apply_column_transform(arr[:, [idx]], column_name, inverse=True)

        return self._restore_type(restored, original_type, original_shape, original_index, original_columns)

    def restore_predictions(self, values, target_columns: List[str]):
        """
        在需要时将预测结果从目标缩放空间恢复到原始量纲。
        """
        values_arr = np.asarray(values)
        if not self.enabled:
            return values_arr
        if not resolve_inverse_target_enabled(self.args):
            return values_arr
        return np.asarray(self.inverse_transform(values_arr, columns=target_columns))

    def prepare_eval_target(self, values, target_columns: List[str]):
        """
        统一评估阶段的目标尺度：
        - inverse_target=True: 使用原始量纲
        - inverse_target=False: 将真实值映射到目标缩放空间
        """
        values_arr = np.asarray(values)
        if not self.enabled:
            return values_arr
        if resolve_inverse_target_enabled(self.args):
            return values_arr
        return np.asarray(self.transform(values_arr, columns=target_columns))

    def prepare_history_target_for_plot(self, df_history: pd.DataFrame, target_columns: List[str]):
        """
        预测图保存前，按输出尺度对历史目标列做对齐。
        """
        if (
            not self.enabled
            or resolve_inverse_target_enabled(self.args)
            or df_history is None
            or df_history.empty
            or "y" not in df_history.columns
        ):
            return df_history

        df_history_plot = df_history.copy()
        df_history_plot["y"] = self.transform(df_history_plot["y"], columns=target_columns)
        return df_history_plot


# ==============================================================================
# 目标日历归一化（自 features/TargetNormalization.py 并入）
# ==============================================================================

# -*- coding: utf-8 -*-

class CalendarDayTargetNormalizer:
    """按时间戳所在月份天数归一化月总量。

    ``per_calendar_day`` 模式把月总量除以当月天数后送入模型；输出按目标月
    天数乘回原始月总量空间。该变换无拟合状态，不接触测试标签。
    """

    SUPPORTED = {"none", "per_calendar_day"}

    def __init__(self, args: Any):
        self.method = str(getattr(args, "target_calendar_normalization", "none") or "none").lower()
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


# ==============================================================================
# 目标变换栈（自 features/TargetTransformation.py 并入）
# ==============================================================================

class TargetTransformPipeline:
    """统一 calendar、decomposition 和 target scaling 的目标空间契约。

    训练顺序固定为 calendar normalization → decomposition → target scaling；
    ``restore`` 自动按严格逆序执行。point 和每个 quantile 必须复用同一实例。
    """

    def __init__(
        self,
        args: Any,
        calendar_normalizer: CalendarDayTargetNormalizer,
        decomposition_pipeline: DecompositionPipeline,
        target_scaler: TargetScaler,
    ) -> None:
        self.args = args
        self.calendar_normalizer = calendar_normalizer
        self.decomposition_pipeline = decomposition_pipeline
        self.target_scaler = target_scaler
        self.time_col = "time"
        self.target_col = "y"
        self.target_columns: tuple[str, ...] = ()
        self._training_steps: list[str] = []
        self.history_is_fitted = False

    @classmethod
    def from_args(cls, args: Any) -> "TargetTransformPipeline":
        return cls(
            args=args,
            calendar_normalizer=CalendarDayTargetNormalizer(args),
            decomposition_pipeline=DecompositionPipeline.from_args(args),
            target_scaler=TargetScaler(
                args,
                scaler_type=resolve_target_scaler_type(args),
                verbose=False,
            ),
        )

    @property
    def training_steps(self) -> tuple[str, ...]:
        return tuple(self._training_steps)

    @property
    def decomposition(self) -> DecompositionPipeline:
        """迁移期兼容访问器。"""
        return self.decomposition_pipeline

    def fit_transform_history(
        self,
        frame: pd.DataFrame,
        time_col: str = "time",
        target_col: str = "y",
    ) -> pd.DataFrame:
        """在可见训练历史上依次拟合/应用 calendar 与 decomposition。"""
        if time_col not in frame.columns or target_col not in frame.columns:
            raise ValueError(
                f"target transform requires columns {time_col!r} and {target_col!r}"
            )
        self.time_col = time_col
        self.target_col = target_col
        self._training_steps = []
        result = frame.copy()
        if self.calendar_normalizer.enabled:
            result = self.calendar_normalizer.transform_history(
                result,
                time_col=time_col,
                target_col=target_col,
            )
            self._training_steps.append("calendar_normalization")
        if self.decomposition_pipeline.enabled:
            result = self.decomposition_pipeline.fit_transform(
                result,
                time_col=time_col,
                target_col=target_col,
            )
            self._training_steps.append("decomposition")
        else:
            # none pipeline 仍需标记 fitted，保持持久化/状态检查一致。
            self.decomposition_pipeline.fit(
                result,
                time_col=time_col,
                target_col=target_col,
            )
        self.history_is_fitted = True
        return result

    def fit_transform_targets(self, targets: Any) -> Any:
        """在 calendar/decomposition 之后拟合并应用 target scaler。"""
        if not self.history_is_fitted:
            raise RuntimeError("fit_transform_history must run before fit_transform_targets")
        transformed = self.target_scaler.fit_transform(targets)
        self.target_columns = tuple(str(column) for column in self.target_scaler.column_names)
        if self.target_scaler.enabled:
            self._training_steps.append("target_scaling")
        return transformed

    def attach_fitted_target_scaler(
        self,
        target_scaler: TargetScaler,
        target_columns: Optional[Sequence[str]] = None,
    ) -> None:
        """迁移期接入 Trainer 已拟合 scaler，不重新 fit。"""
        self.target_scaler = target_scaler
        columns = target_columns or getattr(target_scaler, "column_names", ())
        self.target_columns = tuple(columns)
        if bool(getattr(target_scaler, "enabled", False)) and "target_scaling" not in self._training_steps:
            self._training_steps.append("target_scaling")

    @staticmethod
    def _validate_times(times: Any, n_values: int) -> pd.DatetimeIndex:
        index = pd.DatetimeIndex(pd.to_datetime(times))
        if len(index) != n_values:
            raise ValueError(
                f"target restore length mismatch: values={n_values}, times={len(index)}"
            )
        if index.hasnans:
            raise ValueError("target restore times must not contain NaT")
        return index

    def restore(
        self,
        values: Any,
        times: Any,
        target_columns: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """按 scaler → decomposition → calendar 的严格逆序恢复一条序列。"""
        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            array = array.reshape(1)
        if array.ndim != 1:
            raise ValueError(
                f"target restore expects one-dimensional values; got shape={array.shape}"
            )
        time_index = self._validate_times(times, len(array))
        restored = array.copy()

        if self.target_scaler.enabled:
            if not self.target_scaler.is_fitted:
                raise RuntimeError("target scaler must be fitted before restore")
            columns = list(target_columns or self.target_columns)
            if not columns:
                raise ValueError("target_columns are required for target scaling restore")
            restored = np.asarray(
                self.target_scaler.inverse_transform(restored, columns=columns),
                dtype=float,
            ).reshape(-1)
            if len(restored) != len(array):
                raise ValueError(
                    "target scaler restore changed prediction length: "
                    f"before={len(array)}, after={len(restored)}"
                )

        if self.decomposition_pipeline.enabled:
            if not self.decomposition_pipeline.is_fitted:
                raise RuntimeError("decomposition pipeline must be fitted before restore")
            restored = np.asarray(
                self.decomposition_pipeline.restore(restored, time_index),
                dtype=float,
            ).reshape(-1)

        if self.calendar_normalizer.enabled:
            restored = self.calendar_normalizer.restore_predictions(
                restored,
                time_index,
            )
        return np.asarray(restored, dtype=float).reshape(-1)

    def transform(
        self,
        values: Any,
        times: Any,
        target_columns: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """按 calendar → decomposition → scaler 顺序转换一条目标序列。"""
        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            array = array.reshape(1)
        if array.ndim != 1:
            raise ValueError(
                f"target transform expects one-dimensional values; got shape={array.shape}"
            )
        time_index = self._validate_times(times, len(array))
        frame = pd.DataFrame({self.time_col: time_index, self.target_col: array})

        if self.calendar_normalizer.enabled:
            frame = self.calendar_normalizer.transform_history(
                frame,
                time_col=self.time_col,
                target_col=self.target_col,
            )
        if self.decomposition_pipeline.enabled:
            if not self.decomposition_pipeline.is_fitted:
                raise RuntimeError("decomposition pipeline must be fitted before transform")
            frame = self.decomposition_pipeline.transform(frame)

        transformed = frame[self.target_col].to_numpy(dtype=float)
        if self.target_scaler.enabled:
            if not self.target_scaler.is_fitted:
                raise RuntimeError("target scaler must be fitted before transform")
            columns = list(target_columns or self.target_columns)
            if not columns:
                raise ValueError("target_columns are required for target scaling transform")
            transformed = np.asarray(
                self.target_scaler.transform(transformed, columns=columns),
                dtype=float,
            ).reshape(-1)
        return transformed

    def restore_quantile_matrix(
        self,
        values: Any,
        times: Any,
        target_columns: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """对 ``(n_steps, n_quantiles)`` 每列复用与 point 完全相同的 restorer。"""
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(
                f"quantile matrix must be two-dimensional; got shape={matrix.shape}"
            )
        self._validate_times(times, matrix.shape[0])
        restored_columns = [
            self.restore(matrix[:, index], times, target_columns=target_columns)
            for index in range(matrix.shape[1])
        ]
        return np.column_stack(restored_columns)


class PerSeriesTargetTransformPipeline:
    """Own one fitted target-space pipeline for every ``(series, target)`` key."""

    def __init__(self, pipeline_factory: Callable[[], Any]) -> None:
        if not callable(pipeline_factory):
            raise TypeError("pipeline_factory must be callable")
        self.pipeline_factory = pipeline_factory
        self._pipelines: dict[tuple[Any, str], Any] = {}
        self._series_ids: tuple[Any, ...] = ()
        self._targets: tuple[str, ...] = ()

    @property
    def fitted_keys(self) -> tuple[tuple[Any, str], ...]:
        return tuple(self._pipelines)

    @property
    def targets(self) -> tuple[str, ...]:
        return self._targets

    @property
    def training_steps(self) -> dict[tuple[Any, str], tuple[str, ...]]:
        return {
            key: tuple(getattr(pipeline, "training_steps", ()))
            for key, pipeline in self._pipelines.items()
        }

    def fit_transform(self, history: PointForecastTensor) -> PointForecastTensor:
        if not isinstance(history, PointForecastTensor):
            raise TypeError("history must be a PointForecastTensor")
        transformed = np.empty(history.shape, dtype=float)
        pipelines: dict[tuple[Any, str], Any] = {}
        values = history.values
        for series_index, series_id in enumerate(history.series_ids):
            for target_index, target in enumerate(history.targets):
                pipeline = self.pipeline_factory()
                if not callable(getattr(pipeline, "fit_transform_history", None)):
                    raise TypeError("per-key pipeline must provide fit_transform_history")
                if not callable(getattr(pipeline, "fit_transform_targets", None)):
                    raise TypeError("per-key pipeline must provide fit_transform_targets")
                frame = pd.DataFrame(
                    {
                        "time": history.forecast_times,
                        "y": values[series_index, :, target_index],
                    }
                )
                history_transformed = pipeline.fit_transform_history(
                    frame,
                    time_col="time",
                    target_col="y",
                )
                target_transformed = pipeline.fit_transform_targets(
                    history_transformed[["y"]]
                )
                target_array = np.asarray(target_transformed, dtype=float)
                if target_array.shape != (history.n_steps, 1):
                    raise ValueError(
                        "per-key target transform must preserve one-dimensional history"
                    )
                transformed[series_index, :, target_index] = target_array[:, 0]
                pipelines[(series_id, target)] = pipeline

        self._pipelines = pipelines
        self._series_ids = history.series_ids
        self._targets = history.targets
        return PointForecastTensor(
            values=transformed,
            series_ids=history.series_ids,
            forecast_times=history.forecast_times,
            targets=history.targets,
        )

    def transform_point(self, tensor: PointForecastTensor) -> PointForecastTensor:
        if not isinstance(tensor, PointForecastTensor):
            raise TypeError("tensor must be a PointForecastTensor")
        self._validate_identity(tensor.series_ids, tensor.targets)
        transformed = np.empty(tensor.shape, dtype=float)
        values = tensor.values
        for series_index, series_id in enumerate(tensor.series_ids):
            for target_index, target in enumerate(tensor.targets):
                pipeline = self._pipelines[(series_id, target)]
                target_columns = getattr(pipeline, "target_columns", ("y",)) or ("y",)
                transformed[series_index, :, target_index] = np.asarray(
                    pipeline.transform(
                        values[series_index, :, target_index],
                        tensor.forecast_times,
                        target_columns=target_columns,
                    ),
                    dtype=float,
                )
        return PointForecastTensor(
            values=transformed,
            series_ids=tensor.series_ids,
            forecast_times=tensor.forecast_times,
            targets=tensor.targets,
        )

    def transform_values(
        self,
        series_id: Any,
        target: str,
        values: Any,
        times: Any,
    ) -> np.ndarray:
        self._validate_key(series_id, target)
        pipeline = self._pipelines[(series_id, target)]
        target_columns = getattr(pipeline, "target_columns", ("y",)) or ("y",)
        return np.asarray(
            pipeline.transform(values, times, target_columns=target_columns),
            dtype=float,
        )

    def restore_values(
        self,
        series_id: Any,
        target: str,
        values: Any,
        times: Any,
    ) -> np.ndarray:
        self._validate_key(series_id, target)
        pipeline = self._pipelines[(series_id, target)]
        target_columns = getattr(pipeline, "target_columns", ("y",)) or ("y",)
        return np.asarray(
            pipeline.restore(values, times, target_columns=target_columns),
            dtype=float,
        )

    def restore_point(self, tensor: PointForecastTensor) -> PointForecastTensor:
        if not isinstance(tensor, PointForecastTensor):
            raise TypeError("tensor must be a PointForecastTensor")
        self._validate_identity(tensor.series_ids, tensor.targets)
        restored = np.empty(tensor.shape, dtype=float)
        values = tensor.values
        for series_index, series_id in enumerate(tensor.series_ids):
            for target_index, target in enumerate(tensor.targets):
                pipeline = self._pipelines[(series_id, target)]
                target_columns = getattr(pipeline, "target_columns", ("y",)) or ("y",)
                restored[series_index, :, target_index] = np.asarray(
                    pipeline.restore(
                        values[series_index, :, target_index],
                        tensor.forecast_times,
                        target_columns=target_columns,
                    ),
                    dtype=float,
                )
        return PointForecastTensor(
            values=restored,
            series_ids=tensor.series_ids,
            forecast_times=tensor.forecast_times,
            targets=tensor.targets,
        )

    def restore_quantiles(
        self,
        tensor: MarginalQuantileForecastTensor,
    ) -> MarginalQuantileForecastTensor:
        if not isinstance(tensor, MarginalQuantileForecastTensor):
            raise TypeError("tensor must be a MarginalQuantileForecastTensor")
        self._validate_identity(tensor.series_ids, tensor.targets)
        restored = np.empty(tensor.shape, dtype=float)
        values = tensor.values
        for series_index, series_id in enumerate(tensor.series_ids):
            for target_index, target in enumerate(tensor.targets):
                pipeline = self._pipelines[(series_id, target)]
                target_columns = getattr(pipeline, "target_columns", ("y",)) or ("y",)
                for level_index in range(tensor.n_levels):
                    restored[series_index, :, target_index, level_index] = np.asarray(
                        pipeline.restore(
                            values[series_index, :, target_index, level_index],
                            tensor.forecast_times,
                            target_columns=target_columns,
                        ),
                        dtype=float,
                    )
        return MarginalQuantileForecastTensor(
            values=restored,
            levels=tensor.levels,
            point_level=tensor.point_level,
            series_ids=tensor.series_ids,
            forecast_times=tensor.forecast_times,
            targets=tensor.targets,
        )

    def _validate_identity(
        self,
        series_ids: tuple[Any, ...],
        targets: tuple[str, ...],
    ) -> None:
        if not self._pipelines:
            raise RuntimeError("fit_transform must run before restore")
        if series_ids != self._series_ids or targets != self._targets:
            raise ValueError(
                "restore tensor series/target identity must match the fitted transform state"
            )

    def _validate_key(self, series_id: Any, target: str) -> None:
        if not self._pipelines:
            raise RuntimeError("fit_transform must run before transform or restore")
        if (series_id, target) not in self._pipelines:
            raise ValueError(
                f"unknown fitted target transform key: {(series_id, target)!r}"
            )
