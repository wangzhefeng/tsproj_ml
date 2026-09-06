"""Canonical target and feature transformations consumed by the canonical runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional, List, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.base import clone

from decomposition import DecompositionPipeline
from feature_engineering.transform_specs import (
    normalize_feature_scaling,
    normalize_target_transformations,
)
# CanonicalFeatureScaler 住在同包 scaling.py（2026-09-06 R3 归位，re-export 桥删除）
from forecasting_core.tensors import (
    MarginalQuantileForecastTensor,
    PointForecastTensor,
)
from forecasting_core.artifacts import MarginalForecastDistribution
from utils.log_util import logger


@dataclass(frozen=True)
class _CanonicalPipelineFactory:
    transformations: dict[str, dict[str, Any]]

    def __call__(self) -> TargetTransformPipeline:
        target = self.transformations
        scaling = target["scaling"]
        decomposition = dict(target["decomposition"])
        decomposition.pop("fit_history_steps", None)
        args = SimpleNamespace(
            target_calendar_normalization=target["calendar_normalization"]["method"],
            decomposition=decomposition,
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
        self.fit_window_metadata: dict[str, Any] = {}
        self.is_identity = all(
            self.transformations[name]["method"] == "none"
            for name in ("calendar_normalization", "decomposition", "scaling")
        )
        self._identity_series_ids: tuple[Any, ...] = ()
        self._identity_targets: tuple[str, ...] = ()
        self._pipeline = PerSeriesTargetTransformPipeline(
            _CanonicalPipelineFactory(self.transformations)
        )

    @classmethod
    def from_config(cls, config: Any) -> "CanonicalTargetTransform":
        target = config.features.transformations.get("target", {})
        return cls(target, freq=config.problem.freq)

    @property
    def fitted_keys(self) -> tuple[tuple[Any, str], ...]:
        if self.is_identity:
            return tuple(
                (series_id, target)
                for series_id in self._identity_series_ids
                for target in self._identity_targets
            )
        return self._pipeline.fitted_keys

    @property
    def training_steps(self) -> dict[tuple[Any, str], tuple[str, ...]]:
        if self.is_identity:
            return {key: () for key in self.fitted_keys}
        return self._pipeline.training_steps

    def fit_transform(self, history: PointForecastTensor, *, scaling_times: pd.DatetimeIndex | None = None) -> PointForecastTensor:
        if self.is_identity:
            if not isinstance(history, PointForecastTensor):
                raise TypeError("history must be a PointForecastTensor")
            self.fit_identity(history.series_ids, history.targets)
            return history
        return self._pipeline.fit_transform(history, scaling_times=scaling_times)

    def fit_identity(
        self,
        series_ids: Sequence[Any],
        targets: Sequence[str],
    ) -> None:
        if not self.is_identity:
            raise RuntimeError("fit_identity requires an identity target transform")
        self._identity_series_ids = tuple(series_ids)
        self._identity_targets = tuple(targets)

    def transform_point(self, tensor: PointForecastTensor) -> PointForecastTensor:
        if self.is_identity:
            if not isinstance(tensor, PointForecastTensor):
                raise TypeError("tensor must be a PointForecastTensor")
            self._validate_identity(tensor.series_ids, tensor.targets)
            return tensor
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
        if self.is_identity and not self._identity_series_ids:
            raise RuntimeError("fit_transform must run before transform")
        targets = (
            self._identity_targets
            if self.is_identity
            else self._pipeline.targets
        )
        if array.shape[2] != len(targets):
            raise ValueError("training target axis must match fitted target order")
        if self.is_identity:
            return array
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
        if self.is_identity:
            if not self._identity_series_ids:
                raise RuntimeError(
                    "fit_transform must run before transform or restore"
                )
            if (series_id, target) not in self.fitted_keys:
                raise ValueError(
                    f"unknown fitted target transform key: {(series_id, target)!r}"
                )
            array = np.asarray(values, dtype=float)
            if array.ndim == 0:
                array = array.reshape(1)
            if array.ndim != 1:
                raise ValueError(
                    "target restore expects one-dimensional values; "
                    f"got shape={array.shape}"
                )
            time_index = pd.DatetimeIndex(pd.to_datetime(times))
            if len(time_index) != len(array):
                raise ValueError(
                    "target restore length mismatch: "
                    f"values={len(array)}, times={len(time_index)}"
                )
            if time_index.hasnans:
                raise ValueError("target restore times must not contain NaT")
            return array
        return self._pipeline.restore_values(series_id, target, values, times)

    def restore_point(self, tensor: PointForecastTensor) -> PointForecastTensor:
        if self.is_identity:
            if not isinstance(tensor, PointForecastTensor):
                raise TypeError("tensor must be a PointForecastTensor")
            self._validate_identity(tensor.series_ids, tensor.targets)
            return tensor
        return self._pipeline.restore_point(tensor)

    def _validate_identity(
        self,
        series_ids: tuple[Any, ...],
        targets: tuple[str, ...],
    ) -> None:
        if not self._identity_series_ids:
            raise RuntimeError("fit_transform must run before restore")
        if series_ids != self._identity_series_ids or targets != self._identity_targets:
            raise ValueError(
                "restore tensor series/target identity must match the fitted transform state"
            )

    def restore_quantiles(
        self,
        tensor: MarginalQuantileForecastTensor,
    ) -> MarginalQuantileForecastTensor:
        if self.is_identity:
            if not isinstance(tensor, MarginalQuantileForecastTensor):
                raise TypeError("tensor must be a MarginalQuantileForecastTensor")
            self._validate_identity(tensor.series_ids, tensor.targets)
            return tensor
        return self._pipeline.restore_quantiles(tensor)

    def restore_distribution(
        self,
        distribution: MarginalForecastDistribution,
    ) -> MarginalForecastDistribution:
        if self.is_identity:
            self.restore_quantiles(distribution.quantiles)
            return distribution
        quantiles = self.restore_quantiles(distribution.quantiles)
        return MarginalForecastDistribution(
            point=quantiles.point(),
            quantiles=quantiles,
            dependence_model=None,
            metadata=dict(distribution.metadata),
        )




# ==============================================================================
# 目标变换（2026-08-29 架构收敛 P4/D4 自 features/TargetTransformation.py 并入）
# ==============================================================================

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
        # target scaling 默认启用（canonical 配置无旧版 scale 开关残留）
        self.enabled = True
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

    def fit_transform(self, history: PointForecastTensor, *, scaling_times: pd.DatetimeIndex | None = None) -> PointForecastTensor:
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
                scaling_rows = (
                    history_transformed if scaling_times is None
                    else history_transformed.loc[history_transformed["time"].isin(scaling_times)]
                )
                if scaling_rows.empty:
                    raise ValueError("target scaler has no training label rows")
                target_transformed = pipeline.fit_transform_targets(scaling_rows[["y"]])
                if scaling_times is not None:
                    target_transformed = pipeline.target_scaler.transform(history_transformed[["y"]])
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
