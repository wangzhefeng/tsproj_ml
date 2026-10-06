"""目标变换职责拆分；旧持久化类路径不提供兼容别名。"""
from __future__ import annotations
from typing import Any, Optional, Sequence
import numpy as np
import pandas as pd
from decomposition import DecompositionPipeline, DecompositionSpec, PresetParams
from feature_engineering.transforms.calendar import CalendarDayTargetNormalizer
from feature_engineering.transforms.target_scaling import TargetScaler


class TargetTransformPipeline:
    """统一 calendar、decomposition 和 target scaling 的目标空间契约。

    训练顺序固定为 calendar normalization → decomposition → target scaling；
    ``restore`` 自动按严格逆序执行。point 和每个 quantile 必须复用同一实例。
    """

    def __init__(self, transformations) -> None:
        decomposition = dict(transformations["decomposition"])
        decomposition.pop("fit_history_steps", None)
        if "periods" in decomposition:
            decomposition["periods"] = tuple(decomposition["periods"])
        preset = PresetParams(**decomposition)
        self.calendar_normalizer = CalendarDayTargetNormalizer(transformations["calendar_normalization"]["method"])
        self.decomposition_pipeline = DecompositionPipeline(DecompositionSpec(
            method=preset.method, composition=preset.composition, preset=preset,
        ))
        self.target_scaler = TargetScaler(transformations["scaling"]["method"])
        self.time_col = "time"
        self.target_col = "y"
        self.target_columns: tuple[str, ...] = ()
        self._training_steps: list[str] = []
        self.history_is_fitted = False

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
        self._training_steps.append("target_scaling")
        return transformed

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
