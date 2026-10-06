"""目标变换职责拆分；旧持久化类路径不提供兼容别名。"""
from __future__ import annotations
from typing import Any, Callable
import numpy as np
import pandas as pd
from forecasting_core.tensors.point import PointForecastTensor
from forecasting_core.tensors.quantile import MarginalQuantileForecastTensor


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
