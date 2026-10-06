"""边际分位预测张量及其 pickle 重建合同。"""

from dataclasses import dataclass, field
from forecasting_core.tensors.axes import (
    _TimezoneDescriptor,
    _datetime_index_from_storage,
    _validate_metadata,
)
from forecasting_core.tensors.point import PointForecastTensor
from forecasting_core.tensors.storage import (
    _FrozenTensor,
    _array_from_storage,
    _immutable_array_storage,
    _validate_float_array,
)
from typing import Any
import numpy as np
import pandas as pd


def _reconstruct_marginal_quantile_forecast_tensor(
    value_bytes: bytes,
    value_dtype: str,
    value_shape: tuple[int, int, int, int],
    levels: tuple[float, ...],
    point_level: float,
    series_ids: tuple[Any, ...],
    forecast_time_ns: tuple[int, ...],
    forecast_time_tz: _TimezoneDescriptor | None,
    targets: tuple[str, ...],
) -> "MarginalQuantileForecastTensor":
    return MarginalQuantileForecastTensor(
        values=_array_from_storage(value_bytes, value_dtype, value_shape),
        levels=levels,
        point_level=point_level,
        series_ids=series_ids,
        forecast_times=_datetime_index_from_storage(forecast_time_ns, forecast_time_tz),
        targets=targets,
    )


@dataclass(slots=True, init=False)
class MarginalQuantileForecastTensor(_FrozenTensor):
    """Marginal quantile forecasts with axes ``(N, H, K, Q)``.

    The axes are series, forecast time, target, and increasing quantile level.
    """

    _value_bytes: bytes = field(repr=False)
    _value_dtype: str = field(repr=False)
    _value_shape: tuple[int, int, int, int] = field(repr=False)
    levels: tuple[float, ...]
    point_level: float
    series_ids: tuple[Any, ...]
    _forecast_time_ns: tuple[int, ...] = field(repr=False)
    _forecast_time_tz: _TimezoneDescriptor | None = field(repr=False)
    targets: tuple[str, ...]

    def __init__(
        self,
        values: np.ndarray,
        levels: tuple[float, ...],
        point_level: float,
        series_ids: tuple[Any, ...],
        forecast_times: pd.DatetimeIndex,
        targets: tuple[str, ...],
    ) -> None:
        values = _validate_float_array(values, ndim=4, name="values")
        series_ids, forecast_time_ns, forecast_time_tz, targets = _validate_metadata(
            series_ids,
            forecast_times,
            targets,
            expected_n=values.shape[0],
            expected_h=values.shape[1],
            expected_k=values.shape[2],
        )
        if not isinstance(levels, tuple):
            raise TypeError("levels must be a tuple")
        if len(levels) != values.shape[3]:
            raise ValueError("levels length must match the quantile axis")
        if any(not isinstance(level, (float, np.floating)) for level in levels):
            raise TypeError("levels entries must be scalar floating values")
        validated_levels = tuple(float(level) for level in levels)
        level_array = np.asarray(validated_levels, dtype=float)
        if not np.isfinite(level_array).all():
            raise ValueError("levels must be finite")
        if np.any((level_array <= 0.0) | (level_array >= 1.0)):
            raise ValueError("levels must be inside (0, 1)")
        if level_array.size > 1 and not np.all(np.diff(level_array) > 0.0):
            raise ValueError("levels must be unique and strictly increasing")
        if not isinstance(point_level, (float, np.floating)):
            raise TypeError("point_level must be a scalar floating value")
        point_level = float(point_level)
        if not np.isfinite(point_level):
            raise ValueError("point_level must be finite")
        if point_level in validated_levels:
            canonical_point_level = validated_levels[validated_levels.index(point_level)]
        else:
            matching_levels = [
                level for level in validated_levels if np.isclose(point_level, level, rtol=0.0, atol=1e-8)
            ]
            if len(matching_levels) != 1:
                raise ValueError("point_level must match exactly one level")
            canonical_point_level = matching_levels[0]

        value_bytes, value_dtype, value_shape = _immutable_array_storage(values)
        object.__setattr__(self, "_value_bytes", value_bytes)
        object.__setattr__(self, "_value_dtype", value_dtype)
        object.__setattr__(self, "_value_shape", value_shape)
        object.__setattr__(self, "levels", validated_levels)
        object.__setattr__(self, "point_level", canonical_point_level)
        object.__setattr__(self, "series_ids", series_ids)
        object.__setattr__(self, "_forecast_time_ns", forecast_time_ns)
        object.__setattr__(self, "_forecast_time_tz", forecast_time_tz)
        object.__setattr__(self, "targets", targets)

    @property
    def values(self) -> np.ndarray:
        return _array_from_storage(self._value_bytes, self._value_dtype, self._value_shape)

    @property
    def forecast_times(self) -> pd.DatetimeIndex:
        return _datetime_index_from_storage(self._forecast_time_ns, self._forecast_time_tz)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return _reconstruct_marginal_quantile_forecast_tensor, (
            self._value_bytes,
            self._value_dtype,
            self._value_shape,
            self.levels,
            self.point_level,
            self.series_ids,
            self._forecast_time_ns,
            self._forecast_time_tz,
            self.targets,
        )

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self._value_shape

    @property
    def n_series(self) -> int:
        return self._value_shape[0]

    @property
    def n_steps(self) -> int:
        return self._value_shape[1]

    @property
    def n_targets(self) -> int:
        return self._value_shape[2]

    @property
    def n_levels(self) -> int:
        return self._value_shape[3]

    def point(self) -> PointForecastTensor:
        level_index = self.levels.index(self.point_level)
        return PointForecastTensor(
            values=self.values[:, :, :, level_index],
            series_ids=self.series_ids,
            forecast_times=self.forecast_times,
            targets=self.targets,
        )

    def crossing_mask(self) -> np.ndarray:
        return np.any(np.diff(self.values, axis=3) < 0.0, axis=3)

    def has_crossing(self) -> bool:
        return bool(self.crossing_mask().any())

    def select_target(self, name: str) -> "MarginalQuantileForecastTensor":
        try:
            target_index = self.targets.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return MarginalQuantileForecastTensor(
            values=self.values[:, :, target_index : target_index + 1, :],
            levels=self.levels,
            point_level=self.point_level,
            series_ids=self.series_ids,
            forecast_times=self.forecast_times,
            targets=(name,),
        )

    def select_series(self, series_id: Any) -> "MarginalQuantileForecastTensor":
        try:
            series_index = self.series_ids.index(series_id)
        except ValueError as exc:
            raise KeyError(series_id) from exc
        stored_id = self.series_ids[series_index]
        return MarginalQuantileForecastTensor(
            values=self.values[series_index : series_index + 1, :, :, :],
            levels=self.levels,
            point_level=self.point_level,
            series_ids=(stored_id,),
            forecast_times=self.forecast_times,
            targets=self.targets,
        )
