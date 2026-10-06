"""点预测张量及其 pickle 重建合同。"""

from dataclasses import dataclass, field
from forecasting_core.tensors.axes import (
    _TimezoneDescriptor,
    _datetime_index_from_storage,
    _validate_metadata,
)
from forecasting_core.tensors.storage import (
    _FrozenTensor,
    _array_from_storage,
    _immutable_array_storage,
    _validate_float_array,
)
from typing import Any
import numpy as np
import pandas as pd


def _reconstruct_point_forecast_tensor(
    value_bytes: bytes,
    value_dtype: str,
    value_shape: tuple[int, int, int],
    series_ids: tuple[Any, ...],
    forecast_time_ns: tuple[int, ...],
    forecast_time_tz: _TimezoneDescriptor | None,
    targets: tuple[str, ...],
) -> "PointForecastTensor":
    return PointForecastTensor(
        values=_array_from_storage(value_bytes, value_dtype, value_shape),
        series_ids=series_ids,
        forecast_times=_datetime_index_from_storage(forecast_time_ns, forecast_time_tz),
        targets=targets,
    )


@dataclass(slots=True, init=False)
class PointForecastTensor(_FrozenTensor):
    """Point forecasts with axes ``(N, H, K)`` for series, time, and target.

    Time-major matrix order flattens each series as time first, then target:
    ``(t0, k0), (t0, k1), ..., (t1, k0), ...``.
    """

    _value_bytes: bytes = field(repr=False)
    _value_dtype: str = field(repr=False)
    _value_shape: tuple[int, int, int] = field(repr=False)
    series_ids: tuple[Any, ...]
    _forecast_time_ns: tuple[int, ...] = field(repr=False)
    _forecast_time_tz: _TimezoneDescriptor | None = field(repr=False)
    targets: tuple[str, ...]

    def __init__(
        self,
        values: np.ndarray,
        series_ids: tuple[Any, ...],
        forecast_times: pd.DatetimeIndex,
        targets: tuple[str, ...],
    ) -> None:
        values = _validate_float_array(values, ndim=3, name="values")
        series_ids, forecast_time_ns, forecast_time_tz, targets = _validate_metadata(
            series_ids,
            forecast_times,
            targets,
            expected_n=values.shape[0],
            expected_h=values.shape[1],
            expected_k=values.shape[2],
        )
        value_bytes, value_dtype, value_shape = _immutable_array_storage(values)
        object.__setattr__(self, "_value_bytes", value_bytes)
        object.__setattr__(self, "_value_dtype", value_dtype)
        object.__setattr__(self, "_value_shape", value_shape)
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
        return _reconstruct_point_forecast_tensor, (
            self._value_bytes,
            self._value_dtype,
            self._value_shape,
            self.series_ids,
            self._forecast_time_ns,
            self._forecast_time_tz,
            self.targets,
        )

    @property
    def shape(self) -> tuple[int, int, int]:
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

    def to_time_major_matrix(self) -> np.ndarray:
        """Return ``(N, H*K)`` using time-major, target-minor column order."""
        return self.values.reshape(self.n_series, self.n_steps * self.n_targets).copy()

    @classmethod
    def from_time_major_matrix(
        cls,
        matrix: np.ndarray,
        *,
        series_ids: tuple[Any, ...],
        forecast_times: pd.DatetimeIndex,
        targets: tuple[str, ...],
    ) -> "PointForecastTensor":
        """Build an ``(N, H, K)`` tensor from time-major matrix columns."""
        matrix = _validate_float_array(matrix, ndim=2, name="matrix", check_finite=False)
        expected_shape = (len(series_ids), len(forecast_times) * len(targets))
        if matrix.shape != expected_shape:
            raise ValueError(f"matrix must have shape {expected_shape}")
        return cls(
            values=matrix.reshape(len(series_ids), len(forecast_times), len(targets)),
            series_ids=series_ids,
            forecast_times=forecast_times,
            targets=targets,
        )

    def select_target(self, name: str) -> "PointForecastTensor":
        try:
            target_index = self.targets.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return PointForecastTensor(
            values=self.values[:, :, target_index : target_index + 1],
            series_ids=self.series_ids,
            forecast_times=self.forecast_times,
            targets=(name,),
        )

    def select_series(self, series_id: Any) -> "PointForecastTensor":
        try:
            series_index = self.series_ids.index(series_id)
        except ValueError as exc:
            raise KeyError(series_id) from exc
        stored_id = self.series_ids[series_index]
        return PointForecastTensor(
            values=self.values[series_index : series_index + 1, :, :],
            series_ids=(stored_id,),
            forecast_times=self.forecast_times,
            targets=self.targets,
        )
