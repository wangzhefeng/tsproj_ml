"""联合样本张量类型边界及其 pickle 重建合同。"""

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


def _reconstruct_sample_forecast_tensor(
    value_bytes: bytes,
    value_dtype: str,
    value_shape: tuple[int, int, int, int],
    series_ids: tuple[Any, ...],
    forecast_time_ns: tuple[int, ...],
    forecast_time_tz: _TimezoneDescriptor | None,
    targets: tuple[str, ...],
    dependence_model: None,
) -> "SampleForecastTensor":
    if dependence_model is not None:
        raise ValueError("dependence_model must be None")
    return SampleForecastTensor(
        values=_array_from_storage(value_bytes, value_dtype, value_shape),
        series_ids=series_ids,
        forecast_times=_datetime_index_from_storage(forecast_time_ns, forecast_time_tz),
        targets=targets,
    )


@dataclass(slots=True, init=False)
class SampleForecastTensor(_FrozenTensor):
    """Joint forecast samples with axes ``(N, S, H, K)``.

    The axes are series, sample, forecast time, and target. Sample generation is
    intentionally reserved until a dependence model contract is implemented.
    """

    _value_bytes: bytes = field(repr=False)
    _value_dtype: str = field(repr=False)
    _value_shape: tuple[int, int, int, int] = field(repr=False)
    series_ids: tuple[Any, ...]
    _forecast_time_ns: tuple[int, ...] = field(repr=False)
    _forecast_time_tz: _TimezoneDescriptor | None = field(repr=False)
    targets: tuple[str, ...]
    dependence_model: None = field(default=None, init=False)

    def __init__(
        self,
        values: np.ndarray,
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
            expected_h=values.shape[2],
            expected_k=values.shape[3],
        )
        value_bytes, value_dtype, value_shape = _immutable_array_storage(values)
        object.__setattr__(self, "_value_bytes", value_bytes)
        object.__setattr__(self, "_value_dtype", value_dtype)
        object.__setattr__(self, "_value_shape", value_shape)
        object.__setattr__(self, "series_ids", series_ids)
        object.__setattr__(self, "_forecast_time_ns", forecast_time_ns)
        object.__setattr__(self, "_forecast_time_tz", forecast_time_tz)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "dependence_model", None)

    @classmethod
    def generate(cls, *args: Any, **kwargs: Any) -> "SampleForecastTensor":
        """Reserve the public sample-generation boundary for future work."""
        raise NotImplementedError("sample generation is reserved and not implemented")

    @property
    def values(self) -> np.ndarray:
        return _array_from_storage(self._value_bytes, self._value_dtype, self._value_shape)

    @property
    def forecast_times(self) -> pd.DatetimeIndex:
        return _datetime_index_from_storage(self._forecast_time_ns, self._forecast_time_tz)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return _reconstruct_sample_forecast_tensor, (
            self._value_bytes,
            self._value_dtype,
            self._value_shape,
            self.series_ids,
            self._forecast_time_ns,
            self._forecast_time_tz,
            self.targets,
            self.dependence_model,
        )

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self._value_shape

    @property
    def n_series(self) -> int:
        return self._value_shape[0]

    @property
    def n_samples(self) -> int:
        return self._value_shape[1]

    @property
    def n_steps(self) -> int:
        return self._value_shape[2]

    @property
    def n_targets(self) -> int:
        return self._value_shape[3]

    def select_target(self, name: str) -> "SampleForecastTensor":
        try:
            target_index = self.targets.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return SampleForecastTensor(
            values=self.values[:, :, :, target_index : target_index + 1],
            series_ids=self.series_ids,
            forecast_times=self.forecast_times,
            targets=(name,),
        )

    def select_series(self, series_id: Any) -> "SampleForecastTensor":
        try:
            series_index = self.series_ids.index(series_id)
        except ValueError as exc:
            raise KeyError(series_id) from exc
        stored_id = self.series_ids[series_index]
        return SampleForecastTensor(
            values=self.values[series_index : series_index + 1, :, :, :],
            series_ids=(stored_id,),
            forecast_times=self.forecast_times,
            targets=self.targets,
        )
