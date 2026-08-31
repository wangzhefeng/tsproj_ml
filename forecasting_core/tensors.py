from dataclasses import FrozenInstanceError, dataclass, field
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd


def _validate_float_array(
    values: np.ndarray,
    ndim: int,
    name: str,
    *,
    check_finite: bool = True,
) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if values.ndim != ndim:
        raise ValueError(f"{name} must have exactly {ndim} dimensions")
    if any(size == 0 for size in values.shape):
        raise ValueError(f"{name} axes must be nonempty")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError(f"{name} must have a floating dtype")
    if check_finite and not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def _immutable_array_storage(values: np.ndarray) -> tuple[bytes, str, tuple[int, ...]]:
    return values.tobytes(order="C"), values.dtype.str, values.shape


def _array_from_storage(
    value_bytes: bytes,
    value_dtype: str,
    value_shape: tuple[int, ...],
) -> np.ndarray:
    return np.frombuffer(value_bytes, dtype=np.dtype(value_dtype)).reshape(value_shape)


def _validate_unique_tuple(values: tuple[Any, ...], name: str) -> tuple[Any, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if not values:
        raise ValueError(f"{name} must be nonempty")
    try:
        unique_count = len(set(values))
    except TypeError as exc:
        raise TypeError(f"{name} entries must be hashable") from exc
    if unique_count != len(values):
        raise ValueError(f"{name} must contain unique entries")
    return tuple(values)


@dataclass(frozen=True, slots=True)
class _TimezoneDescriptor:
    kind: str
    value: str | int


def _zoneinfo_key(value: tzinfo) -> str | None:
    if isinstance(value, ZoneInfo):
        return value.key

    filename = getattr(value, "_filename", None)
    if not isinstance(filename, str):
        return None
    parts = Path(filename).parts
    try:
        zoneinfo_index = len(parts) - 1 - tuple(reversed(parts)).index("zoneinfo")
    except ValueError:
        return None
    key = "/".join(parts[zoneinfo_index + 1 :])
    if not key:
        return None
    try:
        ZoneInfo(key)
    except ZoneInfoNotFoundError:
        return None
    return key


def _fixed_timezone_descriptor(values: tuple[datetime, ...]) -> _TimezoneDescriptor:
    observed_offsets: list[timedelta] = []
    for value in values:
        repeated_offsets = tuple(value.utcoffset() for _ in range(3))
        if any(offset is None or not isinstance(offset, timedelta) for offset in repeated_offsets):
            raise ValueError("timezone UTC offset cannot be fixed safely")
        if len(set(repeated_offsets)) != 1:
            raise ValueError("timezone UTC offset cannot be fixed safely")
        observed_offsets.append(repeated_offsets[0])

    if len(set(observed_offsets)) != 1:
        raise ValueError("timezone UTC offset cannot be fixed safely")
    try:
        offset = observed_offsets[0]
        timezone(offset)
        return _TimezoneDescriptor(
            "fixed",
            ((offset.days * 86400 + offset.seconds) * 1_000_000) + offset.microseconds,
        )
    except ValueError as exc:
        raise ValueError("timezone UTC offset cannot be fixed safely") from exc


def _timezone_descriptor(
    source_timezone: tzinfo,
    values: tuple[datetime, ...],
) -> _TimezoneDescriptor:
    zone_key = _zoneinfo_key(source_timezone)
    if zone_key is not None:
        return _TimezoneDescriptor("zoneinfo", zone_key)
    return _fixed_timezone_descriptor(values)


def _timezone_from_descriptor(descriptor: _TimezoneDescriptor) -> tzinfo:
    if not isinstance(descriptor, _TimezoneDescriptor):
        raise TypeError("timezone descriptor must be _TimezoneDescriptor")
    if descriptor.kind == "zoneinfo":
        if not isinstance(descriptor.value, str):
            raise TypeError("zoneinfo timezone descriptor value must be a string")
        return ZoneInfo(descriptor.value)
    if descriptor.kind == "fixed":
        if isinstance(descriptor.value, bool) or not isinstance(descriptor.value, int):
            raise TypeError("fixed timezone descriptor value must be integer microseconds")
        try:
            return timezone(timedelta(microseconds=descriptor.value))
        except ValueError as exc:
            raise ValueError("invalid fixed timezone descriptor") from exc
    raise ValueError(f"unknown timezone descriptor kind: {descriptor.kind!r}")


def _canonical_timezone(source_timezone: tzinfo, values: tuple[datetime, ...]) -> tzinfo:
    return _timezone_from_descriptor(_timezone_descriptor(source_timezone, values))


def _canonicalize_series_id(value: Any) -> Any:
    if isinstance(value, (np.datetime64, pd.Timestamp, datetime, date)) and pd.isna(value):
        raise ValueError("series_ids must not contain missing datetime-like values")
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value)
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, tuple):
        return tuple(_canonicalize_series_id(item) for item in value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bytes):
        return bytes(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        canonical_value = float(value)
        if not np.isfinite(canonical_value):
            raise ValueError("series_ids must not contain nonfinite floats")
        return canonical_value
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            return value.replace(
                tzinfo=_canonical_timezone(value.tzinfo, (value.to_pydatetime(),))
            )
        return pd.Timestamp(value)
    if isinstance(value, datetime):
        immutable_timezone = None
        if value.tzinfo is not None:
            immutable_timezone = _canonical_timezone(value.tzinfo, (value,))
        return datetime(
            value.year,
            value.month,
            value.day,
            value.hour,
            value.minute,
            value.second,
            value.microsecond,
            tzinfo=immutable_timezone,
            fold=value.fold,
        )
    if isinstance(value, date):
        return date(value.year, value.month, value.day)
    raise TypeError(
        "series_ids entries must be immutable scalar values or nested tuples of them"
    )


def _validate_series_ids(series_ids: tuple[Any, ...]) -> tuple[Any, ...]:
    if not isinstance(series_ids, tuple):
        raise TypeError("series_ids must be a tuple")
    canonical_series_ids = tuple(_canonicalize_series_id(value) for value in series_ids)
    return _validate_unique_tuple(canonical_series_ids, "series_ids")


def _validate_metadata(
    series_ids: tuple[Any, ...],
    forecast_times: pd.DatetimeIndex,
    targets: tuple[str, ...],
    expected_n: int,
    expected_h: int,
    expected_k: int,
) -> tuple[
    tuple[Any, ...],
    tuple[int, ...],
    _TimezoneDescriptor | None,
    tuple[str, ...],
]:
    validated_series_ids = _validate_series_ids(series_ids)
    if len(validated_series_ids) != expected_n:
        raise ValueError("series_ids length must match the series axis")

    if not isinstance(forecast_times, pd.DatetimeIndex):
        raise TypeError("forecast_times must be a pandas.DatetimeIndex")
    if len(forecast_times) == 0:
        raise ValueError("forecast_times must be nonempty")
    if len(forecast_times) != expected_h:
        raise ValueError("forecast_times length must match the horizon axis")
    if forecast_times.hasnans:
        raise ValueError("forecast_times must not contain NaT")
    if not forecast_times.is_monotonic_increasing or not forecast_times.is_unique:
        raise ValueError("forecast_times must be strictly increasing")
    forecast_time_ns = tuple(int(timestamp_ns) for timestamp_ns in forecast_times.as_unit("ns").asi8)
    forecast_time_tz = None
    if forecast_times.tz is not None:
        forecast_time_tz = _timezone_descriptor(
            forecast_times.tz,
            tuple(value.to_pydatetime() for value in forecast_times),
        )

    validated_targets = _validate_unique_tuple(targets, "targets")
    if len(validated_targets) != expected_k:
        raise ValueError("targets length must match the target axis")
    for target in validated_targets:
        if not isinstance(target, str):
            raise TypeError("targets must contain strings")
        if not target.strip():
            raise ValueError("targets must contain nonblank strings")
        if target != target.strip():
            raise ValueError("targets must not contain surrounding whitespace")

    return validated_series_ids, forecast_time_ns, forecast_time_tz, validated_targets


def _datetime_index_from_storage(
    forecast_time_ns: tuple[int, ...],
    forecast_time_tz: _TimezoneDescriptor | None,
) -> pd.DatetimeIndex:
    timestamps = np.asarray(forecast_time_ns, dtype=np.int64)
    if forecast_time_tz is None:
        return pd.DatetimeIndex(timestamps.astype("datetime64[ns]"))
    return pd.to_datetime(timestamps, utc=True).tz_convert(
        _timezone_from_descriptor(forecast_time_tz)
    )


class _FrozenTensor:
    __slots__ = ()

    def __setattr__(self, name: str, value: Any) -> None:
        raise FrozenInstanceError(f"cannot assign to field '{name}'")

    def __delattr__(self, name: str) -> None:
        raise FrozenInstanceError(f"cannot delete field '{name}'")


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


def require_matching_point_axes(
    actual: PointForecastTensor, prediction: PointForecastTensor
) -> None:
    """逐轴校验两个 point 张量对齐（series/targets/forecast_times）。

    轴对齐是张量合同的一部分，消费方（评估 `model_evaluation/point.py`、结果读写
    `model_forecasting/results.py`）共用本函数，不各自实现（2026-08-30 评估模块化）。
    """
    if (
        actual.series_ids != prediction.series_ids
        or actual.targets != prediction.targets
        or not actual.forecast_times.equals(prediction.forecast_times)
    ):
        raise ValueError("actual and prediction axes must match")
