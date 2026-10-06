"""预测张量的序列/目标/时间轴校验与时区表示。"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import numpy as np
import pandas as pd


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
