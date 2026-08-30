"""Explicit data-source and column-role specifications."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


class ColumnRole(str, Enum):
    TARGET = "target"
    OBSERVED_PAST = "observed_past"
    KNOWN_FUTURE = "known_future"
    STATIC = "static"
    KEY = "key"
    IGNORED = "ignored"


class AvailabilityPolicy(str, Enum):
    SOURCE_TIME = "source_time"
    COLUMN = "column"
    GENERATOR_DEFINED = "generator_defined"
    FORECAST_ORIGIN = "forecast_origin"


_MODELED_ROLES = frozenset(
    {
        ColumnRole.TARGET,
        ColumnRole.OBSERVED_PAST,
        ColumnRole.KNOWN_FUTURE,
        ColumnRole.STATIC,
    }
)
_TEMPORAL_ROLES = frozenset(
    {
        ColumnRole.TARGET,
        ColumnRole.OBSERVED_PAST,
        ColumnRole.KNOWN_FUTURE,
    }
)
_GENERATED_ROLES = frozenset(
    {
        ColumnRole.KNOWN_FUTURE,
        ColumnRole.KEY,
        ColumnRole.IGNORED,
    }
)
_STATIC_SOURCE_ROLES = frozenset(
    {
        ColumnRole.STATIC,
        ColumnRole.KEY,
        ColumnRole.IGNORED,
    }
)
_SOURCE_TYPES = frozenset({"file", "generated"})


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be blank")
    if value != stripped:
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    return value


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field_name)


def _normalize_role(value: Any) -> ColumnRole:
    if isinstance(value, ColumnRole):
        return value
    if not isinstance(value, str):
        raise TypeError("role must be a ColumnRole or string")
    try:
        return ColumnRole(value)
    except ValueError as exc:
        raise ValueError(f"unknown column role: {value!r}") from exc


def _normalize_availability(value: Any) -> AvailabilityPolicy | None:
    if value is None or isinstance(value, AvailabilityPolicy):
        return value
    if not isinstance(value, str):
        raise TypeError("availability must be an AvailabilityPolicy, string, or None")
    try:
        return AvailabilityPolicy(value)
    except ValueError as exc:
        raise ValueError(f"unknown availability policy: {value!r}") from exc


def _normalize_names(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_names = (value,)
    elif isinstance(value, Sequence):
        raw_names = tuple(value)
    else:
        raise TypeError(f"{field_name} must be a string or sequence of strings")

    names = tuple(_required_string(name, f"{field_name} entries") for name in raw_names)
    if len(names) != len(set(names)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return names


@dataclass(frozen=True, slots=True, init=False)
class ColumnSpec:
    name: str
    role: ColumnRole
    categorical: bool = False

    def __init__(
        self,
        name: str,
        role: ColumnRole | str,
        categorical: bool = False,
    ) -> None:
        normalized_name = _required_string(name, "name")
        normalized_role = _normalize_role(role)
        if not isinstance(categorical, bool):
            raise TypeError("categorical must be a bool")
        if normalized_role is ColumnRole.TARGET and categorical:
            raise ValueError("target columns cannot be categorical")

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "role", normalized_role)
        object.__setattr__(self, "categorical", categorical)


@dataclass(frozen=True, slots=True, init=False)
class DataSourceSpec:
    name: str
    source_type: Literal["file", "generated"]
    columns: tuple[ColumnSpec, ...]
    history_path: str | None = None
    backtest_path: str | None = None
    future_path: str | None = None
    time_col: str | None = None
    series_id_cols: tuple[str, ...] = ()
    availability: AvailabilityPolicy | None = None
    available_at_col: str | None = None
    generator: str | None = None
    provider: str | None = None

    def __init__(
        self,
        name: str,
        source_type: Literal["file", "generated"],
        columns: Sequence[ColumnSpec],
        history_path: str | None = None,
        backtest_path: str | None = None,
        future_path: str | None = None,
        time_col: str | None = None,
        series_id_cols: str | Sequence[str] = (),
        availability: AvailabilityPolicy | str | None = None,
        available_at_col: str | None = None,
        generator: str | None = None,
        provider: str | None = None,
    ) -> None:
        normalized_name = _required_string(name, "name")
        if not isinstance(source_type, str):
            raise TypeError("source_type must be a string")
        if source_type not in _SOURCE_TYPES:
            raise ValueError(f"unknown source_type: {source_type!r}")
        if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence):
            raise TypeError("columns must be a sequence of ColumnSpec")
        normalized_columns = tuple(columns)
        if not normalized_columns:
            raise ValueError("columns must not be empty")
        if any(not isinstance(column, ColumnSpec) for column in normalized_columns):
            raise TypeError("columns entries must be ColumnSpec instances")
        column_names = tuple(column.name for column in normalized_columns)
        if len(column_names) != len(set(column_names)):
            raise ValueError("columns must have unique names within a source")

        normalized_history_path = _optional_string(history_path, "history_path")
        normalized_backtest_path = _optional_string(backtest_path, "backtest_path")
        normalized_future_path = _optional_string(future_path, "future_path")
        normalized_time_col = _optional_string(time_col, "time_col")
        normalized_series_id_cols = _normalize_names(series_id_cols, "series_id_cols")
        normalized_availability = _normalize_availability(availability)
        normalized_available_at_col = _optional_string(available_at_col, "available_at_col")
        normalized_generator = _optional_string(generator, "generator")
        normalized_provider = _optional_string(provider, "provider")
        if normalized_provider not in {None, "persistence", "auxiliary", "provided_scenario"}:
            raise ValueError(
                "provider must be persistence, auxiliary, provided_scenario, or None"
            )

        metadata_names = tuple(
            name
            for name in (normalized_time_col, normalized_available_at_col)
            if name is not None
        )
        if len(metadata_names) != len(set(metadata_names)):
            raise ValueError("time_col and available_at_col must be distinct")
        if set(metadata_names) & set(column_names):
            raise ValueError("metadata columns must be distinct from declared columns")

        roles = frozenset(column.role for column in normalized_columns)
        # Target and observed-past columns may share one historical table because
        # both obey the same source-time visibility. Known-future values require
        # a separate source-level availability contract.
        if ColumnRole.TARGET in roles and ColumnRole.KNOWN_FUTURE in roles:
            raise ValueError("target cannot be mixed with known_future in one source")
        key_columns = frozenset(
            column.name for column in normalized_columns if column.role is ColumnRole.KEY
        )
        if key_columns != frozenset(normalized_series_id_cols):
            raise ValueError("series_id_cols must exactly match columns with role=key")

        paths = (
            normalized_history_path,
            normalized_backtest_path,
            normalized_future_path,
        )
        if source_type == "file":
            if not any(paths):
                raise ValueError("file sources require at least one path")
            if normalized_generator is not None:
                raise ValueError("file sources forbid generator")
        else:
            if normalized_generator is None:
                raise ValueError("generated sources require generator")
            if any(path is not None for path in paths):
                raise ValueError("generated sources forbid file paths")
            if not roles.issubset(_GENERATED_ROLES):
                raise ValueError("generated sources may only emit known_future, key, or ignored columns")

        temporal_roles = roles & _TEMPORAL_ROLES
        if temporal_roles and normalized_availability is None:
            raise ValueError("temporal sources require availability")
        if not temporal_roles:
            if normalized_availability is not None:
                raise ValueError("non-temporal sources require availability=None")
            if normalized_available_at_col is not None:
                raise ValueError("non-temporal sources forbid available_at_col")

        if normalized_availability is AvailabilityPolicy.SOURCE_TIME:
            if normalized_time_col is None:
                raise ValueError("availability=source_time requires time_col")
            if normalized_available_at_col is not None:
                raise ValueError("availability=source_time forbids available_at_col")
            if ColumnRole.KNOWN_FUTURE in temporal_roles:
                raise ValueError("availability=source_time is forbidden for known_future")
        elif normalized_availability is AvailabilityPolicy.COLUMN:
            if normalized_time_col is None:
                raise ValueError("availability=column requires time_col")
            if normalized_available_at_col is None:
                raise ValueError("availability=column requires available_at_col")
        elif normalized_availability is AvailabilityPolicy.GENERATOR_DEFINED:
            if source_type != "generated" or ColumnRole.KNOWN_FUTURE not in temporal_roles:
                raise ValueError(
                    "availability=generator_defined is only valid for generated known_future sources"
                )
            if normalized_available_at_col is not None:
                raise ValueError("availability=generator_defined forbids available_at_col")
        elif normalized_availability is AvailabilityPolicy.FORECAST_ORIGIN:
            if ColumnRole.KNOWN_FUTURE not in temporal_roles:
                raise ValueError(
                    "availability=forecast_origin is only valid for known_future sources"
                )
            if normalized_time_col is None:
                raise ValueError("availability=forecast_origin requires time_col")
            if normalized_available_at_col is not None:
                raise ValueError(
                    "availability=forecast_origin forbids available_at_col"
                )

        if ColumnRole.STATIC in roles:
            if not roles.issubset(_STATIC_SOURCE_ROLES):
                raise ValueError("static sources may only contain static, key, or ignored columns")
            if normalized_time_col is not None:
                raise ValueError("static sources forbid time_col")
            if normalized_backtest_path is not None or normalized_future_path is not None:
                raise ValueError("static sources forbid backtest_path and future_path")
            if normalized_history_path is None:
                raise ValueError("static sources require history_path")
            if not normalized_series_id_cols:
                raise ValueError("static sources require at least one series_id_col")

        if ColumnRole.TARGET in roles:
            if normalized_history_path is None:
                raise ValueError("target sources require history_path")
            if normalized_backtest_path is not None or normalized_future_path is not None:
                raise ValueError("target sources forbid backtest_path and future_path")

        if ColumnRole.OBSERVED_PAST in roles:
            if source_type == "file" and normalized_history_path is None:
                raise ValueError("observed_past file sources require history_path")
            if normalized_future_path is not None:
                raise ValueError("observed_past sources forbid future_path")
            if normalized_provider is None:
                raise ValueError(
                    "observed_past sources require an explicit provider"
                )
        elif normalized_provider is not None:
            raise ValueError("provider is only valid for observed_past sources")

        if source_type == "file" and ColumnRole.KNOWN_FUTURE in roles:
            if normalized_future_path is None:
                raise ValueError("known_future file sources require future_path")
            if normalized_availability not in {
                AvailabilityPolicy.COLUMN,
                AvailabilityPolicy.FORECAST_ORIGIN,
            }:
                raise ValueError(
                    "known_future file sources require availability=column or forecast_origin"
                )
        if source_type == "generated" and ColumnRole.KNOWN_FUTURE in roles:
            if normalized_availability is not AvailabilityPolicy.GENERATOR_DEFINED:
                raise ValueError(
                    "generated known_future sources require availability=generator_defined"
                )

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "columns", normalized_columns)
        object.__setattr__(self, "history_path", normalized_history_path)
        object.__setattr__(self, "backtest_path", normalized_backtest_path)
        object.__setattr__(self, "future_path", normalized_future_path)
        object.__setattr__(self, "time_col", normalized_time_col)
        object.__setattr__(self, "series_id_cols", normalized_series_id_cols)
        object.__setattr__(self, "availability", normalized_availability)
        object.__setattr__(self, "available_at_col", normalized_available_at_col)
        object.__setattr__(self, "generator", normalized_generator)
        object.__setattr__(self, "provider", normalized_provider)


@dataclass(frozen=True, slots=True, init=False)
class DataSpec:
    sources: tuple[DataSourceSpec, ...]

    def __init__(self, sources: Sequence[DataSourceSpec]) -> None:
        if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
            raise TypeError("sources must be a sequence of DataSourceSpec")
        normalized_sources = tuple(sources)
        if not normalized_sources:
            raise ValueError("sources must not be empty")
        if any(not isinstance(source, DataSourceSpec) for source in normalized_sources):
            raise TypeError("sources entries must be DataSourceSpec instances")

        source_names = tuple(source.name for source in normalized_sources)
        if len(source_names) != len(set(source_names)):
            raise ValueError("source names must be unique")

        name_roles: dict[str, set[ColumnRole]] = {}
        for source in normalized_sources:
            for column in source.columns:
                name_roles.setdefault(column.name, set()).add(column.role)
        if any(len(roles) > 1 for roles in name_roles.values()):
            raise ValueError("each column name must have exactly one role across all sources")

        modeled_names = [
            column.name
            for source in normalized_sources
            for column in source.columns
            if column.role in _MODELED_ROLES
        ]
        if len(modeled_names) != len(set(modeled_names)):
            raise ValueError("modeled feature and target column names must be globally unique")
        if not any(
            column.role is ColumnRole.TARGET
            for source in normalized_sources
            for column in source.columns
        ):
            raise ValueError("DataSpec requires at least one target column")

        object.__setattr__(self, "sources", normalized_sources)

    @property
    def target_columns(self) -> tuple[str, ...]:
        return self.role_columns(ColumnRole.TARGET)

    def role_columns(self, role: ColumnRole | str) -> tuple[str, ...]:
        normalized_role = _normalize_role(role)
        names = (
            column.name
            for source in self.sources
            for column in source.columns
            if column.role is normalized_role
        )
        return tuple(dict.fromkeys(names))

    def canonical_payload(self) -> dict[str, object]:
        payload_sources = []
        for source in self.sources:
            source_payload: dict[str, object] = {
                "name": source.name,
                "source_type": source.source_type,
                "columns": [
                    {
                        "name": column.name,
                        "role": column.role.value,
                        "categorical": column.categorical,
                    }
                    for column in source.columns
                ],
                "series_id_cols": list(source.series_id_cols),
            }
            optional_values = (
                ("history_path", source.history_path),
                ("backtest_path", source.backtest_path),
                ("future_path", source.future_path),
                ("time_col", source.time_col),
                ("generator", source.generator),
            )
            for key, value in optional_values:
                if value is not None:
                    source_payload[key] = value
            if source.availability is not None:
                source_payload["availability"] = source.availability.value
            if source.availability is AvailabilityPolicy.COLUMN:
                source_payload["available_at_col"] = source.available_at_col
            if source.provider is not None:
                source_payload["provider"] = source.provider
            payload_sources.append(source_payload)
        return {"sources": payload_sources}


__all__ = [
    "AvailabilityPolicy",
    "ColumnRole",
    "ColumnSpec",
    "DataSourceSpec",
    "DataSpec",
]
