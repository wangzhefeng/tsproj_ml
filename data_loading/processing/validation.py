"""物理数据投影、数值、时间与主键校验；不决定请求可见范围。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting_core.specs.data import AvailabilityPolicy, ColumnRole, DataSourceSpec

_GENERATED_AVAILABLE_AT = "available_at"


def validate_frame(
    source: DataSourceSpec,
    frame: pd.DataFrame,
    *,
    generated: bool,
    path_version: str | None,
) -> pd.DataFrame:
    validated = frame.copy(deep=True)
    declared = {column.name for column in source.columns}
    ignored = {
        column.name
        for column in source.columns
        if column.role is ColumnRole.IGNORED
    }
    required = declared - ignored
    permitted = set(declared)
    if source.time_col is not None:
        required.add(source.time_col)
        permitted.add(source.time_col)
    available_at_col = resolve_available_at_col(source)
    if available_at_col is not None:
        permitted.add(available_at_col)
        if path_version == "history" and available_at_col not in validated:
            if source.time_col is None or source.time_col not in validated:
                raise ValueError(
                    f"source {source.name!r} history cannot derive available_at without time_col"
                )
            validated[available_at_col] = validated[source.time_col]
        else:
            required.add(available_at_col)
    if generated:
        required.add(_GENERATED_AVAILABLE_AT)
        permitted.add(_GENERATED_AVAILABLE_AT)
    actual = set(validated.columns)
    if not required.issubset(actual):
        missing = sorted(required - actual)
        raise ValueError(
            f"source {source.name!r} columns do not match its spec; missing={missing}"
        )
    projected_columns = [
        column for column in validated.columns if column in permitted
    ]
    validated = validated.loc[:, projected_columns].copy()

    if source.time_col is not None:
        validated[source.time_col] = parse_datetime(
            validated[source.time_col],
            f"source {source.name!r} time_col",
        )
    if available_at_col is not None:
        validated[available_at_col] = parse_datetime(
            validated[available_at_col],
            f"source {source.name!r} available_at",
        )

    for column in source.columns:
        if column.name not in validated:
            if column.role is ColumnRole.IGNORED:
                continue
            raise ValueError(
                f"source {source.name!r} required column {column.name!r} is missing"
            )
        values = validated[column.name]
        if column.role is ColumnRole.IGNORED:
            continue
        if values.isna().any():
            raise ValueError(f"source {source.name!r} column {column.name!r} contains missing values")
        if column.categorical or column.role is ColumnRole.KEY:
            continue
        try:
            numeric = pd.to_numeric(values, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"source {source.name!r} column {column.name!r} must be numeric"
            ) from exc
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"source {source.name!r} column {column.name!r} must be finite")
        validated[column.name] = numeric

    primary_key = source_primary_key(source)
    if source.availability not in {AvailabilityPolicy.COLUMN, AvailabilityPolicy.GENERATOR_DEFINED}:
        if validated.duplicated(primary_key, keep=False).any():
            raise ValueError(f"source {source.name!r} has duplicate primary keys")
    return validated


def parse_datetime(values: pd.Series, label: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, errors="raise", format="mixed")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain valid timestamps") from exc
    if parsed.isna().any():
        raise ValueError(f"{label} must not contain NaT")
    return parsed


def resolve_available_at_col(source: DataSourceSpec) -> str | None:
    if source.availability is AvailabilityPolicy.COLUMN:
        return source.available_at_col
    if source.availability is AvailabilityPolicy.GENERATOR_DEFINED:
        return _GENERATED_AVAILABLE_AT
    return None


def source_primary_key(source: DataSourceSpec) -> list[str]:
    keys = list(source.series_id_cols)
    if source.time_col is not None:
        keys.append(source.time_col)
    if not keys:
        raise ValueError(f"source {source.name!r} has no primary key")
    return keys
