"""Aggregate schema-versioned forecasting configuration."""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from model_forecasting.specs.data import ColumnRole, ColumnSpec, DataSourceSpec, DataSpec
from model_forecasting.specs.estimator import EstimatorSpec
from model_forecasting.specs.feature import FeatureSpec
from model_forecasting.specs.problem import ForecastProblemSpec
from model_forecasting.specs.strategy import ForecastStrategySpec


_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "problem",
        "data",
        "features",
        "strategy",
        "estimator",
        "probabilistic",
        "validation",
        "output",
    }
)
_PROBLEM_FIELDS = frozenset(
    {
        "time_col",
        "freq",
        "horizon",
        "targets",
        "information_mode",
        "training_scope",
        "series_id_cols",
    }
)
_DATA_FIELDS = frozenset({"sources"})
_SOURCE_FIELDS = frozenset(
    {
        "name",
        "source_type",
        "columns",
        "history_path",
        "backtest_path",
        "future_path",
        "time_col",
        "series_id_cols",
        "availability",
        "available_at_col",
        "generator",
        "provider",
    }
)
_COLUMN_FIELDS = frozenset({"name", "role", "categorical"})
_FEATURE_FIELDS = frozenset(
    {
        "target_lags",
        "observed_past_lags",
        "datetime_features",
        "transformations",
    }
)
# features.selection 为可选段（监督特征选择，2026-08-30 专项恢复）
_FEATURE_OPTIONAL_FIELDS = frozenset({"selection"})
_STRATEGY_FIELDS = frozenset({"name", "output_chunk_length"})
_ENSEMBLE_FIELDS = frozenset({"members", "method", "weights"})
_ENSEMBLE_MEMBER_FIELDS = frozenset({"name", "strategy"})
_ESTIMATOR_FIELDS = frozenset({"model_type", "target_adapter", "params"})
_FORBIDDEN_FIELD_NAMES = frozenset({"base_config", "overrides", "pred_method"})
_NONSEMANTIC_FIELD_TOKENS = ("parallel", "worker", "thread", "n_jobs", "log")
_NONSEMANTIC_VALIDATION_FIELDS: frozenset[str] = frozenset()


def _freeze_json_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite floats")
        return value
    if isinstance(value, list):
        return tuple(
            _freeze_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            normalized[key] = _freeze_json_value(item, f"{path}.{key}")
        return MappingProxyType(dict(sorted(normalized.items())))
    raise TypeError(f"{path} must contain only JSON-like values")


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    return value


def _semantic_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for key, item in value.items():
        normalized = str(key).lower()
        if normalized in _NONSEMANTIC_VALIDATION_FIELDS or any(
            token in normalized for token in _NONSEMANTIC_FIELD_TOKENS
        ):
            continue
        if isinstance(item, Mapping):
            result[str(key)] = _semantic_mapping(item)
        elif isinstance(item, tuple):
            result[str(key)] = [
                _semantic_mapping(element) if isinstance(element, Mapping) else element
                for element in item
            ]
        else:
            result[str(key)] = item
    return result


def _validate_global_source_keys(problem: ForecastProblemSpec, data: DataSpec) -> None:
    if not problem.is_global:
        return

    expected = problem.series_id_cols
    for source in data.sources:
        source_roles = frozenset(column.role for column in source.columns)
        must_be_keyed = bool(
            source_roles
            & {ColumnRole.TARGET, ColumnRole.OBSERVED_PAST, ColumnRole.STATIC}
        )
        if must_be_keyed and source.series_id_cols != expected:
            raise ValueError(
                "global problem series_id_cols must match keyed data sources exactly"
            )
        if source.series_id_cols and source.series_id_cols != expected:
            raise ValueError(
                "global problem series_id_cols must match each source series_id_cols"
            )


def _validate_feature_columns(data: DataSpec, features: FeatureSpec) -> None:
    target_lag_columns = frozenset(features.target_lags)
    observed_lag_columns = frozenset(features.observed_past_lags)
    overlapping_columns = target_lag_columns & observed_lag_columns
    if overlapping_columns:
        raise ValueError(
            "target_lags and observed_past_lags must not overlap: "
            f"{sorted(overlapping_columns)}"
        )

    target_columns = frozenset(data.target_columns)
    unknown_target_lags = target_lag_columns - target_columns
    if unknown_target_lags:
        raise ValueError(
            "target_lags keys must be declared target columns: "
            f"{sorted(unknown_target_lags)}"
        )

    observed_columns = frozenset(data.role_columns(ColumnRole.OBSERVED_PAST))
    unknown_observed_lags = observed_lag_columns - observed_columns
    if unknown_observed_lags:
        raise ValueError(
            "observed_past_lags keys must be declared observed_past columns: "
            f"{sorted(unknown_observed_lags)}"
        )


@dataclass(frozen=True, slots=True, init=False)
class ForecastConfigSpec:
    schema_version: int
    problem: ForecastProblemSpec
    data: DataSpec
    features: FeatureSpec
    strategy: ForecastStrategySpec | None
    estimator: EstimatorSpec
    probabilistic: Mapping[str, Any]
    validation: Mapping[str, Any]
    output: Mapping[str, Any]

    def __init__(
        self,
        problem: ForecastProblemSpec,
        data: DataSpec,
        features: FeatureSpec,
        strategy: ForecastStrategySpec | None,
        estimator: EstimatorSpec,
        probabilistic: Mapping[str, Any],
        validation: Mapping[str, Any],
        output: Mapping[str, Any],
        schema_version: int = 2,
    ) -> None:
        if strategy is None:
            raise ValueError(
                "ForecastConfigSpec is base-only (v4): strategy is required; "
                "ensemble configs use model_ensemble.specs.EnsembleConfigSpec"
            )
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != 2
        ):
            raise ValueError("schema_version must be 2")
        expected_types = (
            ("problem", problem, ForecastProblemSpec),
            ("data", data, DataSpec),
            ("features", features, FeatureSpec),
            ("estimator", estimator, EstimatorSpec),
        )
        for field_name, value, expected_type in expected_types:
            if not isinstance(value, expected_type):
                raise TypeError(f"{field_name} must be {expected_type.__name__}")
        if not isinstance(strategy, ForecastStrategySpec):
            raise TypeError("strategy must be ForecastStrategySpec")
        for field_name, value in (
            ("probabilistic", probabilistic),
            ("validation", validation),
            ("output", output),
        ):
            if not isinstance(value, Mapping):
                raise TypeError(f"{field_name} must be a mapping")

        if problem.targets != data.target_columns:
            raise ValueError(
                "problem.targets must exactly match data.target_columns in the same order"
            )
        strategy.resolve(problem.horizon)
        _validate_global_source_keys(problem, data)
        _validate_feature_columns(data, features)

        object.__setattr__(self, "schema_version", 2)
        object.__setattr__(self, "problem", problem)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "estimator", estimator)
        object.__setattr__(
            self,
            "probabilistic",
            _freeze_json_value(probabilistic, "probabilistic"),
        )
        object.__setattr__(
            self,
            "validation",
            _freeze_json_value(validation, "validation"),
        )
        object.__setattr__(self, "output", _freeze_json_value(output, "output"))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "problem": self.problem.canonical_payload(),
            "data": self.data.canonical_payload(),
            "features": self.features.canonical_payload(),
            "strategy": self.strategy.canonical_payload(),
            "estimator": self.estimator.canonical_payload(),
            "probabilistic": _thaw_json_value(self.probabilistic),
            "validation": _thaw_json_value(self.validation),
            "output": _thaw_json_value(self.output),
        }

    def semantic_payload(self) -> dict[str, object]:
        payload = self.canonical_payload()
        payload.pop("output")
        payload["validation"] = _semantic_mapping(
            _thaw_json_value(self.validation)
        )
        return payload

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.semantic_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def result_identity(self, hash_length: int = 12) -> str:
        if isinstance(hash_length, bool) or not isinstance(hash_length, int):
            raise TypeError("hash_length must be an integer")
        if hash_length < 8 or hash_length > 64:
            raise ValueError("hash_length must be between 8 and 64")
        strategy_label = self.strategy.name.value
        return "-".join(
            (
                strategy_label,
                self.estimator.model_type,
                self.problem.training_scope,
                f"k{len(self.problem.targets)}",
                self.fingerprint()[:hash_length],
            )
        )


def _source_label(source: str | Path) -> str:
    return str(source)


def _mapping(
    value: Any,
    *,
    path: str,
    source: str | Path,
    allowed: frozenset[str],
    required: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping in {_source_label(source)}")
    keys = set(value)
    if any(not isinstance(key, str) for key in keys):
        raise TypeError(f"{path} keys must be strings in {_source_label(source)}")
    unknown = sorted(keys - allowed)
    missing = sorted(required - keys)
    if unknown:
        raise ValueError(
            f"Unknown fields in {path} from {_source_label(source)}: {unknown}"
        )
    if missing:
        raise ValueError(
            f"Missing required fields in {path} from {_source_label(source)}: {missing}"
        )
    return value


def _sequence(value: Any, *, path: str, source: str | Path) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path} must be a sequence in {_source_label(source)}")
    return value


def _reject_forbidden_fields(value: Any, *, path: str, source: str | Path) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and (
                key in _FORBIDDEN_FIELD_NAMES or key.startswith("enable_")
            ):
                raise ValueError(
                    f"forbidden legacy field {path}.{key} in {_source_label(source)}"
                )
            _reject_forbidden_fields(
                item,
                path=f"{path}.{key}",
                source=source,
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_forbidden_fields(
                item,
                path=f"{path}[{index}]",
                source=source,
            )


def _parse_problem(value: Any, source: str | Path) -> ForecastProblemSpec:
    payload = _mapping(
        value,
        path="problem",
        source=source,
        allowed=_PROBLEM_FIELDS,
        required=_PROBLEM_FIELDS - {"series_id_cols"},
    )
    return ForecastProblemSpec(**payload)


def _parse_column(value: Any, source: str | Path, index_path: str) -> ColumnSpec:
    payload = _mapping(
        value,
        path=index_path,
        source=source,
        allowed=_COLUMN_FIELDS,
        required=_COLUMN_FIELDS - {"categorical"},
    )
    return ColumnSpec(**payload)


def _parse_source(value: Any, source: str | Path, index: int) -> DataSourceSpec:
    path = f"data.sources[{index}]"
    payload = _mapping(
        value,
        path=path,
        source=source,
        allowed=_SOURCE_FIELDS,
        required=frozenset({"name", "source_type", "columns"}),
    )
    columns = _sequence(payload["columns"], path=f"{path}.columns", source=source)
    parsed = dict(payload)
    parsed["columns"] = tuple(
        _parse_column(column, source, f"{path}.columns[{column_index}]")
        for column_index, column in enumerate(columns)
    )
    return DataSourceSpec(**parsed)


def _parse_data(value: Any, source: str | Path) -> DataSpec:
    payload = _mapping(
        value,
        path="data",
        source=source,
        allowed=_DATA_FIELDS,
        required=_DATA_FIELDS,
    )
    sources = _sequence(payload["sources"], path="data.sources", source=source)
    return DataSpec(
        tuple(_parse_source(item, source, index) for index, item in enumerate(sources))
    )


def _parse_features(value: Any, source: str | Path) -> FeatureSpec:
    payload = _mapping(
        value,
        path="features",
        source=source,
        allowed=_FEATURE_FIELDS | _FEATURE_OPTIONAL_FIELDS,
        required=_FEATURE_FIELDS,
    )
    return FeatureSpec(**payload)


def _parse_strategy(
    value: Any,
    source: str | Path,
    *,
    path: str = "strategy",
) -> ForecastStrategySpec | None:
    if value is None:
        return None
    payload = _mapping(
        value,
        path=path,
        source=source,
        allowed=_STRATEGY_FIELDS,
        required=frozenset({"name"}),
    )
    return ForecastStrategySpec(**payload)


def _parse_estimator(value: Any, source: str | Path) -> EstimatorSpec:
    payload = _mapping(
        value,
        path="estimator",
        source=source,
        allowed=_ESTIMATOR_FIELDS,
        required=frozenset({"model_type", "target_adapter"}),
    )
    return EstimatorSpec(**payload)


def _parse_json_mapping(value: Any, field_name: str, source: str | Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping in {_source_label(source)}")
    return value


def parse_model_config(mapping: Mapping[str, Any], source: str | Path) -> ForecastConfigSpec:
    """Parse a single-model canonical config (base-only, v4 §5.1).

    Ensemble-shaped documents are rejected here; they belong to
    `model_ensemble.loader.parse_ensemble_document` (the production dispatcher in
    `config/config_loader.py` routes by the mutually exclusive field sets).
    """
    if isinstance(mapping.get("ensemble"), Mapping):
        raise ValueError(
            f"ensemble-shaped config is not a single-model config; route it "
            f"through model_ensemble.loader instead: {_source_label(source)}"
        )
    _reject_forbidden_fields(mapping, path="config", source=source)
    payload = _mapping(
        mapping,
        path="config",
        source=source,
        allowed=_TOP_LEVEL_FIELDS - {"ensemble"},
        required=_TOP_LEVEL_FIELDS - {"ensemble"},
    )
    schema_version = payload["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 2
    ):
        raise ValueError(f"schema_version must be 2 in {_source_label(source)}")

    return ForecastConfigSpec(
        schema_version=2,
        problem=_parse_problem(payload["problem"], source),
        data=_parse_data(payload["data"], source),
        features=_parse_features(payload["features"], source),
        strategy=_parse_strategy(payload["strategy"], source),
        estimator=_parse_estimator(payload["estimator"], source),
        probabilistic=_parse_json_mapping(payload["probabilistic"], "probabilistic", source),
        validation=_parse_json_mapping(payload["validation"], "validation", source),
        output=_parse_json_mapping(payload["output"], "output", source),
    )


__all__ = ["ForecastConfigSpec", "parse_model_config"]
