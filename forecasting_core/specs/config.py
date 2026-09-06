"""Aggregate schema-versioned forecasting configuration."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forecasting_core.specs.data import ColumnRole, ColumnSpec, DataSourceSpec, DataSpec
from forecasting_core.specs.estimator import EstimatorSpec
from forecasting_core.specs.feature import FeatureSpec
from forecasting_core.specs.output import OutputSpec
from forecasting_core.specs.problem import ForecastProblemSpec
from forecasting_core.specs.probabilistic import ProbabilisticConfigSpec
from forecasting_core.specs.strategy import ForecastStrategySpec
from forecasting_core.specs.validation import (
    CalendarMonthBacktestSpec,
    RuntimeValidationSpec,
)


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
_ESTIMATOR_FIELDS = frozenset({"model_type", "target_adapter", "params"})
_FORBIDDEN_FIELD_NAMES = frozenset({"base_config", "overrides", "pred_method"})


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


def _validate_time_geometry(
    problem: ForecastProblemSpec,
    validation: RuntimeValidationSpec,
) -> None:
    schedule_mode = str(validation.get("schedule_mode", "daily")).lower()
    if schedule_mode not in {"daily", "intraday"}:
        raise ValueError("validation.schedule_mode must be daily or intraday")
    horizon_mode = str(validation.get("horizon_mode", "fixed_steps")).lower()
    if horizon_mode not in {"fixed_steps", "calendar_month"}:
        raise ValueError(
            "validation.horizon_mode must be fixed_steps or calendar_month"
        )
    if horizon_mode != "calendar_month":
        return
    if problem.freq != "1D":
        raise ValueError("horizon_mode=calendar_month requires problem.freq=1D")
    if schedule_mode != "daily":
        raise ValueError("horizon_mode=calendar_month requires schedule_mode=daily")
    if validation.backtest is not None and not isinstance(
        validation.backtest, CalendarMonthBacktestSpec
    ):
        raise TypeError(
            "horizon_mode=calendar_month requires CalendarMonthBacktestSpec"
        )


@dataclass(frozen=True, slots=True, init=False)
class ForecastConfigSpec:
    schema_version: int
    problem: ForecastProblemSpec
    data: DataSpec
    features: FeatureSpec
    strategy: ForecastStrategySpec | None
    estimator: EstimatorSpec
    probabilistic: ProbabilisticConfigSpec
    validation: RuntimeValidationSpec
    output: OutputSpec

    def __init__(
        self,
        problem: ForecastProblemSpec,
        data: DataSpec,
        features: FeatureSpec,
        strategy: ForecastStrategySpec | None,
        estimator: EstimatorSpec,
        probabilistic: Mapping[str, Any] | ProbabilisticConfigSpec,
        validation: Mapping[str, Any] | RuntimeValidationSpec,
        output: Mapping[str, Any] | OutputSpec,
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
        probabilistic_spec = ProbabilisticConfigSpec.from_mapping(probabilistic)
        validation_spec = RuntimeValidationSpec.from_mapping(validation)
        output_spec = OutputSpec.from_mapping(output)

        if problem.targets != data.target_columns:
            raise ValueError(
                "problem.targets must exactly match data.target_columns in the same order"
            )
        strategy.resolve(problem.horizon)
        _validate_global_source_keys(problem, data)
        _validate_feature_columns(data, features)
        _validate_time_geometry(problem, validation_spec)

        object.__setattr__(self, "schema_version", 2)
        object.__setattr__(self, "problem", problem)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "estimator", estimator)
        object.__setattr__(self, "probabilistic", probabilistic_spec)
        object.__setattr__(self, "validation", validation_spec)
        object.__setattr__(self, "output", output_spec)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "problem": self.problem.canonical_payload(),
            "data": self.data.canonical_payload(),
            "features": self.features.canonical_payload(),
            "strategy": self.strategy.canonical_payload(),
            "estimator": self.estimator.canonical_payload(),
            "probabilistic": self.probabilistic.canonical_payload(),
            "validation": self.validation.canonical_payload(),
            "output": self.output.canonical_payload(),
        }

    def semantic_payload(self) -> dict[str, object]:
        payload = self.canonical_payload()
        payload.pop("output")
        payload["validation"] = self.validation.semantic_payload()
        if self.validation.get("schedule_mode") == "intraday":
            payload["intraday_schedule_semantics"] = "formal_origin_grid_v1"
        target = self.features.transformations.get("target", {})
        if target.get("decomposition", {}).get("method", "none") != "none":
            payload["decomposition_semantics"] = "component_fit_v2"
        if any(target.get(name, {}).get("method", "none") != "none" for name in ("decomposition", "scaling")):
            payload["target_fit_window_semantics"] = "unique_training_labels_v1"
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


def parse_problem_spec(value: Any, source: str | Path) -> ForecastProblemSpec:
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


def parse_data_spec(value: Any, source: str | Path) -> DataSpec:
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


def parse_probabilistic_config_spec(
    value: Any,
    source: str | Path,
) -> ProbabilisticConfigSpec:
    return ProbabilisticConfigSpec.from_mapping(value, source=_source_label(source))


def parse_runtime_validation_spec(
    value: Any,
    source: str | Path,
    *,
    require_geometry: bool = True,
) -> RuntimeValidationSpec:
    return RuntimeValidationSpec.from_mapping(
        value,
        source=_source_label(source),
        require_geometry=require_geometry,
    )


def parse_output_spec(value: Any, source: str | Path) -> OutputSpec:
    return OutputSpec.from_mapping(value, source=_source_label(source))


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
        problem=parse_problem_spec(payload["problem"], source),
        data=parse_data_spec(payload["data"], source),
        features=_parse_features(payload["features"], source),
        strategy=_parse_strategy(payload["strategy"], source),
        estimator=_parse_estimator(payload["estimator"], source),
        probabilistic=parse_probabilistic_config_spec(
            payload["probabilistic"], source
        ),
        validation=parse_runtime_validation_spec(payload["validation"], source),
        output=parse_output_spec(payload["output"], source),
    )


__all__ = [
    "ForecastConfigSpec",
    "parse_data_spec",
    "parse_model_config",
    "parse_output_spec",
    "parse_probabilistic_config_spec",
    "parse_problem_spec",
    "parse_runtime_validation_spec",
]
