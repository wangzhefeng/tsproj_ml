"""Visibility-driven feature compilation for canonical forecasting specs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast
import warnings

import numpy as np
import pandas as pd

from data_loading import (
    EndogenousFutureProvider,
    InformationSetRequest,
    MaterializedInformationSet,
    SourceLineage,
)
from forecasting_core.specs import (
    AvailabilityPolicy,
    ColumnRole,
    DataSourceSpec,
    ForecastConfigSpec,
    StrategyName,
)
from feature_engineering.transform_specs import (
    normalize_feature_scaling,
    normalize_target_transformations,
)


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    feature_names: tuple[str, ...]
    categorical_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VisibilityProof:
    feature_name: str
    source_name: str
    role: str
    target_time: pd.Timestamp
    source_time: pd.Timestamp | None
    forecast_origin: pd.Timestamp
    horizon_step: int
    available_at: pd.Timestamp
    provider: str | None = None


class CompiledFeatures:
    """Defensive compiled feature frame plus auditable visibility metadata."""

    __slots__ = ("_frame", "schema", "source_lineage", "visibility_proof")

    def __init__(
        self,
        *,
        frame: pd.DataFrame,
        schema: FeatureSchema,
        source_lineage: Sequence[SourceLineage],
        visibility_proof: Sequence[VisibilityProof],
    ) -> None:
        self._frame = frame.copy(deep=True)
        self.schema = schema
        self.source_lineage = tuple(source_lineage)
        self.visibility_proof = tuple(visibility_proof)

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy(deep=True)


class FeatureCompiler:
    """Compile features from one strict materialized information set."""

    _DATETIME_FEATURES = {
        "minute": lambda value: value.minute,
        "hour": lambda value: value.hour,
        "day": lambda value: value.day,
        "day_of_week": lambda value: value.dayofweek,
        "week_of_year": lambda value: int(value.isocalendar().week),
        "month": lambda value: value.month,
        "quarter": lambda value: value.quarter,
        "day_of_year": lambda value: value.dayofyear,
        "days_in_month": lambda value: value.days_in_month,
        "year": lambda value: value.year,
        "is_weekend": lambda value: int(value.dayofweek >= 5),
        "is_month_start": lambda value: int(value.is_month_start),
        "is_month_end": lambda value: int(value.is_month_end),
        "is_quarter_start": lambda value: int(value.is_quarter_start),
        "is_quarter_end": lambda value: int(value.is_quarter_end),
        "is_year_start": lambda value: int(value.is_year_start),
        "is_year_end": lambda value: int(value.is_year_end),
    }
    _TRANSFORMATION_KEYS = frozenset(
        {
            "direct",
            "advanced",
            "feature_scaling",
            "target",
            "datetime_categorical",
            "interactions",
        }
    )

    def __init__(self, config: ForecastConfigSpec) -> None:
        if not isinstance(config, ForecastConfigSpec):
            raise TypeError("config must be a ForecastConfigSpec")
        self.config = config
        self.problem = config.problem
        self.data = config.data
        self.features = config.features
        self.resolved_strategy = config.strategy.resolve(config.problem.horizon)
        self.datetime_features = tuple(self.features.datetime_features)
        unknown_datetime = sorted(
            set(self.datetime_features) - set(self._DATETIME_FEATURES)
        )
        if unknown_datetime:
            raise ValueError(f"unsupported datetime features: {unknown_datetime}")
        self.datetime_categorical = self._normalize_datetime_categorical(
            self.features.transformations.get("datetime_categorical", ()),
        )
        self._validate_runtime_transformations()

    def compile(
        self,
        information_set: MaterializedInformationSet,
        request: InformationSetRequest,
        *,
        target_future_providers: Mapping[Any, EndogenousFutureProvider] | None = None,
        observed_future_providers: Mapping[Any, EndogenousFutureProvider] | None = None,
        horizon_steps: Sequence[int] | None = None,
        visibility_cutoff: pd.Timestamp | None = None,
    ) -> CompiledFeatures:
        if not isinstance(information_set, MaterializedInformationSet):
            raise TypeError("information_set must be a MaterializedInformationSet")
        if not isinstance(request, InformationSetRequest):
            raise TypeError("request must be an InformationSetRequest")
        if request.H != self.problem.horizon:
            raise ValueError(
                "information-set horizon does not match ForecastProblemSpec: "
                f"request={request.H}, problem={self.problem.horizon}"
            )
        if request.information_mode != self.problem.information_mode:
            raise ValueError("information-set mode does not match ForecastProblemSpec")

        target_providers = self._normalize_providers(target_future_providers)
        observed_providers = self._normalize_providers(observed_future_providers)
        identities = self._request_identities(request)
        selected_steps = self._normalize_horizon_steps(horizon_steps, request.H)
        rows: list[dict[str, Any]] = []
        proofs: list[VisibilityProof] = []

        for identity in identities:
            for step_index in selected_steps:
                target_time = request.forecast_times[step_index]
                target_timestamp = cast(pd.Timestamp, pd.Timestamp(target_time))
                history_anchor_time = self._history_anchor_time(
                    request,
                    target_timestamp,
                )
                row = self._identity_payload(identity)
                row["target_time"] = target_timestamp
                row["horizon_step"] = step_index + 1

                self._compile_lag_mapping(
                    row=row,
                    proofs=proofs,
                    role=ColumnRole.TARGET,
                    lag_mapping=self.features.target_lags,
                    identity=identity,
                    target_time=target_timestamp,
                    source_anchor_time=history_anchor_time,
                    step_index=step_index,
                    request=request,
                    information_set=information_set,
                    providers=target_providers,
                )
                self._compile_lag_mapping(
                    row=row,
                    proofs=proofs,
                    role=ColumnRole.OBSERVED_PAST,
                    lag_mapping=self.features.observed_past_lags,
                    identity=identity,
                    target_time=target_timestamp,
                    source_anchor_time=history_anchor_time,
                    step_index=step_index,
                    request=request,
                    information_set=information_set,
                    providers=observed_providers,
                )
                self._compile_known_future(
                    row,
                    proofs,
                    identity,
                    target_timestamp,
                    step_index,
                    request,
                    information_set,
                )
                self._compile_static(
                    row,
                    proofs,
                    identity,
                    target_timestamp,
                    step_index,
                    request,
                    information_set,
                )
                self._compile_datetime(
                    row,
                    proofs,
                    target_timestamp,
                    step_index,
                    request,
                )
                self._compile_transformations(
                    row,
                    proofs,
                    identity,
                    target_timestamp,
                    step_index,
                    request,
                    information_set,
                )
                rows.append(row)

        frame = pd.DataFrame(rows)
        key_columns = [*self.problem.series_id_cols, "target_time", "horizon_step"]
        feature_names = tuple(column for column in frame.columns if column not in key_columns)
        categorical_names = tuple(
            column
            for source in self.data.sources
            for column_spec in source.columns
            if column_spec.categorical and column_spec.name in feature_names
            for column in (column_spec.name,)
        )
        categorical_names = (
            *categorical_names,
            *(
                f"dt_{name}"
                for name in self.datetime_categorical
                if f"dt_{name}" in feature_names
            ),
        )
        normalized_cutoff = pd.Timestamp(
            request.forecast_origin
            if visibility_cutoff is None
            else visibility_cutoff
        )
        if normalized_cutoff is pd.NaT:
            raise ValueError("visibility_cutoff must be a valid timestamp")
        cutoff = cast(pd.Timestamp, normalized_cutoff)
        if cutoff < request.forecast_origin:
            raise ValueError("visibility_cutoff must be at or after forecast_origin")
        self._validate_visibility_proofs(proofs, cutoff)
        return CompiledFeatures(
            frame=frame,
            schema=FeatureSchema(
                feature_names=feature_names,
                categorical_names=tuple(dict.fromkeys(categorical_names)),
            ),
            source_lineage=information_set.lineage,
            visibility_proof=proofs,
        )

    def _normalize_datetime_categorical(
        self,
        value: Any,
    ) -> tuple[str, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("transformations.datetime_categorical must be a sequence")
        normalized = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError(
                    "transformations.datetime_categorical entries must be strings"
                )
            name = item[3:] if item.startswith("dt_") else item
            if name not in self.datetime_features:
                raise ValueError(
                    "transformations.datetime_categorical must reference enabled "
                    f"datetime features; got {item!r}"
                )
            normalized.append(name)
        return tuple(dict.fromkeys(normalized))

    def _history_anchor_time(
        self,
        request: InformationSetRequest,
        target_time: pd.Timestamp,
    ) -> pd.Timestamp:
        """Resolve whether non-recursive historical features move with horizon."""
        direct = self.features.transformations.get("direct")
        if (
            not self.resolved_strategy.consumes_previous
            and isinstance(direct, Mapping)
            and direct.get("align_to_target") is False
        ):
            return cast(pd.Timestamp, pd.Timestamp(request.forecast_origin))
        return target_time

    def _validate_runtime_transformations(self) -> None:
        normalize_feature_scaling(
            self.features.transformations.get("feature_scaling", {})
        )
        normalize_target_transformations(
            self.features.transformations.get("target", {})
        )

    @staticmethod
    def _validate_visibility_proofs(
        proofs: Sequence[VisibilityProof],
        forecast_origin: pd.Timestamp,
    ) -> None:
        for proof in proofs:
            if proof.available_at is None:
                raise ValueError(
                    f"visibility proof for {proof.feature_name!r} requires available_at"
                )
            if proof.available_at > forecast_origin:
                raise ValueError(
                    f"feature {proof.feature_name!r} is available after forecast_origin"
                )

    @staticmethod
    def _normalize_horizon_steps(
        horizon_steps: Sequence[int] | None,
        horizon: int,
    ) -> tuple[int, ...]:
        if horizon_steps is None:
            return tuple(range(horizon))
        if isinstance(horizon_steps, (str, bytes)) or not isinstance(horizon_steps, Sequence):
            raise TypeError("horizon_steps must be a sequence of one-based integers")
        normalized = []
        for step in horizon_steps:
            if isinstance(step, bool) or not isinstance(step, int):
                raise TypeError("horizon_steps entries must be integers")
            if step <= 0 or step > horizon:
                raise ValueError(f"horizon_steps entries must be in [1, {horizon}]")
            normalized.append(step - 1)
        if len(normalized) != len(set(normalized)):
            raise ValueError("horizon_steps must not contain duplicates")
        return tuple(normalized)

    @staticmethod
    def _normalize_providers(
        providers: Mapping[Any, EndogenousFutureProvider] | None,
    ) -> dict[Any, EndogenousFutureProvider]:
        normalized = dict(providers or {})
        for identity, provider in normalized.items():
            if not isinstance(provider, EndogenousFutureProvider):
                raise TypeError(
                    f"future provider for identity {identity!r} does not satisfy the provider contract"
                )
        return normalized

    def _request_identities(self, request: InformationSetRequest) -> tuple[Any, ...]:
        if not self.problem.series_id_cols:
            if request.series_ids:
                raise ValueError("local forecasting problem forbids request.series_ids")
            return ((),)
        if not request.series_ids:
            raise ValueError("global forecasting problem requires request.series_ids")
        return request.series_ids

    def _identity_payload(self, identity: Any) -> dict[str, Any]:
        columns = self.problem.series_id_cols
        if not columns:
            return {}
        if len(columns) == 1:
            value = identity[0] if isinstance(identity, tuple) else identity
            return {columns[0]: value}
        if not isinstance(identity, tuple) or len(identity) != len(columns):
            raise ValueError(
                f"series identity {identity!r} must have width {len(columns)}"
            )
        return dict(zip(columns, identity))

    def _compile_lag_mapping(
        self,
        *,
        row: dict[str, Any],
        proofs: list[VisibilityProof],
        role: ColumnRole,
        lag_mapping: Mapping[str, tuple[int, ...]],
        identity: Any,
        target_time: pd.Timestamp,
        source_anchor_time: pd.Timestamp,
        step_index: int,
        request: InformationSetRequest,
        information_set: MaterializedInformationSet,
        providers: Mapping[Any, EndogenousFutureProvider],
    ) -> None:
        for column_name, lags in lag_mapping.items():
            source = self._source_for_column(column_name, role)
            for lag in lags:
                source_time = source_anchor_time - lag * pd.tseries.frequencies.to_offset(
                    self.problem.freq
                )
                feature_name = f"{column_name}__lag_{lag}"
                provider_name = None
                if source_time <= request.forecast_origin:
                    value = self._history_value(
                        source,
                        role,
                        column_name,
                        identity,
                        source_time,
                        information_set,
                    )
                else:
                    if role is ColumnRole.TARGET:
                        if not self.resolved_strategy.consumes_previous:
                            raise ValueError(
                                f"{self.resolved_strategy.name.value} cannot consume future target "
                                f"{column_name!r} at {source_time}"
                            )
                        provider = providers.get(identity)
                        if provider is None:
                            raise ValueError(
                                f"future target provider is required for {column_name!r} "
                                f"and series {identity!r}"
                            )
                    else:
                        provider = providers.get(identity)
                        if provider is None:
                            raise ValueError(
                                f"observed_past provider is required for {column_name!r} "
                                f"and series {identity!r}"
                            )
                    future_step = self._future_step(request, source_time)
                    value = provider.value_at(column_name, future_step)
                    provider_name = (
                        provider.provider_name(column_name)
                        if hasattr(provider, "provider_name")
                        else type(provider).__name__
                    )
                row[feature_name] = value
                available_at = (
                    self._history_available_at(
                        source,
                        identity,
                        source_time,
                        information_set,
                    )
                    if provider_name is None
                    else self._provider_available_at(
                        provider,
                        column_name,
                        future_step,
                        request.forecast_origin,
                    )
                )
                proofs.append(
                    VisibilityProof(
                        feature_name=feature_name,
                        source_name=source.name,
                        role=role.value,
                        target_time=target_time,
                        source_time=cast(pd.Timestamp, pd.Timestamp(source_time)),
                        forecast_origin=request.forecast_origin,
                        horizon_step=step_index + 1,
                        provider=provider_name,
                        available_at=available_at,
                    )
                )

    def _compile_known_future(
        self,
        row: dict[str, Any],
        proofs: list[VisibilityProof],
        identity: Any,
        target_time: pd.Timestamp,
        step_index: int,
        request: InformationSetRequest,
        information_set: MaterializedInformationSet,
    ) -> None:
        frames = information_set.known_future
        for source in self.data.sources:
            columns = [
                column
                for column in source.columns
                if column.role is ColumnRole.KNOWN_FUTURE
            ]
            if not columns:
                continue
            if not source.series_id_cols:
                try:
                    frames, _lookup = information_set.row_position_lookup(source.name)
                except KeyError:
                    pass
            frame = frames.get(source.name)
            if frame is None:
                raise ValueError(f"known_future source {source.name!r} was not materialized")
            source_row = self._exact_temporal_row(
                source, frame, identity, target_time, information_set
            )
            available_at_col = (
                source.available_at_col
                if source.availability is AvailabilityPolicy.COLUMN
                else (
                    "available_at"
                    if source.availability is AvailabilityPolicy.GENERATOR_DEFINED
                    else None
                )
            )
            for column in columns:
                row[column.name] = source_row[column.name]
                proofs.append(
                    VisibilityProof(
                        feature_name=column.name,
                        source_name=source.name,
                        role=ColumnRole.KNOWN_FUTURE.value,
                        target_time=target_time,
                        source_time=target_time,
                        forecast_origin=request.forecast_origin,
                        horizon_step=step_index + 1,
                        available_at=(
                            pd.Timestamp(source_row[available_at_col])
                            if available_at_col is not None
                            else request.forecast_origin
                        ),
                    )
                )

    def _compile_static(
        self,
        row: dict[str, Any],
        proofs: list[VisibilityProof],
        identity: Any,
        target_time: pd.Timestamp,
        step_index: int,
        request: InformationSetRequest,
        information_set: MaterializedInformationSet,
    ) -> None:
        frames = information_set.static
        for source in self.data.sources:
            columns = [
                column for column in source.columns if column.role is ColumnRole.STATIC
            ]
            if not columns:
                continue
            frame = frames.get(source.name)
            if frame is None:
                raise ValueError(f"static source {source.name!r} was not materialized")
            selected = self._filter_identity(source, frame, identity)
            if len(selected) != 1:
                raise ValueError(
                    f"static source {source.name!r} requires exactly one row for {identity!r}"
                )
            source_row = selected.iloc[0]
            for column in columns:
                row[column.name] = source_row[column.name]
                proofs.append(
                    VisibilityProof(
                        feature_name=column.name,
                        source_name=source.name,
                        role=ColumnRole.STATIC.value,
                        target_time=target_time,
                        source_time=None,
                        forecast_origin=request.forecast_origin,
                        horizon_step=step_index + 1,
                        available_at=request.forecast_origin,
                    )
                )

    def _compile_datetime(
        self,
        row: dict[str, Any],
        proofs: list[VisibilityProof],
        target_time: pd.Timestamp,
        step_index: int,
        request: InformationSetRequest,
    ) -> None:
        for name in self.datetime_features:
            feature_name = f"dt_{name}"
            row[feature_name] = self._DATETIME_FEATURES[name](target_time)
            proofs.append(
                VisibilityProof(
                    feature_name=feature_name,
                    source_name="calendar",
                    role=ColumnRole.KNOWN_FUTURE.value,
                    target_time=target_time,
                    source_time=target_time,
                    forecast_origin=request.forecast_origin,
                    horizon_step=step_index + 1,
                    available_at=request.forecast_origin,
                )
            )

    def _compile_transformations(
        self,
        row: dict[str, Any],
        proofs: list[VisibilityProof],
        identity: Any,
        target_time: pd.Timestamp,
        step_index: int,
        request: InformationSetRequest,
        information_set: MaterializedInformationSet,
    ) -> None:
        transformations = self.features.transformations
        unknown = sorted(set(transformations) - self._TRANSFORMATION_KEYS)
        if unknown:
            raise ValueError(f"unsupported feature transformations: {unknown}")
        self._compile_direct_transformations(row, transformations.get("direct"))
        advanced = transformations.get("advanced", {})
        if not isinstance(advanced, Mapping):
            raise TypeError("transformations.advanced must be a mapping")
        self._compile_history_transformations(
            row,
            advanced,
            identity,
            request,
            information_set,
        )
        self._compile_cyclical(row, advanced.get("cyclical"))
        self._compile_interaction_spec(row, advanced.get("interaction"))
        self._compile_polynomial(row, advanced.get("polynomial"))
        self._compile_named_interactions(row, transformations.get("interactions", {}))

        existing = {proof.feature_name for proof in proofs}
        for feature_name in row:
            if feature_name in existing or feature_name in {
                *self.problem.series_id_cols,
                "target_time",
                "horizon_step",
            }:
                continue
            proofs.append(
                VisibilityProof(
                    feature_name=feature_name,
                    source_name="derived_visible_features",
                    role="derived",
                    target_time=target_time,
                    source_time=request.forecast_origin,
                    forecast_origin=request.forecast_origin,
                    horizon_step=step_index + 1,
                    available_at=request.forecast_origin,
                )
            )

    def _compile_direct_transformations(
        self,
        row: dict[str, Any],
        direct: Any,
    ) -> None:
        if direct is None:
            return
        if not isinstance(direct, Mapping):
            raise TypeError("transformations.direct must be a mapping")
        layout = direct.get("layout")
        if layout not in {"independent_models", "single_model_horizon"}:
            raise ValueError(f"unsupported direct layout: {layout!r}")
        if layout != "single_model_horizon":
            return
        horizon_feature = direct.get("horizon_feature", {})
        if not isinstance(horizon_feature, Mapping):
            raise TypeError("transformations.direct.horizon_feature must be a mapping")
        name = str(horizon_feature.get("name", "forecast_horizon_idx"))
        value = float(row["horizon_step"])
        row[name] = value
        if bool(horizon_feature.get("cyclical", False)):
            period = float(self.problem.horizon)
            row[f"{name}_sin"] = float(np.sin(2.0 * np.pi * value / period))
            row[f"{name}_cos"] = float(np.cos(2.0 * np.pi * value / period))

    def _compile_history_transformations(
        self,
        row: dict[str, Any],
        advanced: Mapping[str, Any],
        identity: Any,
        request: InformationSetRequest,
        information_set: MaterializedInformationSet,
    ) -> None:
        supported = {
            "rolling",
            "expanding",
            "difference",
            "percent_change",
            "time_since",
            "cyclical",
            "interaction",
            "polynomial",
        }
        unknown = sorted(set(advanced) - supported)
        if unknown:
            raise ValueError(f"unsupported advanced transformations: {unknown}")
        for kind in ("rolling", "expanding", "difference", "percent_change", "time_since"):
            spec = advanced.get(kind)
            if spec is None:
                continue
            if not isinstance(spec, Mapping):
                raise TypeError(f"transformations.advanced.{kind} must be a mapping")
            columns = self._string_sequence(spec.get("columns", ()), f"advanced.{kind}.columns")
            for column in columns:
                history = self._visible_history_series(
                    column,
                    identity,
                    request,
                    information_set,
                )
                if kind == "rolling":
                    windows = self._positive_int_sequence(spec.get("windows", ()), "rolling.windows")
                    stats = self._string_sequence(spec.get("stats", ()), "rolling.stats")
                    for window in windows:
                        values = history.iloc[-window:]
                        for stat in stats:
                            row[f"{column}_rolling_{stat}_{window}"] = self._statistic(values, stat)
                elif kind == "expanding":
                    stats = self._string_sequence(spec.get("stats", ()), "expanding.stats")
                    for stat in stats:
                        row[f"{column}_expanding_{stat}"] = self._statistic(history, stat)
                elif kind == "difference":
                    for period in self._positive_int_sequence(spec.get("periods", ()), "difference.periods"):
                        if len(history) <= period:
                            raise ValueError(f"{column!r} has insufficient visible history for diff {period}")
                        row[f"{column}_diff_{period}"] = float(history.iloc[-1] - history.iloc[-period - 1])
                elif kind == "percent_change":
                    for period in self._positive_int_sequence(spec.get("periods", ()), "percent_change.periods"):
                        if len(history) <= period:
                            raise ValueError(
                                f"{column!r} has insufficient visible history for percent_change {period}"
                            )
                        previous = float(history.iloc[-period - 1])
                        if previous == 0.0:
                            raise ValueError(f"{column!r} percent_change denominator is zero")
                        row[f"{column}_pct_change_{period}"] = float(history.iloc[-1] / previous - 1.0)
                else:
                    events = self._string_sequence(spec.get("events", ()), "time_since.events")
                    for event in events:
                        row[f"{column}_time_since_{event}"] = self._time_since(history, event)

    def _compile_cyclical(self, row: dict[str, Any], spec: Any) -> None:
        if spec is None:
            return
        if not isinstance(spec, Mapping):
            raise TypeError("transformations.advanced.cyclical must be a mapping")
        period = spec.get("period")
        if isinstance(period, bool) or not isinstance(period, (int, float)) or period <= 0:
            raise ValueError("cyclical.period must be positive")
        for configured in self._string_sequence(spec.get("columns", ()), "cyclical.columns"):
            column = configured if configured in row else f"dt_{configured}"
            if column not in row:
                raise ValueError(f"cyclical references unknown feature: {configured!r}")
            value = float(row[column])
            row[f"{column}_sin"] = float(np.sin(2.0 * np.pi * value / float(period)))
            row[f"{column}_cos"] = float(np.cos(2.0 * np.pi * value / float(period)))

    def _compile_interaction_spec(self, row: dict[str, Any], spec: Any) -> None:
        if spec is None:
            return
        if not isinstance(spec, Mapping):
            raise TypeError("transformations.advanced.interaction must be a mapping")
        pairs = spec.get("column_pairs", ())
        if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
            raise TypeError("interaction.column_pairs must be a sequence")
        operations = set(self._string_sequence(spec.get("operations", ()), "interaction.operations"))
        for pair in pairs:
            if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
                raise TypeError("interaction column pairs must contain two feature names")
            left, right = pair
            if left not in row or right not in row:
                raise ValueError(f"interaction references unknown features: {pair}")
            left_value = float(row[left])
            right_value = float(row[right])
            if "add" in operations:
                row[f"{left}_add_{right}"] = left_value + right_value
            if "subtract" in operations:
                row[f"{left}_substract_{right}"] = left_value - right_value
            if "multiply" in operations:
                row[f"{left}_multiply_{right}"] = left_value * right_value
            if "divide" in operations:
                row[f"{left}_divide_{right}"] = left_value / (right_value + 1e-8)

    def _compile_polynomial(self, row: dict[str, Any], spec: Any) -> None:
        if spec is None:
            return
        if not isinstance(spec, Mapping):
            raise TypeError("transformations.advanced.polynomial must be a mapping")
        degree = spec.get("degree", 2)
        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 2:
            raise ValueError("polynomial.degree must be an integer >= 2")
        for column in self._string_sequence(spec.get("columns", ()), "polynomial.columns"):
            if column not in row:
                raise ValueError(f"polynomial references unknown feature: {column!r}")
            for current_degree in range(2, degree + 1):
                row[f"{column}_pow_{current_degree}"] = float(row[column]) ** current_degree

    def _compile_named_interactions(self, row: dict[str, Any], interactions: Any) -> None:
        if not isinstance(interactions, Mapping):
            raise TypeError("transformations.interactions must be a mapping")
        for name, members in interactions.items():
            if isinstance(members, (str, bytes)) or not isinstance(members, Sequence):
                raise TypeError(f"interaction {name!r} must list feature names")
            if len(members) < 2:
                raise ValueError(f"interaction {name!r} requires at least two features")
            missing = [member for member in members if member not in row]
            if missing:
                raise ValueError(f"interaction {name!r} references unknown features: {missing}")
            values = np.asarray([row[member] for member in members], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"interaction {name!r} requires finite numeric features")
            row[str(name)] = float(np.prod(values))

    def _visible_history_series(
        self,
        column_name: str,
        identity: Any,
        request: InformationSetRequest,
        information_set: MaterializedInformationSet,
    ) -> pd.Series:
        matches = [
            (source, column.role)
            for source in self.data.sources
            for column in source.columns
            if column.name == column_name
            and column.role in {ColumnRole.TARGET, ColumnRole.OBSERVED_PAST}
        ]
        if len(matches) != 1:
            raise ValueError(
                f"advanced history column {column_name!r} must resolve to one visible history source"
            )
        source, role = matches[0]
        frames = (
            information_set.target_history
            if role is ColumnRole.TARGET
            else information_set.observed_past
        )
        frame = frames.get(source.name)
        if frame is None:
            raise ValueError(f"history source {source.name!r} was not materialized")
        selected = self._filter_identity(source, frame, identity)
        if source.time_col is None:
            raise ValueError(f"history source {source.name!r} has no time_col")
        selected = selected.loc[
            pd.to_datetime(selected[source.time_col]) <= request.forecast_origin
        ].sort_values(source.time_col, kind="stable")
        if selected.empty:
            raise ValueError(f"history column {column_name!r} has no visible values")
        numeric = pd.to_numeric(selected[column_name], errors="raise").astype(float)
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"history column {column_name!r} must be finite")
        return numeric.reset_index(drop=True)

    @staticmethod
    def _string_sequence(value: Any, field_name: str) -> tuple[str, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{field_name} must be a sequence of strings")
        normalized = tuple(str(item) for item in value)
        if any(not item or item != item.strip() for item in normalized):
            raise ValueError(f"{field_name} entries must be non-blank")
        return normalized

    @staticmethod
    def _positive_int_sequence(value: Any, field_name: str) -> tuple[int, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{field_name} must be a sequence of positive integers")
        normalized = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise ValueError(f"{field_name} entries must be positive integers")
            normalized.append(item)
        return tuple(normalized)

    @staticmethod
    def _statistic(values: pd.Series, stat: str) -> float:
        supported = {"mean", "std", "min", "max", "median", "skew", "kurt"}
        if stat not in supported:
            raise ValueError(f"unsupported history statistic: {stat!r}")
        result = getattr(values, stat)()
        if pd.isna(result):
            # 理论上 minimum_history_rows 门禁已保证窗口足够，此处只是防御
            # 回退（窗口 < window 时 pandas 返回 NaN）；分层约束下本包不可
            # 依赖 utils.log_util，用标准库 warnings 显式告警避免静默。
            warnings.warn(
                f"history statistic {stat!r} produced NaN "
                f"(sample size {len(values)}); falling back to 0.0",
                RuntimeWarning,
                stacklevel=2,
            )
            result = 0.0
        return float(result)

    @staticmethod
    def _time_since(values: pd.Series, event: str) -> float:
        if event == "peak":
            mask = (values.shift(1) < values) & (values > values.shift(-1))
        elif event == "trough":
            mask = (values.shift(1) > values) & (values < values.shift(-1))
        else:
            raise ValueError(f"unsupported time-since event: {event!r}")
        indices = np.flatnonzero(mask.to_numpy())
        position = len(values) - 1
        prior = indices[indices < position]
        return float(position - prior[-1] if len(prior) else position)

    def _source_for_column(
        self,
        column_name: str,
        role: ColumnRole,
    ) -> DataSourceSpec:
        matches = [
            source
            for source in self.data.sources
            if any(
                column.name == column_name and column.role is role
                for column in source.columns
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                f"column {column_name!r} must resolve to exactly one {role.value} source"
            )
        return matches[0]

    def _history_value(
        self,
        source: DataSourceSpec,
        role: ColumnRole,
        column_name: str,
        identity: Any,
        source_time: pd.Timestamp,
        information_set: MaterializedInformationSet,
    ) -> Any:
        use_cache = not source.series_id_cols
        if use_cache:
            # 方案 A：无键 source 直接复用 lookup 登记帧，避免 property 每次
            # 访问的 deep copy（同 origin 的 16 个 call 共享同一份帧）。
            try:
                frames, _lookup = information_set.row_position_lookup(source.name)
            except KeyError:
                frames = (
                    information_set.target_history
                    if role is ColumnRole.TARGET
                    else information_set.observed_past
                )
        else:
            frames = (
                information_set.target_history
                if role is ColumnRole.TARGET
                else information_set.observed_past
            )
        frame = frames.get(source.name)
        if frame is None:
            raise ValueError(f"history source {source.name!r} was not materialized")
        return self._exact_temporal_row(
            source, frame, identity, source_time, information_set
        )[column_name]

    def _history_available_at(
        self,
        source: DataSourceSpec,
        identity: Any,
        source_time: pd.Timestamp,
        information_set: MaterializedInformationSet,
    ) -> pd.Timestamp:
        if source.availability is AvailabilityPolicy.SOURCE_TIME:
            return pd.Timestamp(source_time)
        frame = (
            information_set.target_history
            if any(column.role is ColumnRole.TARGET for column in source.columns)
            else information_set.observed_past
        )[source.name]
        row = self._exact_temporal_row(
            source, frame, identity, source_time, information_set
        )
        if source.availability is AvailabilityPolicy.COLUMN:
            return pd.Timestamp(row[source.available_at_col])
        if source.availability is AvailabilityPolicy.GENERATOR_DEFINED:
            return pd.Timestamp(row["available_at"])
        return pd.Timestamp(source_time)

    @staticmethod
    def _provider_available_at(
        provider: EndogenousFutureProvider,
        column_name: str,
        future_step: int,
        forecast_origin: pd.Timestamp,
    ) -> pd.Timestamp:
        metadata = getattr(provider, "available_at", None)
        available_at = (
            pd.Timestamp(metadata(column_name, future_step))
            if callable(metadata)
            else pd.Timestamp(forecast_origin)
        )
        if available_at > forecast_origin:
            raise ValueError(
                f"provider value for {column_name!r} is available after forecast_origin"
            )
        return available_at

    def _exact_temporal_row(
        self,
        source: DataSourceSpec,
        frame: pd.DataFrame,
        identity: Any,
        source_time: pd.Timestamp,
        information_set: MaterializedInformationSet | None = None,
    ) -> pd.Series:
        if source.time_col is None:
            raise ValueError(f"temporal source {source.name!r} has no time_col")
        selected = self._filter_identity(source, frame, identity)
        # 性能（2026-08-30 方案 A）：无 series 键的 source（local 单序列主场景），
        # 时间→行位置映射登记在 information_set 上（帧的所有者，不可变），同一
        # origin 的全部查询共享一次解析；有键 source 的筛选语义与位置映射耦合，
        # 保持原逐次精确匹配路径（正确性优先）。语义均不变：非恰好一行即 RAISE。
        use_cache = information_set is not None and not source.series_id_cols
        if use_cache:
            try:
                _frames, time_lookup = information_set.row_position_lookup(
                    source.name
                )
            except KeyError:
                information_set.register_row_position_lookup(
                    source.name,
                    {source.name: frame},
                    source.time_col,
                )
                _frames, time_lookup = information_set.row_position_lookup(
                    source.name
                )
            position = time_lookup.get(pd.Timestamp(source_time).value, -1)
        else:
            parsed = pd.DatetimeIndex(
                pd.to_datetime(selected[source.time_col].to_numpy())
            )
            matches = np.flatnonzero(parsed.asi8 == pd.Timestamp(source_time).value)
            position = int(matches[0]) if len(matches) == 1 else -1
        if position < 0 or position >= len(selected):
            raise ValueError(
                f"source {source.name!r} requires exactly one row at {source_time} "
                f"for series {identity!r}"
            )
        return selected.iloc[position]

    @staticmethod
    def _filter_identity(
        source: DataSourceSpec,
        frame: pd.DataFrame,
        identity: Any,
    ) -> pd.DataFrame:
        if not source.series_id_cols:
            return frame
        if identity == ():
            identities = frame.loc[:, list(source.series_id_cols)].drop_duplicates()
            if len(identities) != 1:
                raise ValueError(
                    f"keyed source {source.name!r} requires exactly one identity for local compilation"
                )
            return frame
        if len(source.series_id_cols) == 1:
            value = identity[0] if isinstance(identity, tuple) else identity
            return frame.loc[frame[source.series_id_cols[0]] == value]
        if not isinstance(identity, tuple) or len(identity) != len(source.series_id_cols):
            raise ValueError(
                f"series identity {identity!r} must have width {len(source.series_id_cols)}"
            )
        mask = np.ones(len(frame), dtype=bool)
        for column, value in zip(source.series_id_cols, identity):
            mask &= frame[column].to_numpy() == value
        return frame.loc[mask]

    @staticmethod
    def _future_step(
        request: InformationSetRequest,
        source_time: pd.Timestamp,
    ) -> int:
        matches = np.flatnonzero(request.forecast_times == pd.Timestamp(source_time))
        if len(matches) != 1:
            raise ValueError(
                f"future provider source_time {source_time} is outside the forecast grid"
            )
        return int(matches[0])


__all__ = [
    "CompiledFeatures",
    "FeatureCompiler",
    "FeatureSchema",
    "VisibilityProof",
]
