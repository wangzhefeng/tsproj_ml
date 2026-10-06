"""Visibility-driven feature compilation for canonical forecasting specs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import field
from typing import Any, cast

import numpy as np
import pandas as pd
from feature_engineering.kernels.seasonal import (
    normalize_same_slot_spec, normalize_recent_state_spec,
    same_slot_stats, recent_state_stats,
    normalize_seasonal_baseline_spec,
)

from data_loading import EndogenousFutureProvider, InformationSetRequest, MaterializedInformationSet
from forecasting_core.specs import (
    AvailabilityPolicy,
    ColumnRole,
    DataSourceSpec,
    ForecastConfigSpec,
)
from feature_engineering.kernels.spectral import (
    fourier_features,
    normalize_band_periods,
    wavelet_energy_features,
)
from feature_engineering.kernels.history import history_statistic, time_since_event
from feature_engineering.transform_specs import (
    normalize_feature_scaling,
    normalize_target_transformations,
)
from feature_engineering.statistics.provider import HistoryStatisticsProvider
from feature_engineering.compilation.planning import planned_feature_names
from feature_engineering.kernels.windows import window_features








from feature_engineering.compilation.contracts import (
    FeatureSchema, BatchEligibility, VisibilityProof, CompiledFeatures, CompilationContext, ProofMode,
)
from feature_engineering.compilation.batch import BatchExecutor






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
        {"seasonal_baseline"} |
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
        self.planned_feature_names = planned_feature_names(config)
        self._context: ContextVar[CompilationContext | None] = ContextVar(
            "feature_compilation_context", default=None,
        )
        self.last_batch_stage_wall_seconds: dict[str, float] = {
            "batch_prepare": 0.0,
            "batch_feature_columns": 0.0,
            "finish_and_proof_validation": 0.0,
        }

    @contextmanager
    def _compilation_scope(
        self,
        information_set: MaterializedInformationSet | None = None,
        *,
        frames: dict[str, dict[str, Any]] | None = None,
    ) -> Iterator[CompilationContext]:
        if frames is None:
            frames = {} if information_set is None else {
                ColumnRole.TARGET.value: information_set.target_history,
                ColumnRole.OBSERVED_PAST.value: information_set.observed_past,
                ColumnRole.KNOWN_FUTURE.value: information_set.known_future,
            }
        context = CompilationContext(frames)
        token = self._context.set(context)
        try:
            yield context
        finally:
            self._context.reset(token)

    def _active_context(self) -> CompilationContext:
        context = self._context.get()
        if context is None:
            raise RuntimeError("feature compilation requires an active scope")
        return context

    def _role_frames(
        self,
        information_set: MaterializedInformationSet,
        role: ColumnRole,
    ) -> dict[str, Any]:
        """返回 compile 作用域内的角色帧（避免逐行 property 访问触发 deep copy）。"""
        frames = self._active_context().frames.get(role.value)
        if frames is None:
            raise RuntimeError(
                "compile-scope frames not primed; call compile() entry first"
            )
        return frames

    def compile(
        self,
        information_set: MaterializedInformationSet,
        request: InformationSetRequest,
        *,
        target_future_providers: Mapping[Any, EndogenousFutureProvider] | None = None,
        observed_future_providers: Mapping[Any, EndogenousFutureProvider] | None = None,
        horizon_steps: Sequence[int] | None = None,
        visibility_cutoff: pd.Timestamp | None = None,
        statistics_provider: HistoryStatisticsProvider | None = None,
    ) -> CompiledFeatures:
        if not isinstance(information_set, MaterializedInformationSet):
            raise TypeError("information_set must be a MaterializedInformationSet")
        if not isinstance(request, InformationSetRequest):
            raise TypeError("request must be an InformationSetRequest")
        with self._compilation_scope(information_set) as context:
            if statistics_provider is not None:
                if (self.problem.training_scope != "local" or self.problem.series_id_cols
                        or len(self.data.sources) != 1
                        or self.data.sources[0].availability != AvailabilityPolicy.SOURCE_TIME):
                    raise ValueError("statistics provider requires Local single source_time history")
                statistics_provider.require_binding(self.config.fingerprint(), request.forecast_origin)
                context.statistics_provider = statistics_provider
            if request.H != self.problem.horizon:
                raise ValueError(
                    "information-set horizon does not match ForecastProblemSpec: "
                    f"request={request.H}, problem={self.problem.horizon}"
                )
            if request.target_access != "history_only":
                raise ValueError("feature compilation requires target_access='history_only'")

            target_providers = self._normalize_providers(target_future_providers)
            observed_providers = self._normalize_providers(observed_future_providers)
            identities = self._request_identities(request)
            selected_steps = self._normalize_horizon_steps(horizon_steps, request.H)
            rows: list[dict[str, Any]] = []
            proofs: list[VisibilityProof] = []

            for identity in identities:
                for step_index in selected_steps:
                    target_time = cast(pd.Timestamp, request.forecast_times[step_index])
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

            frame = self._ordered_frame(pd.DataFrame(rows))
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
                source_lineage=(*information_set.lineage, statistics_provider.source_lineage)
                    if statistics_provider is not None else information_set.lineage,
                visibility_proof=proofs,
            )

    def _ordered_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        keys = (*self.problem.series_id_cols, "target_time", "horizon_step")
        expected = (*keys, *self.planned_feature_names)
        if not frame.columns.is_unique or set(frame.columns) != set(expected):
            raise ValueError("compiled feature schema differs from planned names")
        return frame.loc[:, list(expected)]

    def compile_batch(
        self,
        information_sets: Sequence[MaterializedInformationSet],
        requests: Sequence[InformationSetRequest],
        *,
        horizon_steps: Sequence[int] | None = None,
        visibility_cutoffs: Sequence[pd.Timestamp] | None = None,
        proof_mode: ProofMode = "materialize",
    ) -> tuple[CompiledFeatures, ...]:
        return BatchExecutor(self).compile_batch(
            information_sets, requests, horizon_steps=horizon_steps,
            visibility_cutoffs=visibility_cutoffs, proof_mode=proof_mode,
        )

    def batch_eligibility(
        self,
        requests: Sequence[InformationSetRequest],
        *,
        horizon_steps: Sequence[int] | None = None,
    ) -> BatchEligibility:
        """Resolve batch eligibility using the compiler's actual semantics."""
        selected_steps = tuple(
            self._normalize_horizon_steps(horizon_steps, request.H)
            for request in requests
        )
        return self._batch_eligibility(requests, selected_steps)

    def _batch_eligibility(
        self,
        requests: Sequence[InformationSetRequest],
        selected_steps: Sequence[tuple[int, ...]],
    ) -> BatchEligibility:
        reason_codes: list[str] = []
        trigger_fields: set[str] = set()
        advanced = self.features.transformations.get("advanced", {})
        for kind in ("rolling", "expanding", "difference", "fourier", "wavelet", "same_slot", "recent_state"):
            for column in advanced.get(kind, {}).get("columns", ()):
                source = next((source for source in self.data.sources
                               if any(spec.name == column for spec in source.columns)), None)
                if source is not None and source.availability is not AvailabilityPolicy.SOURCE_TIME:
                    if "origin_sensitive_history" not in reason_codes:
                        reason_codes.append("origin_sensitive_history")
                    trigger_fields.add(f"advanced.{kind}:{column}")
        for role, lag_mapping in (
            (ColumnRole.TARGET, self.features.target_lags),
            (ColumnRole.OBSERVED_PAST, self.features.observed_past_lags),
        ):
            if not lag_mapping:
                continue
            for request, steps in zip(requests, selected_steps):
                target_times = request.forecast_times.take(list(steps))
                anchors = pd.DatetimeIndex(
                    [
                        self._history_anchor_time(request, pd.Timestamp(target_time))
                        for target_time in target_times
                    ]
                )
                for column, lags in lag_mapping.items():
                    for lag in lags:
                        source_times = anchors - lag * pd.tseries.frequencies.to_offset(
                            self.problem.freq
                        )
                        if bool((source_times > request.forecast_origin).any()):
                            if "provider_dependent_lag" not in reason_codes:
                                reason_codes.append("provider_dependent_lag")
                            trigger_fields.add(f"{role.value}:{column}:lag={lag}")
        call_count = len(selected_steps[0]) if selected_steps else 0
        return BatchEligibility(
            eligible=not reason_codes,
            reason_codes=tuple(reason_codes),
            trigger_fields=tuple(sorted(trigger_fields)),
            origin_count=len(requests),
            call_count=call_count,
            estimated_origin_call_count=sum(len(steps) for steps in selected_steps),
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
        advanced = self.features.transformations.get("advanced", {})
        block = advanced.get("block_weather")
        if block is not None:
            if not isinstance(block, Mapping) or set(block) != {"columns", "stats"}:
                raise ValueError("block_weather requires exactly columns/stats")
            for field in ("columns", "stats"):
                sequence = block[field]
                if isinstance(sequence, str) or not isinstance(sequence, Sequence) or not sequence or len(sequence) != len(set(sequence)):
                    raise ValueError(f"block_weather.{field} requires a nonempty unique sequence")
            known = {c.name for s in self.data.sources for c in s.columns if c.role is ColumnRole.KNOWN_FUTURE and not c.categorical}
            if set(block["columns"]) - known or set(block["stats"]) - {"mean", "min", "max"}:
                raise ValueError("block_weather requires numeric known_future columns and mean/min/max")
        for name, normalize in (("same_slot", normalize_same_slot_spec), ("recent_state", normalize_recent_state_spec)):
            if name in advanced:
                spec = normalize(advanced[name])
                for column in spec["columns"]:
                    matches = [c for s in self.data.sources for c in s.columns
                               if c.name == column and c.role in {ColumnRole.TARGET, ColumnRole.OBSERVED_PAST}]
                    if len(matches) != 1:
                        raise ValueError(f"{name} requires a unique history column: {column}")
                if name == "same_slot" and self.problem.horizon > spec["period"]:
                    raise ValueError("same_slot horizon must not exceed period")
        baseline = self.features.transformations.get("seasonal_baseline")
        if baseline is not None:
            spec = normalize_seasonal_baseline_spec(baseline)
            if spec["column"] not in self.problem.targets or self.problem.horizon > spec["period"]:
                raise ValueError("seasonal_baseline requires a target column and horizon <= period")
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
        frames = self._role_frames(information_set, ColumnRole.KNOWN_FUTURE)
        for source in self.data.sources:
            columns = [
                column
                for column in source.columns
                if column.role is ColumnRole.KNOWN_FUTURE
            ]
            if not columns:
                continue
            frame = frames.get(source.name)
            if frame is None:
                raise ValueError(f"known_future source {source.name!r} was not materialized")
            source_row = self._exact_temporal_row(
                source,
                frame,
                identity,
                target_time,
                information_set,
                cache_key=self._row_cache_key(source, ColumnRole.KNOWN_FUTURE),
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
        for kind, normalize in (("same_slot", normalize_same_slot_spec), ("recent_state", normalize_recent_state_spec)):
            if kind not in advanced:
                continue
            spec = normalize(advanced[kind])
            for column in spec["columns"]:
                source = next(s for s in self.data.sources if any(c.name == column for c in s.columns))
                role = next(c.role for c in source.columns if c.name == column)
                frame = self._filter_identity(source, self._role_frames(information_set, role)[source.name], identity)
                history = pd.Series(frame[column].to_numpy(), index=pd.DatetimeIndex(frame[source.time_col]))
                row.update(self._causal_statistics(kind, spec, column, history, request.forecast_origin, target_time))
        if "block_weather" in advanced:
            values, available_at = self._block_weather_values(identity, step_index + 1, request, information_set)
            row.update(values)
            for name in values:
                proofs.append(VisibilityProof(feature_name=name, source_name="block_known_future",
                    role="known_future", target_time=target_time, source_time=target_time,
                    forecast_origin=request.forecast_origin, horizon_step=step_index + 1, available_at=available_at))
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
        *,
        vectorized: bool = False,
    ) -> None:
        horizon_feature = self._direct_horizon_feature(direct)
        if horizon_feature is None:
            return
        name = str(horizon_feature.get("name", "forecast_horizon_idx"))
        value = np.asarray(row["horizon_step"], dtype=float) if vectorized else float(row["horizon_step"])
        row[name] = value
        if bool(horizon_feature.get("cyclical", False)):
            period = float(self.problem.horizon)
            sine = np.sin(2.0 * np.pi * value / period)
            cosine = np.cos(2.0 * np.pi * value / period)
            row[f"{name}_sin"] = sine if vectorized else float(sine)
            row[f"{name}_cos"] = cosine if vectorized else float(cosine)

    @staticmethod
    def _direct_horizon_feature(direct: Any) -> Mapping[str, Any] | None:
        if direct is None:
            return None
        if not isinstance(direct, Mapping):
            raise TypeError("transformations.direct must be a mapping")
        layout = direct.get("layout")
        if layout not in {"independent_models", "single_model_horizon"}:
            raise ValueError(f"unsupported direct layout: {layout!r}")
        if layout != "single_model_horizon":
            return None
        horizon_feature = direct.get("horizon_feature", {})
        if not isinstance(horizon_feature, Mapping):
            raise TypeError("transformations.direct.horizon_feature must be a mapping")
        enabled = horizon_feature.get("enabled", True)
        if not isinstance(enabled, bool):
            raise TypeError("transformations.direct.horizon_feature.enabled must be a boolean")
        if not enabled:
            return None
        return horizon_feature

    @staticmethod
    def _validate_advanced_transformations(advanced: Mapping[str, Any]) -> None:
        supported = {
            "same_slot",
            "recent_state",
            "block_weather",
            "rolling_quantile",
            "lagged_rolling",
            "rolling",
            "expanding",
            "difference",
            "percent_change",
            "time_since",
            "ewm",
            "fourier",
            "wavelet",
            "cyclical",
            "interaction",
            "polynomial",
        }
        unknown = sorted(set(advanced) - supported)
        if unknown:
            raise ValueError(f"unsupported advanced transformations: {unknown}")

    def _block_weather_values(self, identity, horizon_step, request, information_set):
        spec = self.features.transformations["advanced"]["block_weather"]
        width = self.resolved_strategy.steps_per_call
        start = ((horizon_step - 1) // width) * width
        # 同一信息集内按原点、序列和块复用；single 入口和每个 batch item 都重置作用域。
        cache = self._active_context().auxiliary.setdefault("block_weather", {})
        cache_key = (request.forecast_origin, identity, start)
        if cache_key in cache:
            return cache[cache_key]
        values = {name: [] for name in spec["columns"]}
        available = []
        for index in range(start, min(start + width, request.H)):
            row, proofs = {}, []
            self._compile_known_future(row, proofs, identity, request.forecast_times[index], index, request, information_set)
            for name in values:
                values[name].append(float(row[name]))
                available.extend(p.available_at for p in proofs if p.feature_name == name)
        result = {}
        for name, sequence in values.items():
            if not np.isfinite(sequence).all():
                raise ValueError("block_weather must be finite over the entire block")
            for stat in spec["stats"]:
                result[f"{name}_blk_{stat}"] = float(getattr(np, stat)(sequence))
        # 仅缓存成功完成且可见性证据齐全的块，失败不留下半成品。
        cache[cache_key] = result, max(available)
        return cache[cache_key]

    @staticmethod
    def _causal_statistics(kind, spec, column, history, origin, anchor):
        if kind == "same_slot":
            stats = same_slot_stats(history, anchor=anchor, origin=origin, period=spec["period"], days=spec["days"])
            return {f"{column}_slot_{stat}_{days}d": stats[days][stat]
                    for days in spec["days"] for stat in spec["stats"]}
        stats = recent_state_stats(history, origin=origin, windows=spec["windows"])
        return {f"{column}_rs_{stat}_{window}": stats[window][stat]
                for window in spec["windows"] for stat in spec["stats"]}

    def _compile_history_transformations(
        self,
        row: dict[str, Any],
        advanced: Mapping[str, Any],
        identity: Any,
        request: InformationSetRequest,
        information_set: MaterializedInformationSet,
    ) -> None:
        self._validate_advanced_transformations(advanced)
        for kind in (
            "rolling_quantile",
            "lagged_rolling",
            "rolling",
            "expanding",
            "difference",
            "percent_change",
            "time_since",
            "ewm",
            "fourier",
            "wavelet",
        ):
            spec = advanced.get(kind)
            if spec is None:
                continue
            if not isinstance(spec, Mapping):
                raise TypeError(f"transformations.advanced.{kind} must be a mapping")
            columns = self._string_sequence(spec.get("columns", ()), f"advanced.{kind}.columns")
            for column in columns:
                provider = self._active_context().statistics_provider
                if provider is not None and kind in {"ewm", "expanding", "time_since"}:
                    if kind == "ewm":
                        for halflife in self._positive_number_sequence(spec.get("halflives", ()), "ewm.halflives"):
                            for stat in self._validated_stats(spec.get("stats", ()), "ewm.stats", self.EWM_STATS):
                                row[f"{column}_ewm_{stat}_{halflife}"] = provider.value(
                                    kind, column, stat, origin=request.forecast_origin, identity=identity, parameter=halflife)
                    else:
                        stats = (self._string_sequence(spec.get("events", ()), "time_since.events") if kind == "time_since"
                                 else self._validated_stats(spec.get("stats", ()), "expanding.stats", self.ROLLING_STATS))
                        for stat in stats:
                            row[f"{column}_{kind}_{stat}"] = provider.value(
                                kind, column, stat, origin=request.forecast_origin, identity=identity)
                    continue
                history = self._visible_history_series(
                    column,
                    identity,
                    request,
                    information_set,
                )
                if kind in {"rolling_quantile", "lagged_rolling"}:
                    row.update({f"{column}_{name}": value for name, value in window_features(history, kind, spec).items()})
                elif kind == "rolling":
                    windows = self._positive_int_sequence(spec.get("windows", ()), "rolling.windows")
                    stats = self._validated_stats(spec.get("stats", ()), "rolling.stats", self.ROLLING_STATS)
                    for window in windows:
                        if len(history) < window:
                            raise ValueError(f"{column!r} has insufficient visible history for rolling window {window}")
                        values = history.iloc[-window:]
                        for stat in stats:
                            row[f"{column}_rolling_{stat}_{window}"] = history_statistic(values, stat)
                elif kind == "ewm":
                    # 指数加权统计（近期行为权重更高）：按半衰期计。
                    # ewm 半衰期以样本步数计；与 rolling 窗口一样只消费可见历史。
                    halflives = self._positive_number_sequence(
                        spec.get("halflives", ()), "ewm.halflives"
                    )
                    ewm_stats = self._validated_stats(spec.get("stats", ()), "ewm.stats", self.EWM_STATS)
                    for halflife in halflives:
                        series = history.ewm(halflife=halflife, adjust=True)
                        for stat in ewm_stats:
                            if stat == "mean":
                                value = series.mean().iloc[-1]
                            elif stat == "std":
                                value = series.std().iloc[-1]
                            else:
                                raise ValueError(
                                    f"unsupported ewm statistic: {stat!r}; expected 'mean' or 'std'"
                                )
                            if pd.isna(value):
                                raise ValueError(
                                    f"{column!r} ewm_{stat} halflife={halflife} "
                                    "produced NaN (insufficient visible history)"
                                )
                            row[f"{column}_ewm_{stat}_{halflife}"] = float(value)
                elif kind == "expanding":
                    stats = self._validated_stats(spec.get("stats", ()), "expanding.stats", self.ROLLING_STATS)
                    for stat in stats:
                        row[f"{column}_expanding_{stat}"] = history_statistic(history, stat)
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
                elif kind == "fourier":
                    windows = self._positive_int_sequence(spec.get("windows", ()), "fourier.windows")
                    band_periods = normalize_band_periods(spec.get("band_periods"))
                    for window in windows:
                        if len(history) < window:
                            raise ValueError(
                                f"{column!r} has insufficient visible history "
                                f"for fourier window {window}"
                            )
                        features = fourier_features(
                            history.iloc[-window:].to_numpy(),
                            top_k=int(spec.get("top_k", 5)),
                            band_periods=band_periods,
                        )
                        for suffix, value in features.items():
                            row[f"{column}_fft_{suffix}_{window}"] = value
                elif kind == "wavelet":
                    windows = self._positive_int_sequence(spec.get("windows", ()), "wavelet.windows")
                    wavelet_level = spec.get("level", 3)
                    if (
                        isinstance(wavelet_level, bool)
                        or not isinstance(wavelet_level, int)
                        or wavelet_level < 1
                    ):
                        raise ValueError("wavelet.level must be a positive integer")
                    for window in windows:
                        if len(history) < window:
                            raise ValueError(
                                f"{column!r} has insufficient visible history "
                                f"for wavelet window {window}"
                            )
                        features = wavelet_energy_features(
                            history.iloc[-window:].to_numpy(),
                            wavelet=str(spec.get("wavelet", "db4")),
                            level=wavelet_level,
                        )
                        for suffix, value in features.items():
                            row[f"{column}_wavelet_energy_{suffix}_{window}"] = value
                else:
                    events = self._string_sequence(spec.get("events", ()), "time_since.events")
                    for event in events:
                        row[f"{column}_time_since_{event}"] = time_since_event(history, event)

    def _compile_cyclical(
        self, row: dict[str, Any], spec: Any, *, vectorized: bool = False,
    ) -> None:
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
            value = np.asarray(row[column], dtype=float) if vectorized else float(row[column])
            sine = np.sin(2.0 * np.pi * value / float(period))
            cosine = np.cos(2.0 * np.pi * value / float(period))
            row[f"{column}_sin"] = sine if vectorized else float(sine)
            row[f"{column}_cos"] = cosine if vectorized else float(cosine)

    def _compile_interaction_spec(
        self, row: dict[str, Any], spec: Any, *, vectorized: bool = False,
    ) -> None:
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
            left_value = np.asarray(row[left], dtype=float) if vectorized else float(row[left])
            right_value = np.asarray(row[right], dtype=float) if vectorized else float(row[right])
            if "add" in operations:
                row[f"{left}_add_{right}"] = left_value + right_value
            if "subtract" in operations:
                row[f"{left}_substract_{right}"] = left_value - right_value
            if "multiply" in operations:
                row[f"{left}_multiply_{right}"] = left_value * right_value
            if "divide" in operations:
                if np.any(np.asarray(right_value) == 0):
                    raise ValueError("interaction divide denominator is zero")
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    result = left_value / right_value
                if not np.isfinite(result).all():
                    raise ValueError("interaction divide must produce finite values")
                row[f"{left}_divide_{right}"] = result

    def _compile_polynomial(
        self, row: dict[str, Any], spec: Any, *, vectorized: bool = False,
    ) -> None:
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
            # 保留 Python float 与 NumPy 数组各自的幂运算，不统一数值后端。
            value = np.asarray(row[column], dtype=float) if vectorized else float(row[column])
            for current_degree in range(2, degree + 1):
                row[f"{column}_pow_{current_degree}"] = value ** current_degree

    def _compile_named_interactions(
        self, row: dict[str, Any], interactions: Any, *, vectorized: bool = False,
    ) -> None:
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
            values = (
                np.column_stack([np.asarray(row[member], dtype=float) for member in members])
                if vectorized
                else np.asarray([row[member] for member in members], dtype=float)
            )
            if not np.isfinite(values).all():
                raise ValueError(f"interaction {name!r} requires finite numeric features")
            row[str(name)] = np.prod(values, axis=1) if vectorized else float(np.prod(values))

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
        frames = self._role_frames(information_set, role)
        frame = frames.get(source.name)
        if frame is None:
            raise ValueError(f"history source {source.name!r} was not materialized")
        # 缓存只在本次 compile 有效；同源不同列可共享行定位，但不同序列
        # 必须隔离。使用结构化身份键，避免将 identity 字符串化造成碰撞。
        histories = self._active_context().auxiliary.setdefault("visible_history", {})
        cache_key = (source.name, source.time_col, identity)
        cached = histories.get(cache_key)
        if cached is None:
            selected = self._filter_identity(source, frame, identity)
            if source.time_col is None:
                raise ValueError(f"history source {source.name!r} has no time_col")
            parsed = pd.DatetimeIndex(
                pd.to_datetime(selected[source.time_col].to_numpy())
            )
            order = np.argsort(parsed.asi8, kind="stable")
            parsed = parsed[order]
            histories[cache_key] = (parsed, order, selected)
        else:
            parsed, order, selected = cached
        # searchsorted：origin 之前的可见前缀（排序后），替代全列布尔掩码
        cutoff_ns = pd.Timestamp(request.forecast_origin).value
        visible_count = int(np.searchsorted(parsed.asi8, cutoff_ns, side="right"))
        ordered = selected.iloc[order[:visible_count]]
        if ordered.empty:
            raise ValueError(f"history column {column_name!r} has no visible values")
        numeric = pd.to_numeric(ordered[column_name], errors="raise").astype(float)
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"history column {column_name!r} must be finite")
        return numeric.reset_index(drop=True)

    # 解析期 stats 白名单：rolling/expanding 共用 history_statistic 的全集，
    # ewm 只支持 mean/std。拼错统计名必须在 spec 解析期 RAISE，而不是
    # 编译中段（与 preflight「未知参数 RAISE」同精神）。
    ROLLING_STATS = frozenset(
        {"mean", "std", "min", "max", "median", "skew", "kurt", "entropy",
         "max_diff", "min_diff"}
    )
    EWM_STATS = frozenset({"mean", "std"})

    @classmethod
    def _validated_stats(
        cls, value: Any, field_name: str, allowed: frozenset[str],
    ) -> tuple[str, ...]:
        stats = cls._string_sequence(value, field_name)
        unknown = sorted(set(stats) - allowed)
        if unknown:
            raise ValueError(
                f"unsupported {field_name} entries {unknown}; "
                f"expected subset of {sorted(allowed)}"
            )
        return stats

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
    def _positive_number_sequence(value: Any, field_name: str) -> tuple[float, ...]:
        """正数序列（整数或浮点，ewm 半衰期用）。"""
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{field_name} must be a sequence of positive numbers")
        normalized = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(f"{field_name} entries must be positive numbers")
            if not np.isfinite(float(item)) or float(item) <= 0:
                raise ValueError(f"{field_name} entries must be positive finite numbers")
            normalized.append(float(item))
        return tuple(normalized)



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
        frames = self._role_frames(information_set, role)
        frame = frames.get(source.name)
        if frame is None:
            raise ValueError(f"history source {source.name!r} was not materialized")
        return self._exact_temporal_row(
            source,
            frame,
            identity,
            source_time,
            information_set,
            cache_key=self._row_cache_key(source, role),
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
        role = (
            ColumnRole.TARGET
            if any(column.role is ColumnRole.TARGET for column in source.columns)
            else ColumnRole.OBSERVED_PAST
        )
        frame = self._role_frames(information_set, role)[source.name]
        row = self._exact_temporal_row(
            source,
            frame,
            identity,
            source_time,
            information_set,
            cache_key=self._row_cache_key(source, role),
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
        *,
        cache_key: str | None = None,
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
            resolved_cache_key = cache_key or source.name
            try:
                _frames, time_lookup = information_set.row_position_lookup(
                    resolved_cache_key
                )
            except KeyError:
                information_set.register_row_position_lookup(
                    resolved_cache_key,
                    {resolved_cache_key: frame},
                    source.time_col,
                )
                _frames, time_lookup = information_set.row_position_lookup(
                    resolved_cache_key
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
    def _row_cache_key(source: DataSourceSpec, role: ColumnRole) -> str:
        return f"{role.value}:{source.name}"

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
