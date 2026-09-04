"""Canonical information-set design and backtest geometry services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from data_loading import (
    EndogenousFutureProvider,
    InformationSetRequest,
    MaterializedInformationSet,
    SourceRegistry,
    TargetAccess,
)
from feature_engineering import CompiledFeatures, FeatureCompiler
from forecasting_core.specs import (
    CalendarMonthBacktestSpec,
    ColumnRole,
    FixedStepBacktestSpec,
    ForecastConfigSpec,
)
from forecasting_core.tensors import PointForecastTensor
from model_forecasting.transforms import CanonicalTargetTransform
from model_testing import validation
from model_testing.backtest import actual_tensor as _actual_tensor
from model_training.strategies import (
    TargetCoordinate,
    target_plan_for_config,
)


class _PredictedTargetProvider:
    def __init__(
        self,
        horizon: int,
        predicted: Mapping[TargetCoordinate, np.ndarray],
        allowed_dependencies: tuple[TargetCoordinate, ...],
        forecast_origin: pd.Timestamp,
        freq: str,
        series_id: Any = "__local__",
        series_index: int = 0,
        target_transform: CanonicalTargetTransform | None = None,
    ) -> None:
        self._horizon = horizon
        self._predicted = predicted
        self._allowed_dependencies = frozenset(allowed_dependencies)
        self._origin = forecast_origin
        self._offset = pd.tseries.frequencies.to_offset(freq)
        self._series_id = series_id
        self._series_index = series_index
        self._target_transform = target_transform

    @property
    def horizon(self) -> int:
        return self._horizon

    def value_at(self, name: str, step: int) -> float:
        coordinate = TargetCoordinate(name, step + 1)
        if coordinate not in self._allowed_dependencies:
            raise ValueError(
                f"{coordinate!r} is not an allowed prior dependency for this forecast call"
            )
        try:
            value = float(self._predicted[coordinate][self._series_index])
        except KeyError as exc:
            raise ValueError(
                f"prediction dependency is unavailable for {name!r} at horizon {step + 1}"
            ) from exc
        if self._target_transform is None:
            return value
        restored = self._target_transform.restore_values(
            self._series_id,
            name,
            [value],
            [self._origin + (step + 1) * self._offset],
        )
        return float(restored[0])

    def available_at(self, _name: str, _step: int) -> pd.Timestamp:
        return self._origin


class _OracleTargetProvider:
    def __init__(
        self,
        horizon: int,
        trajectories: Mapping[str, tuple[float, ...]],
        allowed_dependencies: tuple[TargetCoordinate, ...],
        forecast_origin: pd.Timestamp,
    ) -> None:
        self._horizon = horizon
        self._trajectories = trajectories
        self._allowed_dependencies = frozenset(allowed_dependencies)
        self._origin = forecast_origin

    @property
    def horizon(self) -> int:
        return self._horizon

    def value_at(self, name: str, step: int) -> float:
        coordinate = TargetCoordinate(name, step + 1)
        if coordinate not in self._allowed_dependencies:
            raise ValueError(
                f"{coordinate!r} is not an allowed prior dependency for this training call"
            )
        try:
            return float(self._trajectories[name][step])
        except (KeyError, IndexError) as exc:
            raise ValueError(f"oracle dependency is unavailable for {coordinate!r}") from exc

    def available_at(self, _name: str, _step: int) -> pd.Timestamp:
        return self._origin


@dataclass(frozen=True, slots=True)
class TrainingDesignProbe:
    """一个真实训练 origin 的 canonical 设计编译摘要。"""

    origin: pd.Timestamp
    design_shapes: tuple[tuple[int, ...], ...]
    target_shape: tuple[int, ...]
    feature_names: tuple[str, ...]


def _split_batch_designs(
    frame: pd.DataFrame,
    *,
    schema: Sequence[str],
    call_steps: Sequence[int],
    n_series: int,
) -> tuple[np.ndarray, ...]:
    """按 compiler 的 identity-major / call-minor 行序拆分设计矩阵。"""
    normalized_steps = tuple(int(step) for step in call_steps)
    expected_rows = n_series * len(normalized_steps)
    if len(frame) != expected_rows:
        raise ValueError(
            "batch compilation returned an unexpected row count per strategy call"
        )
    actual_steps = frame["horizon_step"].to_numpy(copy=False)
    expected_steps = np.tile(np.asarray(normalized_steps), n_series)
    if not np.array_equal(actual_steps, expected_steps):
        raise ValueError(
            "batch compilation violated identity-major / call-minor row order"
        )
    matrix = frame.loc[:, list(schema)].to_numpy(copy=True)
    shaped = matrix.reshape(n_series, len(normalized_steps), matrix.shape[1])
    return tuple(shaped[:, call_index, :] for call_index in range(len(normalized_steps)))


class _RegistryDesignBuilder:
    def __init__(self, config: ForecastConfigSpec, registry: SourceRegistry) -> None:
        self.config = config
        self.registry = registry
        self.compiler = FeatureCompiler(config)
        self.offset = pd.tseries.frequencies.to_offset(config.problem.freq)
        self.plan = target_plan_for_config(config)
        self.feature_schema: tuple[str, ...] = ()
        self.categorical_schema: tuple[str, ...] = ()
        self._audit: list[CompiledFeatures] = []
        self.series_ids = self._resolve_series_ids()

    @property
    def n_series(self) -> int:
        return len(self.series_ids)

    @property
    def is_global(self) -> bool:
        return self.config.problem.training_scope == "global"

    def _provider_key(self, series_id: Any) -> Any:
        return series_id if self.is_global else ()

    def _training_scope_validation(self) -> Mapping[str, Any]:
        value = self.config.validation.get("training_scope", {})
        if not isinstance(value, Mapping):
            raise TypeError("validation.training_scope must be a mapping")
        return value

    def _normalize_identity(self, value: Any) -> Any:
        width = len(self.config.problem.series_id_cols)
        if width == 1:
            if isinstance(value, (tuple, list)):
                if len(value) != 1:
                    raise ValueError("series identity must have width 1")
                return value[0]
            return value
        if not isinstance(value, (tuple, list)) or len(value) != width:
            raise ValueError(f"series identity must have width {width}")
        return tuple(value)

    @staticmethod
    def _source_identities(source, frame: pd.DataFrame) -> tuple[Any, ...]:
        if len(source.series_id_cols) == 1:
            values = frame[source.series_id_cols[0]].tolist()
        else:
            values = list(
                frame.loc[:, list(source.series_id_cols)].itertuples(
                    index=False,
                    name=None,
                )
            )
        return tuple(dict.fromkeys(values))

    def _target_source_frames(self) -> tuple[tuple[Any, pd.DataFrame], ...]:
        frames = []
        for source in self.config.data.sources:
            if not any(column.role is ColumnRole.TARGET for column in source.columns):
                continue
            if source.source_type != "file" or source.history_path is None:
                raise ValueError(
                    "global generated target sources require explicit "
                    "validation.training_scope.series_order"
                )
            frame = self.registry._validate_frame(
                source,
                self.registry._read_path(source.history_path),
                generated=False,
                path_version="history",
            )
            frames.append((source, frame))
        if not frames:
            raise ValueError("canonical runtime requires target history")
        return tuple(frames)

    def _resolve_series_ids(self) -> tuple[Any, ...]:
        if not self.is_global:
            return ("__local__",)
        validation = self._training_scope_validation()
        unknown_policy = str(
            validation.get("unknown_series_policy", "raise")
        ).lower()
        if unknown_policy != "raise":
            raise ValueError("global unknown_series_policy must be raise")
        configured = validation.get("series_order")
        target_frames = self._target_source_frames()
        discovered_by_source = tuple(
            self._source_identities(source, frame) for source, frame in target_frames
        )
        if configured is None:
            series_ids = discovered_by_source[0]
        else:
            if isinstance(configured, (str, bytes)) or not isinstance(
                configured, (tuple, list)
            ):
                raise TypeError(
                    "validation.training_scope.series_order must be a sequence"
                )
            series_ids = tuple(self._normalize_identity(value) for value in configured)
        if not series_ids or len(series_ids) != len(set(series_ids)):
            raise ValueError("series order must contain unique series identities")
        expected = set(series_ids)
        for identities in discovered_by_source:
            unknown = [identity for identity in identities if identity not in expected]
            if unknown:
                raise ValueError(f"target source contains unknown series: {unknown}")
        policy = str(validation.get("incomplete_series_policy", "raise")).lower()
        if policy not in {"raise", "drop"}:
            raise ValueError("incomplete_series_policy must be raise or drop")
        expected_times_by_source = tuple(
            pd.DatetimeIndex(pd.to_datetime(frame[source.time_col]).unique()).sort_values()
            for source, frame in target_frames
        )
        reference_times = expected_times_by_source[0]
        if any(
            not reference_times.equals(times)
            for times in expected_times_by_source[1:]
        ):
            raise ValueError("target history sources must share the same timestamps")
        complete = []
        for series_id in series_ids:
            series_complete = True
            for (source, frame), expected_times in zip(
                target_frames,
                expected_times_by_source,
            ):
                selected = self.registry._filter_identity(source, frame, series_id)
                times = pd.DatetimeIndex(pd.to_datetime(selected[source.time_col])).sort_values()
                if not times.equals(expected_times):
                    series_complete = False
                    break
            if series_complete:
                complete.append(series_id)
            elif policy == "raise":
                raise ValueError(f"series {series_id!r} is incomplete")
        if not complete:
            raise ValueError("global training has no complete series")
        return tuple(complete)

    def request(
        self,
        origin: pd.Timestamp,
        *,
        target_access: TargetAccess = "history_only",
    ) -> InformationSetRequest:
        return InformationSetRequest(
            forecast_origin=origin,
            forecast_times=pd.date_range(
                origin,
                periods=self.config.problem.horizon + 1,
                freq=self.config.problem.freq,
            )[1:],
            series_ids=self.series_ids if self.is_global else (),
            target_access=target_access,
        )

    def reset_audit(self) -> None:
        self._audit = []

    @property
    def audit(self) -> tuple[CompiledFeatures, ...]:
        return tuple(self._audit)

    def target_history_times(self, origin: pd.Timestamp) -> pd.DatetimeIndex:
        information_set = self.registry.materialize(self.request(origin))
        indices = []
        for source in self.config.data.sources:
            if not any(column.role is ColumnRole.TARGET for column in source.columns):
                continue
            frame = information_set.target_history[source.name]
            if self.is_global:
                frame = self.registry._filter_identity(source, frame, self.series_ids[0])
            indices.append(pd.DatetimeIndex(pd.to_datetime(frame[source.time_col])))
        if not indices:
            raise ValueError("canonical runtime requires target history")
        reference = indices[0]
        if any(not reference.equals(index) for index in indices[1:]):
            raise ValueError("target history sources must share the same timestamps")
        return reference

    def target_history(self, origin: pd.Timestamp) -> PointForecastTensor:
        information_set = self.registry.materialize(self.request(origin))
        target_frames: dict[tuple[Any, str], pd.Series] = {}
        reference: pd.DatetimeIndex | None = None
        for target in self.config.problem.targets:
            source = next(
                source
                for source in self.config.data.sources
                if any(
                    column.name == target and column.role is ColumnRole.TARGET
                    for column in source.columns
                )
            )
            frame = information_set.target_history[source.name]
            for series_id in self.series_ids:
                selected = (
                    self.registry._filter_identity(source, frame, series_id)
                    if self.is_global
                    else frame
                )
                times = pd.DatetimeIndex(pd.to_datetime(selected[source.time_col]))
                if reference is None:
                    reference = times
                elif not reference.equals(times):
                    raise ValueError("target history sources must share the same timestamps")
                target_frames[(series_id, target)] = pd.Series(
                    pd.to_numeric(selected[target], errors="raise").to_numpy(dtype=float),
                    index=times,
                )
        if reference is None:
            raise ValueError("canonical runtime requires target history")
        values = np.stack(
            [
                np.column_stack(
                    [
                        target_frames[(series_id, target)].reindex(reference).to_numpy()
                        for target in self.config.problem.targets
                    ]
                )
                for series_id in self.series_ids
            ],
            axis=0,
        )
        if not np.isfinite(values).all():
            raise ValueError("canonical target history must be finite")
        return PointForecastTensor(
            values=values,
            series_ids=self.series_ids,
            forecast_times=reference,
            targets=self.config.problem.targets,
        )

    def _labels(
        self,
        origin: pd.Timestamp,
    ) -> tuple[np.ndarray, dict[Any, dict[str, tuple[float, ...]]]]:
        request = self.request(origin, target_access="supervised_labels")
        information_set = self.registry.materialize(request)
        return self._labels_from_information_set(request, information_set)

    def _labels_from_information_set(
        self,
        request: InformationSetRequest,
        information_set: MaterializedInformationSet,
    ) -> tuple[np.ndarray, dict[Any, dict[str, tuple[float, ...]]]]:
        values = np.empty(
            (
                self.n_series,
                self.config.problem.horizon,
                len(self.config.problem.targets),
            ),
            dtype=float,
        )
        trajectories: dict[Any, dict[str, tuple[float, ...]]] = {
            series_id: {} for series_id in self.series_ids
        }
        for target_index, target in enumerate(self.config.problem.targets):
            source = next(
                source
                for source in self.config.data.sources
                if any(
                    column.name == target and column.role is ColumnRole.TARGET
                    for column in source.columns
                )
            )
            frame = information_set.target_history[source.name]
            for series_index, series_id in enumerate(self.series_ids):
                series_frame = (
                    self.registry._filter_identity(source, frame, series_id)
                    if self.is_global
                    else frame
                )
                # 性能（2026-08-30 方案 A）：与特征编译共用 information_set 的
                # 时间→行位置映射；未注册时登记一次。语义不变：非恰好一行即 RAISE。
                try:
                    _frames, time_lookup = information_set.row_position_lookup(
                        source.name
                    )
                except KeyError:
                    information_set.register_row_position_lookup(
                        source.name,
                        {source.name: series_frame},
                        source.time_col,
                    )
                    _frames, time_lookup = information_set.row_position_lookup(
                        source.name
                    )
                positions = np.fromiter(
                    (
                        time_lookup.get(int(timestamp_ns), -1)
                        for timestamp_ns in request.forecast_times.asi8
                    ),
                    dtype=np.int64,
                    count=len(request.forecast_times),
                )
                invalid = (positions < 0) | (positions >= len(series_frame))
                if bool(invalid.any()):
                    missing = int(np.flatnonzero(invalid)[0])
                    raise ValueError(
                        f"target label {target!r} requires one row for "
                        f"series {series_id!r} at {request.forecast_times[missing]}"
                    )
                target_column = np.asarray(
                    pd.to_numeric(series_frame[target], errors="raise"),
                    dtype=float,
                )
                selected_values = target_column.take(positions)
                values[series_index, :, target_index] = selected_values
                trajectories[series_id][target] = tuple(selected_values.tolist())
        return values, trajectories

    def _compile_call(
        self,
        origin: pd.Timestamp,
        call_index: int,
        information_set,
        *,
        target_providers: Mapping[Any, EndogenousFutureProvider] | None,
        visibility_cutoff: pd.Timestamp | None = None,
        collect_audit: bool = True,
    ) -> np.ndarray:
        request = self.request(origin)
        call_step = self.plan.call_coordinates[call_index][0].horizon_step
        compiled = self.compiler.compile(
            information_set,
            request,
            target_future_providers=target_providers,
            observed_future_providers=information_set.observed_future_providers,
            horizon_steps=(call_step,),
            visibility_cutoff=visibility_cutoff,
        )
        schema = self._update_feature_schema(compiled)
        frame = compiled.frame.loc[:, list(schema)]
        design = frame.to_numpy(copy=True)
        if collect_audit:
            self._audit.append(compiled)
        return design

    def _update_feature_schema(
        self,
        compiled: CompiledFeatures,
    ) -> tuple[str, ...]:
        schema = (
            (*self.config.problem.series_id_cols, *compiled.schema.feature_names)
            if self.is_global
            else compiled.schema.feature_names
        )
        if not self.feature_schema:
            self.feature_schema = schema
            self.categorical_schema = tuple(
                dict.fromkeys(
                    (
                        *self.config.problem.series_id_cols,
                        *compiled.schema.categorical_names,
                    )
                )
            ) if self.is_global else compiled.schema.categorical_names
        elif schema != self.feature_schema:
            raise ValueError("compiled feature schema changed across forecast origins")
        return schema

    def training_row(
        self,
        origin: pd.Timestamp,
    ) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
        training_request = self.request(origin, target_access="supervised_labels")
        information_set = self.registry.materialize(training_request)
        target_values, trajectories = self._labels_from_information_set(
            training_request,
            information_set,
        )
        visibility_cutoff = pd.Timestamp(training_request.forecast_times[-1])
        designs = tuple(
            self._compile_call(
                origin,
                call_index,
                information_set,
                target_providers={
                    self._provider_key(series_id): _OracleTargetProvider(
                        self.config.problem.horizon,
                        trajectories[series_id],
                        self.plan.dependencies[call_index],
                        origin,
                    )
                    for series_id in self.series_ids
                },
                visibility_cutoff=visibility_cutoff,
                collect_audit=False,
            )
            for call_index in range(len(self.plan.call_coordinates))
        )
        return designs, target_values if self.is_global else target_values[0]

    def training_rows(
        self,
        origins: Sequence[pd.Timestamp],
    ) -> tuple[tuple[tuple[np.ndarray, ...], np.ndarray], ...]:
        """批量编译一组监督 origins；provider 依赖策略保持逐行路径。"""
        normalized_origins = tuple(pd.Timestamp(origin) for origin in origins)
        if not normalized_origins:
            return ()
        call_steps = tuple(
            coordinates[0].horizon_step
            for coordinates in self.plan.call_coordinates
        )
        if not self._can_batch_training(call_steps):
            return tuple(self.training_row(origin) for origin in normalized_origins)

        training_requests = tuple(
            self.request(origin, target_access="supervised_labels")
            for origin in normalized_origins
        )
        union_forecast_times = pd.DatetimeIndex(
            np.unique(
                np.concatenate(
                    [request.forecast_times.asi8 for request in training_requests]
                )
            )
        )
        materialization_request = InformationSetRequest(
            forecast_origin=max(normalized_origins),
            forecast_times=union_forecast_times,
            series_ids=self.series_ids if self.is_global else (),
            target_access="supervised_labels",
        )
        shared_information_set = self.registry.materialize(materialization_request)
        information_sets = (shared_information_set,) * len(training_requests)
        target_values = tuple(
            self._labels_from_information_set(request, information_set)[0]
            for request, information_set in zip(
                training_requests,
                information_sets,
            )
        )
        compile_requests = tuple(
            self.request(origin) for origin in normalized_origins
        )
        compiled_items = self.compiler.compile_batch(
            information_sets,
            compile_requests,
            horizon_steps=call_steps,
            visibility_cutoffs=tuple(
                pd.Timestamp(request.forecast_times[-1])
                for request in training_requests
            ),
            proof_mode="validate_only",
        )
        rows = []
        for compiled, values in zip(compiled_items, target_values):
            schema = self._update_feature_schema(compiled)
            designs = _split_batch_designs(
                compiled.frame,
                schema=schema,
                call_steps=call_steps,
                n_series=self.n_series,
            )
            rows.append(
                (
                    designs,
                    values if self.is_global else values[0],
                )
            )
        return tuple(rows)

    def _can_batch_training(self, call_steps: Sequence[int]) -> bool:
        if self.config.strategy is None:
            return False
        advanced = self.config.features.transformations.get("advanced", {})
        if isinstance(advanced, Mapping) and any(
            advanced.get(kind) is not None
            for kind in ("percent_change", "time_since", "ewm")
        ):
            return False
        direct = self.config.features.transformations.get("direct")
        if isinstance(direct, Mapping) and direct.get("align_to_target") is False:
            return True
        max_step = max(call_steps)
        return all(
            lag >= max_step
            for lag_mapping in (
                self.config.features.target_lags,
                self.config.features.observed_past_lags,
            )
            for lags in lag_mapping.values()
            for lag in lags
        )

    def forecast_designs(
        self,
        origin: pd.Timestamp,
        *,
        target_transform: CanonicalTargetTransform | None = None,
    ):
        information_set = self.registry.materialize(self.request(origin))
        base_design = self._compile_call(
            origin,
            0,
            information_set,
            target_providers={
                self._provider_key(series_id): _PredictedTargetProvider(
                    self.config.problem.horizon,
                    {},
                    self.plan.dependencies[0],
                    origin,
                    self.config.problem.freq,
                    series_id,
                    series_index,
                    target_transform,
                )
                for series_index, series_id in enumerate(self.series_ids)
            },
        )

        def provider(call_index, _coordinates, dependencies, predicted):
            return self._compile_call(
                origin,
                call_index,
                information_set,
                target_providers={
                    self._provider_key(series_id): _PredictedTargetProvider(
                        self.config.problem.horizon,
                        predicted,
                        dependencies,
                        origin,
                        self.config.problem.freq,
                        series_id,
                        series_index,
                        target_transform,
                    )
                    for series_index, series_id in enumerate(self.series_ids)
                },
            )

        return (base_design,), provider


def probe_training_design(
    config: ForecastConfigSpec,
    registry: SourceRegistry,
    origin: pd.Timestamp,
) -> TrainingDesignProbe:
    """Compile one production-equivalent training row without fitting a model."""
    normalized_origin = pd.Timestamp(origin)
    if not isinstance(normalized_origin, pd.Timestamp):
        raise ValueError("training design probe origin must be a valid timestamp")
    builder = _RegistryDesignBuilder(config, registry)
    designs, targets = builder.training_row(normalized_origin)
    return TrainingDesignProbe(
        origin=normalized_origin,
        design_shapes=tuple(tuple(design.shape) for design in designs),
        target_shape=tuple(targets.shape),
        feature_names=builder.feature_schema,
    )


def minimum_history_rows(config: ForecastConfigSpec) -> int:
    """Return visible rows required before one supervised origin is valid."""
    configured_lags = tuple(
        lag
        for lag_mapping in (
            config.features.target_lags,
            config.features.observed_past_lags,
        )
        for lags in lag_mapping.values()
        for lag in lags
    )
    required = max((*configured_lags, 1))
    if config.strategy is None:
        raise ValueError("minimum_history_rows requires a single-model strategy")
    resolved_strategy = config.strategy.resolve(config.problem.horizon)
    direct = config.features.transformations.get("direct")
    if (
        configured_lags
        and not resolved_strategy.consumes_previous
        and isinstance(direct, Mapping)
        and direct.get("align_to_target") is False
    ):
        required = max(required, max(configured_lags) + 1)
    advanced = config.features.transformations.get("advanced", {})
    if not isinstance(advanced, Mapping):
        return required
    rolling = advanced.get("rolling")
    if isinstance(rolling, Mapping):
        windows = rolling.get("windows", ())
        if isinstance(windows, Sequence) and not isinstance(windows, (str, bytes)):
            required = max(required, *(int(value) for value in windows))
    for kind in ("difference", "percent_change"):
        spec = advanced.get(kind)
        if not isinstance(spec, Mapping):
            continue
        periods = spec.get("periods", ())
        if isinstance(periods, Sequence) and not isinstance(periods, (str, bytes)):
            required = max(required, *(int(value) + 1 for value in periods))
    return required


def _supervised_arrays(
    builder: _RegistryDesignBuilder,
    origin: pd.Timestamp,
) -> tuple[
    tuple[np.ndarray, ...],
    np.ndarray,
    tuple[pd.Timestamp, ...],
    tuple[pd.Timestamp, ...],
    tuple[Any, ...],
]:
    minimum_history = minimum_history_rows(builder.config)
    timestamps = builder.target_history_times(origin)
    origin_position = timestamps.get_loc(origin)
    if not isinstance(origin_position, (int, np.integer)):
        raise ValueError("forecast_origin must resolve to one unique row")
    available_origins = tuple(
        pd.Timestamp(value)
        for value in timestamps[
            minimum_history - 1 : origin_position - builder.config.problem.horizon + 1
        ]
    )
    backtest = builder.config.validation.backtest
    if isinstance(backtest, FixedStepBacktestSpec):
        candidate_origins = available_origins[-backtest.history_steps:]
    elif isinstance(backtest, CalendarMonthBacktestSpec):
        candidate_origins = available_origins
    else:
        candidate_origins = available_origins
    rows = []
    batch_size = 128
    for start in range(0, len(candidate_origins), batch_size):
        rows.extend(
            builder.training_rows(candidate_origins[start : start + batch_size])
        )
    if len(rows) < 2:
        raise ValueError("canonical runtime requires at least two complete supervised samples")
    designs_by_call = tuple(
        np.concatenate([row[0][call_index] for row in rows], axis=0)
        for call_index in range(len(builder.plan.call_coordinates))
    )
    targets = (
        np.concatenate([row[1] for row in rows], axis=0)
        if builder.is_global
        else np.stack([row[1] for row in rows], axis=0)
    )
    sample_origins = tuple(
        candidate
        for candidate in candidate_origins
        for _series_id in builder.series_ids
    )
    sample_series_ids = tuple(
        series_id
        for _candidate in candidate_origins
        for series_id in builder.series_ids
    )
    return (
        designs_by_call,
        targets,
        candidate_origins,
        sample_origins,
        sample_series_ids,
    )


def _sample_indices(
    origin_indices: tuple[int, ...],
    n_series: int,
) -> tuple[int, ...]:
    return tuple(
        origin_index * n_series + series_index
        for origin_index in origin_indices
        for series_index in range(n_series)
    )


def _label_start(
    builder: _RegistryDesignBuilder,
    origin: pd.Timestamp,
) -> pd.Timestamp:
    return validation.label_start(origin, builder.offset)


def _label_end(
    builder: _RegistryDesignBuilder,
    origin: pd.Timestamp,
) -> pd.Timestamp:
    return validation.label_end(origin, builder.offset, builder.config.problem.horizon)


def _holdout_training_indices(
    builder: _RegistryDesignBuilder,
    supervised_origins: tuple[pd.Timestamp, ...],
) -> tuple[tuple[int, ...], dict[str, Any]]:
    holdout_origin = supervised_origins[-1]
    geometry = validation.TimeGeometry(
        offset=builder.offset,
        horizon=builder.config.problem.horizon,
    )
    train_indices = tuple(
        index
        for index, candidate in enumerate(supervised_origins[:-1])
        if geometry.label_end(candidate) < geometry.label_start(holdout_origin)
    )
    metadata = validation.validate_no_overlap(
        supervised_origins,
        train_indices,
        holdout_origin,
        geometry,
    )
    return train_indices, metadata


@dataclass(frozen=True, slots=True)
class _BacktestWindow:
    window: int
    origin_index: int
    origin: pd.Timestamp
    train_indices: tuple[int, ...]
    metadata: dict[str, Any]


def _rolling_backtest_windows(
    builder: _RegistryDesignBuilder,
    supervised_origins: tuple[pd.Timestamp, ...],
) -> tuple[_BacktestWindow, ...]:
    backtest = builder.config.validation.backtest
    if not isinstance(backtest, FixedStepBacktestSpec):
        raise TypeError("fixed-step backtest requires FixedStepBacktestSpec")
    geometry = validation.TimeGeometry(
        offset=builder.offset,
        horizon=builder.config.problem.horizon,
    )
    folds = validation.rolling_origin_folds(
        supervised_origins,
        geometry,
        history_steps=backtest.history_steps,
        train_window_steps=backtest.train_window_steps,
        fold_count=backtest.fold_count,
        stride_steps=backtest.stride_steps,
    )
    return tuple(
        _BacktestWindow(
            window=fold.window,
            origin_index=fold.origin_index,
            origin=fold.origin,
            train_indices=fold.train_indices,
            metadata=fold.metadata,
        )
        for fold in folds
    )


def _actual_at_origin(
    config: ForecastConfigSpec,
    Y_all: np.ndarray,
    origin_index: int,
    forecast_times: pd.DatetimeIndex,
    series_ids: tuple[Any, ...],
) -> PointForecastTensor:
    n_series = len(series_ids)
    start = origin_index * n_series
    return _actual_tensor(
        config,
        Y_all[start : start + n_series],
        forecast_times,
        series_ids,
    )
