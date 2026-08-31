"""Canonical information-set design and backtest geometry services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from data_loading import EndogenousFutureProvider, InformationSetRequest, SourceRegistry
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
from model_training.strategies import StrategyTargetPlan, TargetCoordinate


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


class _RegistryDesignBuilder:
    def __init__(self, config: ForecastConfigSpec, registry: SourceRegistry) -> None:
        self.config = config
        self.registry = registry
        self.compiler = FeatureCompiler(config)
        self.offset = pd.tseries.frequencies.to_offset(config.problem.freq)
        self.plan = StrategyTargetPlan.from_spec(
            config.strategy,
            config.problem.targets,
            config.problem.horizon,
        )
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

    def request(self, origin: pd.Timestamp, *, mode: str | None = None) -> InformationSetRequest:
        return InformationSetRequest(
            forecast_origin=origin,
            forecast_times=pd.date_range(
                origin,
                periods=self.config.problem.horizon + 1,
                freq=self.config.problem.freq,
            )[1:],
            series_ids=self.series_ids if self.is_global else (),
            information_mode=mode or self.config.problem.information_mode,
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
        request = self.request(origin, mode="oracle")
        information_set = self.registry.materialize(request)
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
                target_values = []
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
                for step_index, target_time in enumerate(request.forecast_times):
                    position = time_lookup.get(
                        pd.Timestamp(target_time).value, -1
                    )
                    if position < 0 or position >= len(series_frame):
                        raise ValueError(
                            f"target label {target!r} requires one row for "
                            f"series {series_id!r} at {target_time}"
                        )
                    value = float(series_frame.iloc[position][target])
                    values[series_index, step_index, target_index] = value
                    target_values.append(value)
                trajectories[series_id][target] = tuple(target_values)
        return values, trajectories

    def _compile_call(
        self,
        origin: pd.Timestamp,
        call_index: int,
        information_set,
        *,
        target_providers: Mapping[Any, EndogenousFutureProvider] | None,
        visibility_cutoff: pd.Timestamp | None = None,
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
        frame = compiled.frame.loc[:, list(schema)]
        design = frame.to_numpy(copy=True)
        self._audit.append(compiled)
        return design

    def training_row(
        self,
        origin: pd.Timestamp,
    ) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
        training_request = self.request(origin, mode="oracle")
        information_set = self.registry.materialize(training_request)
        target_values, trajectories = self._labels(origin)
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
            )
            for call_index in range(len(self.plan.call_coordinates))
        )
        return designs, target_values if self.is_global else target_values[0]

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
    rows = [builder.training_row(candidate) for candidate in candidate_origins]
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
