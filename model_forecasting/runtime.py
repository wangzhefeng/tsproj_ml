"""Executable canonical runtime without legacy configuration translation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from model_testing import validation
from data_loading import (
    EndogenousFutureProvider,
    InformationSetRequest,
    SourceRegistry,
)
from model_training.estimators import (
    make_model_factory,
    resolve_model_capabilities,
)
from feature_engineering import CompiledFeatures, FeatureCompiler
from model_evaluation.marginal import evaluate_marginal_distribution
from model_evaluation.point import (
    build_eval_mask_payload,
    evaluate_point_forecasts,
    resolve_aggregate_weighting,
)
from feature_engineering.selection import (
    CanonicalFeatureSelector,
    normalize_feature_selection,
    selected_indices_for_artifact,
)
from utils.log_util import logger
from model_forecasting.results import (
    backtest_tensors_to_long,
    write_backtest_results,
    write_forecast_results,
)
from model_forecasting.specs import ColumnRole, ForecastConfigSpec, TargetAdapter
from model_testing.backtest import (
    actual_tensor,
    positive_validation_int,
    resolve_origin,
    seasonal_naive_tensor,
)
from model_forecasting.forecaster import CanonicalForecaster
from model_training.strategies import (
    CanonicalStrategyArtifact,
    StrategyTargetPlan,
    TargetCoordinate,
)
from model_forecasting.tensors import PointForecastTensor
from model_forecasting.transforms import CanonicalFeatureScaler, CanonicalTargetTransform
from model_training.trainer import CanonicalTrainer
from models.ModelFactory import ModelFactory
from models.ModelSaveLoad import ModelDeployPkl
from probabilistic.pipeline import CanonicalMarginalQuantileForecaster
from probabilistic.training import CanonicalMarginalQuantileTrainer
from probabilistic.types import ForecastModelBundle, MarginalForecastDistribution

# P3/D3：回测原语已公开化至 model_testing/backtest.py；保留私有别名转发（历史调用点零改动）。
_positive_validation_int = positive_validation_int
_seasonal_naive_tensor = seasonal_naive_tensor
_actual_tensor = actual_tensor
_resolve_origin = resolve_origin


@dataclass(frozen=True, slots=True)
class CanonicalRuntimeResult:
    run_dir: Path
    model_dir: Path
    test_dir: Path
    forecast_dir: Path
    fingerprint: str
    bundle: ForecastModelBundle


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
    ) -> np.ndarray:
        request = self.request(origin)
        call_step = self.plan.call_coordinates[call_index][0].horizon_step
        compiled = self.compiler.compile(
            information_set,
            request,
            target_future_providers=target_providers,
            observed_future_providers=information_set.observed_future_providers,
            horizon_steps=(call_step,),
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
        information_set = self.registry.materialize(self.request(origin))
        target_values, trajectories = self._labels(origin)
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
    max_lag = max(
        (
            *(
                lag
                for lags in builder.config.features.target_lags.values()
                for lag in lags
            ),
            *(
                lag
                for lags in builder.config.features.observed_past_lags.values()
                for lag in lags
            ),
            1,
        )
    )
    timestamps = builder.target_history_times(origin)
    origin_position = timestamps.get_loc(origin)
    if not isinstance(origin_position, (int, np.integer)):
        raise ValueError("forecast_origin must resolve to one unique row")
    candidate_origins = tuple(
        pd.Timestamp(value)
        for value in timestamps[max_lag - 1 : origin_position - builder.config.problem.horizon + 1]
    )
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
    validation_cfg = builder.config.validation
    history_length = _positive_validation_int(
        validation_cfg,
        "history_length",
        len(supervised_origins),
    )
    window_length = _positive_validation_int(
        validation_cfg,
        "window_length",
        history_length,
    )
    max_windows = _positive_validation_int(validation_cfg, "max_test_windows", 1)
    stride = _positive_validation_int(
        validation_cfg,
        "test_window_stride",
        builder.config.problem.horizon,
    )
    geometry = validation.TimeGeometry(
        offset=builder.offset,
        horizon=builder.config.problem.horizon,
    )
    folds = validation.rolling_origin_folds(
        supervised_origins,
        geometry,
        history_length=history_length,
        window_length=window_length,
        max_windows=max_windows,
        stride=stride,
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


def _fit_runtime_transforms(
    config: ForecastConfigSpec,
    builder: _RegistryDesignBuilder,
    X_by_call: tuple[np.ndarray, ...],
    Y: np.ndarray,
    origins: tuple[pd.Timestamp, ...],
    sample_series_ids: tuple[Any, ...],
    history_cutoff: pd.Timestamp,
) -> tuple[
    CanonicalFeatureScaler,
    CanonicalTargetTransform,
    tuple[np.ndarray, ...],
    np.ndarray,
]:
    feature_scaler = CanonicalFeatureScaler.from_config(
        config,
        feature_names=builder.feature_schema,
        categorical_names=builder.categorical_schema,
        category_orders={
            column: tuple(
                dict.fromkeys(
                    (
                        identity[index]
                        if isinstance(identity, tuple)
                        else identity
                    )
                    for identity in builder.series_ids
                )
            )
            for index, column in enumerate(config.problem.series_id_cols)
        },
    )
    transformed_X = feature_scaler.fit_transform_calls(X_by_call)
    target_transform = CanonicalTargetTransform.from_config(config)
    target_transform.fit_transform(builder.target_history(history_cutoff))
    transformed_Y = target_transform.transform_training(
        Y,
        origins,
        series_ids=sample_series_ids,
    )
    return feature_scaler, target_transform, transformed_X, transformed_Y


def _forecast_designs_with_scaler(
    builder: _RegistryDesignBuilder,
    origin: pd.Timestamp,
    feature_scaler: CanonicalFeatureScaler,
    target_transform: CanonicalTargetTransform,
):
    raw_designs, raw_provider = builder.forecast_designs(
        origin,
        target_transform=target_transform,
    )

    def provider(call_index, coordinates, dependencies, predicted):
        return feature_scaler.transform(
            raw_provider(call_index, coordinates, dependencies, predicted)
        )

    return (feature_scaler.transform(raw_designs[0]),), provider


def _restore_prediction(
    prediction: PointForecastTensor | MarginalForecastDistribution,
    target_transform: CanonicalTargetTransform,
) -> PointForecastTensor | MarginalForecastDistribution:
    if isinstance(prediction, PointForecastTensor):
        return target_transform.restore_point(prediction)
    if isinstance(prediction, MarginalForecastDistribution):
        return target_transform.restore_distribution(prediction)
    raise TypeError(f"unsupported canonical prediction type: {type(prediction).__name__}")


def _fit_point(
    config: ForecastConfigSpec,
    feature_schema: tuple[str, ...],
    X_by_call: tuple[np.ndarray, ...],
    Y: np.ndarray,
    *,
    n_series: int,
):
    capabilities = resolve_model_capabilities(
        config.estimator.model_type,
        config.estimator.params,
        feature_names=feature_schema,
        probe_native=config.estimator.target_adapter is TargetAdapter.NATIVE,
    )
    trainer = CanonicalTrainer(
        config,
        estimator_factory=make_model_factory(
            config.estimator.model_type,
            config.estimator.params,
            feature_names=feature_schema,
        ),
        capabilities=capabilities,
        feature_schema=feature_schema,
    )
    return trainer, trainer.train(X_by_call, Y, n_series=n_series)


def _fit_quantile(
    config: ForecastConfigSpec,
    feature_schema: tuple[str, ...],
    X_by_call: tuple[np.ndarray, ...],
    Y: np.ndarray,
    *,
    n_series: int,
):
    capabilities = resolve_model_capabilities(
        config.estimator.model_type,
        config.estimator.params,
        feature_names=feature_schema,
        probe_native=config.estimator.target_adapter is TargetAdapter.NATIVE,
    )
    if not capabilities.scalar_quantile:
        raise ValueError(
            f"model_type {config.estimator.model_type!r} does not support scalar quantiles"
        )
    trainer = CanonicalMarginalQuantileTrainer(
        config,
        estimator_factory_for_level=lambda level: make_model_factory(
            config.estimator.model_type,
            config.estimator.params,
            feature_names=feature_schema,
            quantile=level,
        ),
        capabilities=capabilities,
        feature_schema=feature_schema,
    )
    return trainer, trainer.train(X_by_call, Y, n_series=n_series), capabilities


def _predict(
    config: ForecastConfigSpec,
    artifact,
    base_design: np.ndarray,
    provider,
    forecast_times: pd.DatetimeIndex,
    series_ids: tuple[Any, ...],
):
    kwargs = {
        "series_ids": series_ids,
        "forecast_times": forecast_times,
        "feature_provider": provider,
    }
    if str(config.probabilistic.get("mode", "point")) == "point":
        return CanonicalForecaster(config, artifact).predict(base_design, **kwargs)
    return CanonicalMarginalQuantileForecaster(config, artifact).predict(
        base_design,
        **kwargs,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _proof_payload(compiled_items: tuple[CompiledFeatures, ...]) -> list[dict[str, Any]]:
    payload = []
    seen = set()
    for compiled in compiled_items:
        for proof in compiled.visibility_proof:
            item = asdict(proof)
            key = tuple(item.items())
            if key in seen:
                continue
            seen.add(key)
            item["target_time"] = proof.target_time.isoformat()
            item["source_time"] = (
                proof.source_time.isoformat() if proof.source_time is not None else None
            )
            item["forecast_origin"] = proof.forecast_origin.isoformat()
            item["available_at"] = (
                proof.available_at.isoformat() if proof.available_at is not None else None
            )
            payload.append(item)
    return payload


def _source_lineage_payload(
    compiled_items: tuple[CompiledFeatures, ...],
) -> list[dict[str, Any]]:
    payload = []
    seen = set()
    for compiled in compiled_items:
        for lineage in compiled.source_lineage:
            item = {
                "source": lineage.source_name,
                "path_version": lineage.path_version,
                "path": lineage.path,
                "availability": lineage.availability_policy,
                "oracle": lineage.oracle,
            }
            key = tuple(item.items())
            if key not in seen:
                seen.add(key)
                payload.append(item)
    return payload


def _compiled_lineage(
    feature_schema: tuple[str, ...],
    proof_payload: list[dict[str, Any]],
    config: ForecastConfigSpec,
) -> tuple[tuple[dict[str, Any], ...], dict[str, list[str]]]:
    by_feature = {}
    for item in proof_payload:
        by_feature.setdefault(item["feature_name"], item)
    feature_lineage = []
    availability_summary: dict[str, list[str]] = {}
    source_availability = {
        source.name: (
            source.availability.value if source.availability is not None else "static"
        )
        for source in config.data.sources
    }
    for feature in feature_schema:
        if feature in config.problem.series_id_cols:
            feature_lineage.append(
                {
                    "feature": feature,
                    "source": "series_identity",
                    "role": "key",
                    "source_time": None,
                    "provider": None,
                    "availability": "static",
                }
            )
            availability_summary.setdefault("static", []).append(feature)
            continue
        proof = by_feature[feature]
        availability = (
            "known_future"
            if proof["source_name"] == "calendar"
            else source_availability.get(proof["source_name"], proof["role"])
        )
        feature_lineage.append(
            {
                "feature": feature,
                "source": proof["source_name"],
                "role": proof["role"],
                "source_time": proof["source_time"],
                "provider": proof["provider"],
                "availability": availability,
            }
        )
        availability_summary.setdefault(availability, []).append(feature)
    return (
        tuple(feature_lineage),
        {key: availability_summary[key] for key in sorted(availability_summary)},
    )


def _output_paths(
    config: ForecastConfigSpec,
    fingerprint: str,
    output_root: str | Path | None,
) -> tuple[Path, Path, Path, Path]:
    output = config.output
    identity = output.get("identity", {})
    scenario = str(
        identity.get("scenario_subpath", output.get("scenario_subpath", "canonical"))
        if isinstance(identity, Mapping)
        else output.get("scenario_subpath", "canonical")
    ).strip("/") or "canonical"
    result_identity = config.result_identity()
    if output_root is not None:
        run_dir = Path(output_root) / scenario / result_identity
        return (
            run_dir,
            run_dir / "pretrained_models",
            run_dir / "results_test",
            run_dir / "results_forecast",
        )
    directories = output.get("directories", {})
    if isinstance(directories, Mapping) and {
        "checkpoints",
        "tests",
        "forecast",
    }.issubset(directories):
        model_dir = Path(str(directories["checkpoints"])) / scenario / result_identity
        test_dir = Path(str(directories["tests"])) / scenario / result_identity
        forecast_dir = Path(str(directories["forecast"])) / scenario / result_identity
        return forecast_dir, model_dir, test_dir, forecast_dir
    legacy_directories = {
        "checkpoints_dir": "model",
        "test_results_dir": "test",
        "pred_results_dir": "forecast",
    }
    if set(legacy_directories).issubset(output):
        model_dir = Path(str(output["checkpoints_dir"])) / scenario / result_identity
        test_dir = Path(str(output["test_results_dir"])) / scenario / result_identity
        forecast_dir = Path(str(output["pred_results_dir"])) / scenario / result_identity
        return forecast_dir, model_dir, test_dir, forecast_dir
    legacy_scenario = str(output.get("scenario_subpath", scenario)).strip("/") or scenario
    legacy_root = Path(str(output.get("results_root", "results")))
    run_dir = legacy_root / legacy_scenario / result_identity
    return (
        run_dir,
        run_dir / "pretrained_models",
        run_dir / "results_test",
        run_dir / "results_forecast",
    )


class CanonicalBaseModelRunner:
    """Public narrow facade over the validated single-model runtime path.

    Extracted in E1 (v4 §6.1). Single-model `run_canonical_config` delegates to
    `run`; ensemble members (E5+) depend only on this facade, never on runtime
    private helpers.
    """

    def __init__(
        self,
        config: ForecastConfigSpec,
        registry: SourceRegistry,
        origin: pd.Timestamp,
    ) -> None:
        if not isinstance(config, ForecastConfigSpec):
            raise TypeError("config must be a ForecastConfigSpec")
        if config.strategy is None:
            raise ValueError("CanonicalBaseModelRunner requires a strategy")
        self.config = config
        self.registry = registry
        self.origin = origin
        self.builder = _RegistryDesignBuilder(config, registry)
        (
            self.X_all,
            self.Y_all,
            self.supervised_origins,
            self.supervised_sample_origins,
            self.supervised_sample_series_ids,
        ) = _supervised_arrays(self.builder, origin)

    @property
    def geometry(self) -> validation.TimeGeometry:
        return validation.TimeGeometry(
            offset=self.builder.offset,
            horizon=self.config.problem.horizon,
        )

    @property
    def series_ids(self) -> tuple[Any, ...]:
        return self.builder.series_ids

    @property
    def feature_schema(self) -> tuple[str, ...]:
        return self.builder.feature_schema

    def backtest_windows(self) -> tuple[_BacktestWindow, ...]:
        return _rolling_backtest_windows(self.builder, self.supervised_origins)

    def fit(
        self,
        train_indices: tuple[int, ...],
    ) -> tuple[
        CanonicalFeatureScaler,
        CanonicalTargetTransform,
        tuple[np.ndarray, ...],
        np.ndarray,
        Any,
    ]:
        """Fit this member's own transforms and model on given train origins.

        Returns ``(feature_scaler, target_transform, X_train_transformed,
        Y_train_transformed, artifact)``; the artifact is a point
        `CanonicalStrategyArtifact` or a quantile
        `CanonicalMarginalQuantileArtifact` depending on the config mode.
        """
        mode = self._mode()
        train_sample_indices = _sample_indices(
            train_indices,
            self.builder.n_series,
        )
        X_train = tuple(
            design[list(train_sample_indices)] for design in self.X_all
        )
        Y_train = self.Y_all[list(train_sample_indices)]
        training_origins = tuple(
            self.supervised_sample_origins[index]
            for index in train_sample_indices
        )
        training_series_ids = tuple(
            self.supervised_sample_series_ids[index]
            for index in train_sample_indices
        )
        training_history_cutoff = max(
            _label_end(self.builder, self.supervised_origins[index])
            for index in train_indices
        )
        (
            feature_scaler,
            target_transform,
            X_train_transformed,
            Y_train_transformed,
        ) = _fit_runtime_transforms(
            self.config,
            self.builder,
            X_train,
            Y_train,
            training_origins,
            training_series_ids,
            training_history_cutoff,
        )
        # 监督特征选择（2026-08-30 专项）：有监督步骤挂在训练 fit 边界，
        # 每个回测窗口/最终训练各自重拟合，只消费当前训练窗 (X, Y)，无泄漏；
        # 选中集写入 artifact.feature_schema，预测端按同名子集对齐。
        X_train_transformed, feature_schema = self._apply_feature_selection(
            X_train_transformed, Y_train_transformed
        )
        if mode == "point":
            _, artifact = _fit_point(
                self.config,
                feature_schema,
                X_train_transformed,
                Y_train_transformed,
                n_series=self.builder.n_series,
            )
        else:
            _, artifact, _ = _fit_quantile(
                self.config,
                feature_schema,
                X_train_transformed,
                Y_train_transformed,
                n_series=self.builder.n_series,
            )
        return (
            feature_scaler,
            target_transform,
            X_train_transformed,
            Y_train_transformed,
            artifact,
        )

    def forecast_designs(
        self,
        origin: pd.Timestamp,
        feature_scaler: CanonicalFeatureScaler,
        target_transform: CanonicalTargetTransform,
    ) -> tuple[tuple[np.ndarray, ...], Any]:
        return _forecast_designs_with_scaler(
            self.builder,
            origin,
            feature_scaler,
            target_transform,
        )

    def predict(
        self,
        artifact: Any,
        designs: tuple[np.ndarray, ...],
        provider: Any,
        forecast_times: pd.DatetimeIndex,
        target_transform: CanonicalTargetTransform,
    ) -> PointForecastTensor | MarginalForecastDistribution:
        """Predict at the given origin and restore to the original target space."""
        base_design = designs[0]
        # 特征选择对齐（2026-08-30 专项）：artifact.feature_schema 是训练期选中集，
        # 预测端把全 schema 设计矩阵按同名子集对齐（provider 输出同为全 schema 宽）。
        artifact_schema = (
            artifact.feature_schema
            if isinstance(artifact, CanonicalStrategyArtifact)
            else next(iter(artifact.artifacts_by_level.values())).feature_schema
        )
        indices = selected_indices_for_artifact(
            self.builder.feature_schema, tuple(artifact_schema)
        )
        if indices is not None:
            base_design = np.asarray(base_design, dtype=float)[:, indices]
            full_provider = provider

            def selected_provider(call_index, coordinates, dependencies, predicted):
                return np.asarray(
                    full_provider(call_index, coordinates, dependencies, predicted),
                    dtype=float,
                )[:, indices]

            provider = selected_provider

        raw = _predict(
            self.config,
            artifact,
            base_design,
            provider,
            forecast_times,
            self.builder.series_ids,
        )
        return _restore_prediction(raw, target_transform)

    def actual(
        self,
        origin_index: int,
        forecast_times: pd.DatetimeIndex,
    ) -> PointForecastTensor:
        return _actual_at_origin(
            self.config,
            self.Y_all,
            origin_index,
            forecast_times,
            self.builder.series_ids,
        )

    def seasonal_naive(
        self,
        origin: pd.Timestamp,
        forecast_times: pd.DatetimeIndex,
    ) -> PointForecastTensor:
        return _seasonal_naive_tensor(
            self.builder,
            origin,
            forecast_times,
        )

    def forecast_times(
        self,
        origin: pd.Timestamp,
    ) -> pd.DatetimeIndex:
        return pd.date_range(
            origin,
            periods=self.config.problem.horizon + 1,
            freq=self.config.problem.freq,
        )[1:]

    def final_bundle_inputs(self) -> tuple[
        CanonicalFeatureScaler,
        CanonicalTargetTransform,
        tuple[np.ndarray, ...],
        np.ndarray,
    ]:
        """Final-fit transforms over the full visible history."""
        (
            feature_scaler,
            target_transform,
            X_all_transformed,
            Y_all_transformed,
        ) = _fit_runtime_transforms(
            self.config,
            self.builder,
            self.X_all,
            self.Y_all,
            self.supervised_sample_origins,
            self.supervised_sample_series_ids,
            self.origin,
        )
        return feature_scaler, target_transform, X_all_transformed, Y_all_transformed

    def fit_final(
        self,
        X_transformed: tuple[np.ndarray, ...],
        Y_transformed: np.ndarray,
    ) -> tuple[CanonicalTrainer, Any, Any]:
        """Train the final artifact and return (trainer, artifact, capabilities)."""
        mode = self._mode()
        X_transformed, feature_schema = self._apply_feature_selection(
            X_transformed, Y_transformed
        )
        if mode == "point":
            trainer, artifact = _fit_point(
                self.config,
                feature_schema,
                X_transformed,
                Y_transformed,
                n_series=self.builder.n_series,
            )
            capabilities = trainer.capabilities
        else:
            trainer, artifact, capabilities = _fit_quantile(
                self.config,
                feature_schema,
                X_transformed,
                Y_transformed,
                n_series=self.builder.n_series,
            )
        return trainer, artifact, capabilities

    def _apply_feature_selection(
        self,
        X_by_call: tuple[np.ndarray, ...],
        Y: np.ndarray,
    ) -> tuple[tuple[np.ndarray, ...], tuple[str, ...]]:
        """监督特征选择（features.selection，2026-08-30 专项）。

        有监督步骤挂在训练 fit 边界：每个回测窗口与最终训练各自重拟合选择器，
        只消费当前训练窗的 (X, Y)，无泄漏；未配置/未启用时原样直通。
        选中集进入 artifact.feature_schema，预测端按同名子集对齐。
        """
        feature_schema = self.builder.feature_schema
        spec = normalize_feature_selection(self.config.features.selection)
        if spec is None or not spec.enabled:
            return X_by_call, feature_schema
        selector = CanonicalFeatureSelector(spec, feature_schema)
        y_signal = Y.reshape(Y.shape[0], -1).mean(axis=1)
        selector.fit(X_by_call[0], y_signal)
        assert selector.selected_names_ is not None  # fit 后必有选中集
        logger.info(
            "[FeatureSelection] %d -> %d features (method=%s)",
            len(feature_schema),
            len(selector.selected_names_),
            spec.method,
        )
        return (
            tuple(selector.transform(design) for design in X_by_call),
            selector.selected_names_,
        )

    def run(
        self,
        output_root: str | Path | None = None,
    ) -> CanonicalRuntimeResult:
        """Full single-model lifecycle: rolling backtest, final fit, persist."""
        builder = self.builder
        config = self.config
        origin = self.origin
        mode = self._mode()

        fingerprint = config.fingerprint()
        run_dir, model_dir, test_dir, forecast_dir = _output_paths(
            config,
            fingerprint,
            output_root,
        )
        aggregate_weights = resolve_aggregate_weighting(
            config.problem.targets,
            config.validation.get("aggregate_weighting"),
        )
        cv_frames = []
        score_frames = []
        prob_score_frames = []
        eval_mask_config = (
            config.validation.get("eval_mask")
            if isinstance(config.validation.get("eval_mask"), Mapping)
            else None
        )
        holdout_audits = []
        for backtest_window in self.backtest_windows():
            (
                holdout_feature_scaler,
                holdout_target_transform,
                _X_train_transformed,
                _Y_train_transformed,
                holdout_artifact,
            ) = self.fit(backtest_window.train_indices)

            builder.reset_audit()
            holdout_designs, holdout_provider = self.forecast_designs(
                backtest_window.origin,
                holdout_feature_scaler,
                holdout_target_transform,
            )
            holdout_times = self.forecast_times(backtest_window.origin)
            holdout_prediction = self.predict(
                holdout_artifact,
                holdout_designs,
                holdout_provider,
                holdout_times,
                holdout_target_transform,
            )
            actual = self.actual(
                backtest_window.origin_index,
                holdout_times,
            )
            seasonal_naive = self.seasonal_naive(
                backtest_window.origin,
                holdout_times,
            )
            holdout_point = (
                holdout_prediction
                if isinstance(holdout_prediction, PointForecastTensor)
                else holdout_prediction.point
            )
            cv_frames.append(
                backtest_tensors_to_long(
                    actual,
                    holdout_prediction,
                    window=backtest_window.window,
                )
            )
            score_frames.append(
                evaluate_point_forecasts(
                    actual,
                    holdout_point,
                    aggregate_weighting=aggregate_weights,
                    seasonal_naive=seasonal_naive,
                    window=backtest_window.window,
                    eval_mask=eval_mask_config,
                )
            )
            # 概率评估接线（2026-08-30）：quantile 模式逐窗产出 pinball/central 区间
            # 指标；掩码口径与点评估共用同一 eval_mask payload（同一业务口径）。
            if not isinstance(holdout_prediction, PointForecastTensor):
                prob_mask_payload = build_eval_mask_payload(eval_mask_config, actual)
                prob_score_frames.append(
                    evaluate_marginal_distribution(
                        actual,
                        holdout_prediction,
                        valid_masks=(
                            {
                                target: payload["valid_mask"]
                                for target, payload in prob_mask_payload.items()
                            }
                            if prob_mask_payload is not None
                            else None
                        ),
                        window=backtest_window.window,
                    )
                )
            holdout_audits.extend(builder.audit)
        holdout_metadata = {
            **backtest_window.metadata,
            "history_length": int(config.validation.get("history_length", len(self.supervised_origins))),
            "window_length": int(config.validation.get("window_length", len(self.supervised_origins))),
            "max_test_windows": int(config.validation.get("max_test_windows", 1)),
            "test_window_stride": int(
                config.validation.get("test_window_stride", config.problem.horizon)
            ),
            "windows": [window.metadata for window in self.backtest_windows()],
        }
        holdout_audit = tuple(holdout_audits)
        write_backtest_results(
            test_dir,
            pd.concat(cv_frames, ignore_index=True),
            pd.concat(score_frames, ignore_index=True),
            aggregate_weighting=aggregate_weights,
            metadata={"backtest": holdout_metadata},
            probabilistic_scores_df=(
                pd.concat(prob_score_frames, ignore_index=True)
                if prob_score_frames
                else None
            ),
        )

        (
            final_feature_scaler,
            final_target_transform,
            X_all_transformed,
            Y_all_transformed,
        ) = self.final_bundle_inputs()
        final_trainer, final_artifact, final_capabilities = self.fit_final(
            X_all_transformed,
            Y_all_transformed,
        )

        builder.reset_audit()
        final_designs, final_provider = self.forecast_designs(
            origin,
            final_feature_scaler,
            final_target_transform,
        )
        forecast_times = self.forecast_times(origin)
        forecast = self.predict(
            final_artifact,
            final_designs,
            final_provider,
            forecast_times,
            final_target_transform,
        )
        final_audit = builder.audit
        visibility_proof = _proof_payload(final_audit)
        holdout_visibility_proof = _proof_payload(holdout_audit)
        source_lineage = _source_lineage_payload(final_audit)
        feature_lineage, availability_summary = _compiled_lineage(
            builder.feature_schema,
            visibility_proof,
            config,
        )
        if mode == "point":
            bundle_builder = final_trainer
            bundle_artifact = final_artifact
        else:
            point_level = float(config.probabilistic.get("point_quantile", 0.5))
            bundle_builder = CanonicalTrainer(
                config,
                estimator_factory=make_model_factory(
                    config.estimator.model_type,
                    config.estimator.params,
                    feature_names=builder.feature_schema,
                    quantile=point_level,
                ),
                capabilities=final_capabilities,
                feature_schema=builder.feature_schema,
            )
            bundle_artifact = final_artifact.artifacts_by_level[point_level]
        bundle = bundle_builder.build_model_bundle(
            bundle_artifact,
            feature_scaler=final_feature_scaler,
            target_transform=final_target_transform,
            input_schema={
                "columns": list(builder.feature_schema),
                "panel": {
                    "series_id_cols": list(config.problem.series_id_cols),
                    "known_series_ids": [
                        list(value) if isinstance(value, tuple) else value
                        for value in builder.series_ids
                    ],
                    "unknown_series_policy": str(
                        builder._training_scope_validation().get(
                            "unknown_series_policy",
                            "raise",
                        )
                    ).lower(),
                },
                "availability_summary": availability_summary,
                "visibility_proof": visibility_proof,
            },
            feature_lineage=feature_lineage,
            source_lineage=source_lineage,
            series_ids=builder.series_ids,
        )
        if mode == "quantile":
            bundle.model = final_artifact

        model_dir.mkdir(parents=True, exist_ok=True)
        ModelDeployPkl(model_dir / "model.pkl").save_model(bundle)
        bundle.write_schema_json(model_dir / "resolved_model.json")

        write_forecast_results(forecast_dir, forecast)
        _write_json(
            forecast_dir / "resolved_config.json",
            {
                **config.canonical_payload(),
                "config_fingerprint": fingerprint,
                "runtime": {
                    "series_order": [
                        list(value) if isinstance(value, tuple) else value
                        for value in builder.series_ids
                    ],
                    "feature_schema": list(builder.feature_schema),
                    "availability_summary": availability_summary,
                    "source_lineage": source_lineage,
                    "visibility_proof": visibility_proof,
                    "holdout_visibility_proof": holdout_visibility_proof,
                    "capability_probe": {
                        "native_multioutput_probed": (
                            config.estimator.target_adapter is TargetAdapter.NATIVE
                        ),
                        "resolved": final_capabilities.canonical_payload(),
                    },
                    "strategy": {
                        "model_count": builder.plan.model_count,
                        "dependencies": [
                            [
                                {
                                    "target": coordinate.target,
                                    "horizon_step": coordinate.horizon_step,
                                }
                                for coordinate in dependencies
                            ]
                            for dependencies in builder.plan.dependencies
                        ],
                    },
                    "holdout": holdout_metadata,
                },
            },
        )
        return CanonicalRuntimeResult(
            run_dir=run_dir,
            model_dir=model_dir,
            test_dir=test_dir,
            forecast_dir=forecast_dir,
            fingerprint=fingerprint,
            bundle=bundle,
        )

    def _mode(self) -> str:
        mode = str(self.config.probabilistic.get("mode", "point"))
        if mode not in {"point", "quantile"}:
            raise ValueError(
                f"unsupported canonical probabilistic mode: {mode!r}"
            )
        return mode


def run_canonical_config(
    config: ForecastConfigSpec,
    output_root: str | Path | None = None,
    *,
    generators: Mapping[str, Any] | None = None,
) -> CanonicalRuntimeResult:
    """Train, backtest, forecast, and persist one canonical config."""
    if not isinstance(config, ForecastConfigSpec):
        raise TypeError("config must be a ForecastConfigSpec")
    if config.strategy is None:
        raise ValueError("run_canonical_config requires a strategy or ensemble")
    registry = SourceRegistry(config.data, Path.cwd(), generators=generators)
    origin = _resolve_origin(registry, config.validation.get("forecast_origin"))
    return CanonicalBaseModelRunner(config, registry, origin).run(output_root)


__all__ = [
    "CanonicalBaseModelRunner",
    "CanonicalRuntimeResult",
    "run_canonical_config",
]
