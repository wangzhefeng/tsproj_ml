"""Explicit future-value providers independent of legacy forecasting models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

import numpy as np
import pandas as pd

from data_loading.sources.source_io import SourceFrames
from data_loading.processing.validation import validate_frame, resolve_available_at_col
from data_loading.processing.visibility import select_as_of_vintage
from data_loading.processing.alignment import select_requested_series, request_identities, filter_identity
from forecasting_core.specs.data import AvailabilityPolicy, ColumnRole, DataSourceSpec

if TYPE_CHECKING:
    from data_loading.information.information_set import InformationSetRequest


def _validate_horizon(horizon: int) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise TypeError("horizon must be an integer")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return horizon


def _validate_name(name: object) -> str:
    if not isinstance(name, str):
        raise TypeError("trajectory names must be strings")
    if not name or name != name.strip():
        raise ValueError("trajectory names must be non-blank without surrounding whitespace")
    return name


def _validate_step(step: int, horizon: int) -> int:
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError("step must be an integer")
    if step < 0 or step >= horizon:
        raise IndexError(f"step must be in [0, {horizon})")
    return step


def _finite_trajectories(
    trajectories: Mapping[str, Sequence[float]],
    horizon: int,
) -> dict[str, tuple[float, ...]]:
    if not isinstance(trajectories, Mapping) or not trajectories:
        raise ValueError("trajectories must be a non-empty mapping")
    normalized: dict[str, tuple[float, ...]] = {}
    for raw_name, raw_values in trajectories.items():
        name = _validate_name(raw_name)
        if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence):
            if not isinstance(raw_values, np.ndarray):
                raise TypeError("trajectory values must be sequences")
        values = np.asarray(raw_values, dtype=float)
        if values.ndim != 1 or len(values) != horizon:
            raise ValueError(f"trajectory {name!r} must contain exactly horizon values")
        if not np.isfinite(values).all():
            raise ValueError(f"trajectory {name!r} must contain only finite values")
        normalized[name] = tuple(float(value) for value in values)
    return normalized


@runtime_checkable
class EndogenousFutureProvider(Protocol):
    @property
    def horizon(self) -> int:
        ...

    def value_at(self, name: str, step: int) -> float:
        ...


class _TrajectoryProvider:
    __slots__ = ("_horizon", "_trajectories")

    def __init__(self, *, horizon: int, trajectories: Mapping[str, Sequence[float]]) -> None:
        self._horizon = _validate_horizon(horizon)
        self._trajectories = _finite_trajectories(trajectories, self._horizon)

    @property
    def horizon(self) -> int:
        return self._horizon

    def value_at(self, name: str, step: int) -> float:
        normalized_name = _validate_name(name)
        normalized_step = _validate_step(step, self._horizon)
        try:
            return self._trajectories[normalized_name][normalized_step]
        except KeyError as exc:
            raise KeyError(f"unknown trajectory: {normalized_name!r}") from exc


class ProvidedScenarioProvider(_TrajectoryProvider):
    pass


class AuxiliaryProvider(_TrajectoryProvider):
    pass


class CompositeProvider(_TrajectoryProvider):
    __slots__ = ("_methods",)

    def __init__(
        self,
        *,
        horizon: int,
        trajectories: Mapping[str, Sequence[float]],
        methods: Mapping[str, str],
    ) -> None:
        super().__init__(horizon=horizon, trajectories=trajectories)
        normalized_methods = {
            _validate_name(name): _validate_name(method)
            for name, method in methods.items()
        }
        if set(normalized_methods) != set(self._trajectories):
            raise ValueError("methods must exactly cover provider trajectories")
        self._methods = normalized_methods

    def provider_name(self, name: str) -> str:
        normalized_name = _validate_name(name)
        try:
            return self._methods[normalized_name]
        except KeyError as exc:
            raise KeyError(f"unknown trajectory: {normalized_name!r}") from exc


class PersistenceProvider(_TrajectoryProvider):
    def __init__(self, *, horizon: int, history: Mapping[str, Sequence[float]]) -> None:
        normalized_horizon = _validate_horizon(horizon)
        if not isinstance(history, Mapping) or not history:
            raise ValueError("history must be a non-empty mapping")
        trajectories: dict[str, tuple[float, ...]] = {}
        for raw_name, raw_values in history.items():
            name = _validate_name(raw_name)
            if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence):
                if not isinstance(raw_values, np.ndarray):
                    raise TypeError("history values must be sequences")
            values = np.asarray(raw_values, dtype=float)
            if values.ndim != 1 or values.size == 0:
                raise ValueError(f"history {name!r} must be a non-empty one-dimensional sequence")
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                raise ValueError(f"history {name!r} has no finite value")
            trajectories[name] = (float(finite[-1]),) * normalized_horizon
        super().__init__(horizon=normalized_horizon, trajectories=trajectories)


def create_endogenous_future_provider(
    method: str,
    *,
    horizon: int,
    trajectories: Mapping[str, Sequence[float]] | None = None,
    history: Mapping[str, Sequence[float]] | None = None,
) -> EndogenousFutureProvider:
    if method == "provided_scenario":
        if trajectories is None or history is not None:
            raise ValueError("provided_scenario requires trajectories and forbids history")
        return ProvidedScenarioProvider(horizon=horizon, trajectories=trajectories)
    if method == "persistence":
        if history is None or trajectories is not None:
            raise ValueError("persistence requires history and forbids trajectories")
        return PersistenceProvider(horizon=horizon, history=history)
    if method == "auxiliary":
        if trajectories is None or history is not None:
            raise ValueError("auxiliary requires trajectories and forbids history")
        return AuxiliaryProvider(horizon=horizon, trajectories=trajectories)
    raise ValueError(f"unknown endogenous future provider method: {method!r}")


class ObservedFutureProvider:
    __slots__ = ("_provider", "_available_at")

    def __init__(
        self,
        *,
        horizon: int,
        trajectories: Mapping[str, tuple[float, ...]],
        methods: Mapping[str, str],
        available_at: Mapping[str, tuple[pd.Timestamp, ...]],
    ) -> None:
        self._provider = CompositeProvider(
            horizon=horizon,
            trajectories=trajectories,
            methods=methods,
        )
        if set(available_at) != set(trajectories):
            raise ValueError("available_at must exactly cover provider trajectories")
        normalized = {
            name: tuple(pd.Timestamp(value) for value in values)
            for name, values in available_at.items()
        }
        if any(len(values) != horizon for values in normalized.values()):
            raise ValueError("provider available_at must exactly cover the horizon")
        self._available_at = normalized

    @property
    def horizon(self) -> int:
        return self._provider.horizon

    def value_at(self, name: str, step: int) -> float:
        return self._provider.value_at(name, step)

    def provider_name(self, name: str) -> str:
        return self._provider.provider_name(name)

    def available_at(self, name: str, step: int) -> pd.Timestamp:
        self._provider.value_at(name, step)
        return self._available_at[name][step]


def collect_observed_provider(
    frames: SourceFrames,
    source: DataSourceSpec,
    history_frame: pd.DataFrame,
    request: InformationSetRequest,
    values_by_identity: dict[Any, dict[str, tuple[float, ...]]],
    methods_by_identity: dict[Any, dict[str, str]],
    available_at_by_identity: dict[
        Any, dict[str, tuple[pd.Timestamp, ...]]
    ],
) -> None:
    method = source.provider
    if method is None:
        raise ValueError(f"observed_past source {source.name!r} requires provider")
    columns = [
        column.name
        for column in source.columns
        if column.role is ColumnRole.OBSERVED_PAST
    ]
    if method == "persistence":
        provider_frame = history_frame
    else:
        if source.backtest_path is None:
            return
        if source.availability is AvailabilityPolicy.SOURCE_TIME:
            raise ValueError(
                f"observed_past source {source.name!r} with availability=source_time "
                "cannot expose a future observed provider trajectory"
            )
        provider_frame = validate_frame(
            source,
            frames.read_path(source.backtest_path),
            generated=False,
            path_version="backtest",
        )
        if source.availability in {
            AvailabilityPolicy.COLUMN,
            AvailabilityPolicy.GENERATOR_DEFINED,
        }:
            provider_frame = select_as_of_vintage(
                source,
                provider_frame,
                request,
            )
        provider_frame = provider_frame.loc[
            provider_frame[source.time_col].isin(request.forecast_times)
        ].copy()
        provider_frame = select_requested_series(
            source,
            provider_frame,
            request,
            strict_extra=True,
        )
    identities = request_identities(source, request)
    for identity in identities:
        selected = filter_identity(source, provider_frame, identity)
        selected = selected.sort_values(source.time_col, kind="stable")
        target = values_by_identity.setdefault(identity, {})
        methods = methods_by_identity.setdefault(identity, {})
        availability = available_at_by_identity.setdefault(identity, {})
        for column in columns:
            if column in target:
                raise ValueError(f"duplicate observed provider column: {column!r}")
            if method == "persistence":
                if selected.empty:
                    raise ValueError(
                        f"observed history {source.name!r} is empty for {identity!r}"
                    )
                trajectory = (float(selected.iloc[-1][column]),) * request.H
                available_at = (request.forecast_origin,) * request.H
            else:
                if len(selected) != request.H:
                    raise ValueError(
                        f"observed provider {source.name!r} must exactly cover the horizon"
                    )
                trajectory = tuple(float(value) for value in selected[column])
                available_at_col = resolve_available_at_col(source)
                available_at = (
                    tuple(
                        pd.Timestamp(value)
                        for value in selected[available_at_col]
                    )
                    if available_at_col is not None
                    else (request.forecast_origin,) * request.H
                )
                if any(value > request.forecast_origin for value in available_at):
                    raise ValueError(
                        f"observed provider {source.name!r} contains values "
                        "available after forecast_origin"
                    )
            target[column] = trajectory
            methods[column] = method
            availability[column] = available_at


__all__ = [
    "ObservedFutureProvider",
    "collect_observed_provider",
    "AuxiliaryProvider",
    "CompositeProvider",
    "EndogenousFutureProvider",
    "PersistenceProvider",
    "ProvidedScenarioProvider",
    "create_endogenous_future_provider",
]
