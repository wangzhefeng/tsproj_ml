"""Canonical data-source registry and strict as-of materialization."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_loading.information_set import (
    InformationSetRequest,
    MaterializedInformationSet,
    SourceLineage,
)
from data_loading.providers import CompositeProvider
from model_forecasting.specs.data import AvailabilityPolicy, ColumnRole, DataSourceSpec, DataSpec


FrameReader = Callable[[Path], pd.DataFrame]
SourceGenerator = Callable[[DataSourceSpec, InformationSetRequest], pd.DataFrame]
_GENERATED_AVAILABLE_AT = "available_at"
_PATH_FIELDS = (
    ("history", "history_path"),
    ("backtest", "backtest_path"),
    ("future", "future_path"),
)


class _ObservedFutureProvider:
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


class SourceRegistry:
    def __init__(
        self,
        data_spec: DataSpec,
        base_dir: str | Path,
        *,
        reader: FrameReader = pd.read_csv,
        generators: Mapping[str, SourceGenerator] | None = None,
    ) -> None:
        if not isinstance(data_spec, DataSpec):
            raise TypeError("data_spec must be a DataSpec")
        if not callable(reader):
            raise TypeError("reader must be callable")
        normalized_generators = dict(generators or {})
        if any(not isinstance(name, str) or not callable(generator) for name, generator in normalized_generators.items()):
            raise TypeError("generators must map names to callables")
        self._data_spec = data_spec
        self._base_dir = Path(base_dir)
        self._reader = reader
        self._generators = normalized_generators
        self._raw_cache: dict[Path, pd.DataFrame] = {}
        # 性能缓存（2026-08-30 方案 A）：(path, source_name, generated) -> 验证后帧。
        # 同一数据文件与 source 规格的校验结果在一次 run 内不变，避免每个 origin
        # 的 materialize 重复 deep copy + to_datetime + 数值校验（profile 实测
        # 该路径占单配置冒烟 >40% 时间）。键含 source_name 隔离不同 source 规格。
        self._validated_cache: dict[tuple[Path, str, bool], pd.DataFrame] = {}

    def materialize(self, request: InformationSetRequest) -> MaterializedInformationSet:
        if not isinstance(request, InformationSetRequest):
            raise TypeError("request must be an InformationSetRequest")
        target_history: dict[str, pd.DataFrame] = {}
        observed_past: dict[str, pd.DataFrame] = {}
        known_future: dict[str, pd.DataFrame] = {}
        static: dict[str, pd.DataFrame] = {}
        observed_provider_values: dict[Any, dict[str, tuple[float, ...]]] = {}
        observed_provider_methods: dict[Any, dict[str, str]] = {}
        observed_provider_available_at: dict[
            Any, dict[str, tuple[pd.Timestamp, ...]]
        ] = {}
        lineage: list[SourceLineage] = []

        for source in self._data_spec.sources:
            frame, source_lineage = self._load_source(source, request)
            lineage.extend(source_lineage)
            roles = {column.role for column in source.columns}
            if ColumnRole.TARGET in roles:
                target_history[source.name] = self._history_frame(
                    source,
                    frame,
                    request,
                    role=ColumnRole.TARGET,
                )
            if ColumnRole.OBSERVED_PAST in roles:
                observed_frame = self._history_frame(
                    source,
                    frame,
                    request,
                    role=ColumnRole.OBSERVED_PAST,
                )
                observed_past[source.name] = observed_frame
                self._collect_observed_provider(
                    source,
                    observed_frame,
                    request,
                    observed_provider_values,
                    observed_provider_methods,
                    observed_provider_available_at,
                )
                if source.provider != "persistence" and source.backtest_path is not None:
                    lineage.append(
                        self._lineage(source, "provider", source.backtest_path, request)
                    )
            if ColumnRole.KNOWN_FUTURE in roles:
                known_future[source.name] = self._known_future_frame(source, frame, request)
            if ColumnRole.STATIC in roles:
                static[source.name] = self._static_frame(source, frame, request)

        return MaterializedInformationSet(
            target_history=target_history,
            observed_past=observed_past,
            known_future=known_future,
            static=static,
            observed_future_providers={
                identity: _ObservedFutureProvider(
                    horizon=request.H,
                    trajectories=trajectories,
                    methods=observed_provider_methods[identity],
                    available_at=observed_provider_available_at[identity],
                )
                for identity, trajectories in observed_provider_values.items()
            },
            lineage=lineage,
        )

    def latest_target_time(self) -> pd.Timestamp:
        latest: list[pd.Timestamp] = []
        for source in self._data_spec.sources:
            if not any(column.role is ColumnRole.TARGET for column in source.columns):
                continue
            if source.source_type != "file" or source.history_path is None:
                raise ValueError("target history discovery requires file history_path")
            frame = self._validate_frame(
                source,
                self._read_path(source.history_path),
                generated=False,
            )
            if source.time_col is None or frame.empty:
                raise ValueError(f"target source {source.name!r} has no timestamps")
            latest.append(pd.Timestamp(frame[source.time_col].max()))
        if not latest:
            raise ValueError("DataSpec has no target source")
        if len(set(latest)) != 1:
            raise ValueError("target sources must share one latest timestamp")
        return latest[0]

    def _load_source(
        self,
        source: DataSourceSpec,
        request: InformationSetRequest,
    ) -> tuple[pd.DataFrame, list[SourceLineage]]:
        if source.source_type == "generated":
            generator = self._generators.get(source.generator or "")
            if generator is None:
                raise ValueError(f"no generator registered for source {source.name!r}")
            generated = generator(source, request)
            if not isinstance(generated, pd.DataFrame):
                raise TypeError(f"generator for source {source.name!r} must return a DataFrame")
            frame = self._validate_frame(source, generated, generated=True)
            lineage = [self._lineage(source, "generated", None, request)]
        else:
            frames = []
            lineage = []
            roles = {column.role for column in source.columns}
            path_fields = (
                _PATH_FIELDS
                if ColumnRole.KNOWN_FUTURE in roles
                else (("history", "history_path"),)
            )
            for version, field_name in path_fields:
                configured_path = getattr(source, field_name)
                if configured_path is None:
                    continue
                # 方案 A：验证帧缓存——同一 (path, source, version) 的校验结果
                # 在一次 run 内不变；缓存值仅供读取（下游 information_set 仍做
                # 防御 copy），不破坏「边界防御拷贝」语义。
                resolved_path = Path(configured_path)
                if not resolved_path.is_absolute():
                    resolved_path = self._base_dir / resolved_path
                resolved_path = resolved_path.resolve()
                cache_key = (resolved_path, source.name, version == "history")
                cached = self._validated_cache.get(cache_key)
                if cached is None:
                    raw = self._read_path(configured_path)
                    cached = self._validate_frame(source, raw, generated=False)
                    self._validated_cache[cache_key] = cached
                frames.append(cached)
                lineage.append(self._lineage(source, version, configured_path, request))
            frame = pd.concat(frames, ignore_index=True)

        if source.availability in {AvailabilityPolicy.COLUMN, AvailabilityPolicy.GENERATOR_DEFINED}:
            roles = {column.role for column in source.columns}
            oracle_target = (
                request.information_mode == "oracle" and ColumnRole.TARGET in roles
            )
            frame = self._select_as_of_vintage(
                source,
                frame,
                request,
                enforce_origin=not oracle_target,
            )
        return frame, lineage

    def _read_path(self, configured_path: str) -> pd.DataFrame:
        path = Path(configured_path)
        if not path.is_absolute():
            path = self._base_dir / path
        path = path.resolve()
        if path not in self._raw_cache:
            frame = self._reader(path)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"reader must return a DataFrame for {path}")
            self._raw_cache[path] = frame.copy(deep=True)
        return self._raw_cache[path].copy(deep=True)

    @staticmethod
    def _lineage(
        source: DataSourceSpec,
        version: str,
        path: str | None,
        request: InformationSetRequest,
    ) -> SourceLineage:
        return SourceLineage(
            source_name=source.name,
            path_version=version,
            path=path,
            availability_policy=source.availability.value if source.availability else None,
            oracle=request.information_mode == "oracle",
        )

    def _validate_frame(
        self,
        source: DataSourceSpec,
        frame: pd.DataFrame,
        *,
        generated: bool,
    ) -> pd.DataFrame:
        validated = frame.copy(deep=True)
        expected = {column.name for column in source.columns}
        if source.time_col is not None:
            expected.add(source.time_col)
        if source.availability is AvailabilityPolicy.COLUMN:
            expected.add(source.available_at_col or "")
        if generated:
            expected.add(_GENERATED_AVAILABLE_AT)
        actual = set(validated.columns)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"source {source.name!r} columns do not match its spec; missing={missing}, extra={extra}"
            )

        if source.time_col is not None:
            validated[source.time_col] = self._parse_datetime(
                validated[source.time_col],
                f"source {source.name!r} time_col",
            )
        available_at_col = self._available_at_col(source)
        if available_at_col is not None:
            validated[available_at_col] = self._parse_datetime(
                validated[available_at_col],
                f"source {source.name!r} available_at",
            )

        for column in source.columns:
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

        primary_key = self._primary_key(source)
        if source.availability not in {AvailabilityPolicy.COLUMN, AvailabilityPolicy.GENERATOR_DEFINED}:
            if validated.duplicated(primary_key, keep=False).any():
                raise ValueError(f"source {source.name!r} has duplicate primary keys")
        return validated

    @staticmethod
    def _parse_datetime(values: pd.Series, label: str) -> pd.Series:
        try:
            parsed = pd.to_datetime(values, errors="raise", format="mixed")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must contain valid timestamps") from exc
        if parsed.isna().any():
            raise ValueError(f"{label} must not contain NaT")
        return parsed

    @staticmethod
    def _available_at_col(source: DataSourceSpec) -> str | None:
        if source.availability is AvailabilityPolicy.COLUMN:
            return source.available_at_col
        if source.availability is AvailabilityPolicy.GENERATOR_DEFINED:
            return _GENERATED_AVAILABLE_AT
        return None

    @staticmethod
    def _primary_key(source: DataSourceSpec) -> list[str]:
        keys = list(source.series_id_cols)
        if source.time_col is not None:
            keys.append(source.time_col)
        if not keys:
            raise ValueError(f"source {source.name!r} has no primary key")
        return keys

    def _select_as_of_vintage(
        self,
        source: DataSourceSpec,
        frame: pd.DataFrame,
        request: InformationSetRequest,
        *,
        enforce_origin: bool,
    ) -> pd.DataFrame:
        available_at_col = self._available_at_col(source)
        if available_at_col is None:
            raise ValueError(f"source {source.name!r} has no available_at semantics")
        if enforce_origin:
            eligible = frame.loc[frame[available_at_col] <= request.forecast_origin].copy()
        else:
            eligible = frame.copy()
        primary_key = self._primary_key(source)
        if eligible.empty:
            return eligible
        max_available = eligible.groupby(primary_key, dropna=False)[available_at_col].transform("max")
        latest = eligible.loc[eligible[available_at_col] == max_available].copy()
        if latest.duplicated(primary_key, keep=False).any():
            raise ValueError(f"source {source.name!r} has duplicate latest as-of vintages")
        return latest

    def _history_frame(
        self,
        source: DataSourceSpec,
        frame: pd.DataFrame,
        request: InformationSetRequest,
        *,
        role: ColumnRole,
    ) -> pd.DataFrame:
        if source.time_col is None:
            raise ValueError(f"history source {source.name!r} requires time_col at runtime")
        if role is ColumnRole.TARGET and request.information_mode == "oracle":
            allowed_time = (frame[source.time_col] <= request.forecast_origin) | frame[source.time_col].isin(
                request.forecast_times
            )
        else:
            allowed_time = frame[source.time_col] <= request.forecast_origin
        selected = frame.loc[allowed_time].copy()
        selected = self._select_requested_series(source, selected, request, strict_extra=False)
        self._require_history_per_series(source, selected, request)
        selected = self._order_temporal(source, selected, request)
        return self._role_frame(source, selected, role)

    def _known_future_frame(
        self,
        source: DataSourceSpec,
        frame: pd.DataFrame,
        request: InformationSetRequest,
    ) -> pd.DataFrame:
        if source.time_col is None:
            raise ValueError(f"known_future source {source.name!r} requires time_col at runtime")
        horizon_rows = frame.loc[frame[source.time_col].isin(request.forecast_times)].copy()
        selected = self._select_requested_series(source, horizon_rows, request, strict_extra=True)
        expected = self._expected_temporal_keys(source, request)
        actual = self._actual_temporal_keys(source, selected)
        if actual != expected or len(selected) != len(expected):
            missing = expected - actual
            extra = actual - expected
            raise ValueError(
                f"known_future source {source.name!r} must exactly cover the request; "
                f"missing={len(missing)}, extra={len(extra)}"
            )
        selected = self._order_temporal(source, selected, request)
        return self._role_frame(source, selected, ColumnRole.KNOWN_FUTURE)

    def _static_frame(
        self,
        source: DataSourceSpec,
        frame: pd.DataFrame,
        request: InformationSetRequest,
    ) -> pd.DataFrame:
        if source.series_id_cols and not request.series_ids:
            identities = tuple(dict.fromkeys(self._frame_identities(source, frame)))
            if len(identities) != 1:
                raise ValueError(
                    f"static source {source.name!r} requires one keyed row for a local request"
                )
            selected = frame.copy()
            expected_ids = identities
        else:
            selected = self._select_requested_series(source, frame, request, strict_extra=False)
            expected_ids = self._request_identities(source, request)
        actual_ids = self._frame_identities(source, selected)
        if len(selected) != len(expected_ids) or set(actual_ids) != set(expected_ids):
            raise ValueError(f"static source {source.name!r} requires exactly one row per requested series")
        if request.series_ids:
            selected = self._order_static(source, selected, request)
        return self._role_frame(source, selected, ColumnRole.STATIC)

    def _collect_observed_provider(
        self,
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
            provider_frame = self._validate_frame(
                source,
                self._read_path(source.backtest_path),
                generated=False,
            )
            if source.availability in {
                AvailabilityPolicy.COLUMN,
                AvailabilityPolicy.GENERATOR_DEFINED,
            }:
                provider_frame = self._select_as_of_vintage(
                    source,
                    provider_frame,
                    request,
                    enforce_origin=True,
                )
            provider_frame = provider_frame.loc[
                provider_frame[source.time_col].isin(request.forecast_times)
            ].copy()
            provider_frame = self._select_requested_series(
                source,
                provider_frame,
                request,
                strict_extra=True,
            )
        identities = self._request_identities(source, request)
        for identity in identities:
            selected = self._filter_identity(source, provider_frame, identity)
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
                    available_at_col = self._available_at_col(source)
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

    @staticmethod
    def _filter_identity(
        source: DataSourceSpec,
        frame: pd.DataFrame,
        identity: Any,
    ) -> pd.DataFrame:
        if not source.series_id_cols:
            return frame
        if len(source.series_id_cols) == 1:
            return frame.loc[frame[source.series_id_cols[0]] == identity]
        mask = np.ones(len(frame), dtype=bool)
        for column, value in zip(source.series_id_cols, identity):
            mask &= frame[column].to_numpy() == value
        return frame.loc[mask]

    def _select_requested_series(
        self,
        source: DataSourceSpec,
        frame: pd.DataFrame,
        request: InformationSetRequest,
        *,
        strict_extra: bool,
    ) -> pd.DataFrame:
        expected_ids = self._request_identities(source, request)
        if not source.series_id_cols:
            return frame.copy()
        identities = self._frame_identities(source, frame)
        expected_set = set(expected_ids)
        if strict_extra and any(identity not in expected_set for identity in identities):
            raise ValueError(f"source {source.name!r} contains unrequested series in the requested horizon")
        mask = [identity in expected_set for identity in identities]
        return frame.loc[mask].copy()

    @staticmethod
    def _request_identities(
        source: DataSourceSpec,
        request: InformationSetRequest,
    ) -> tuple[Any, ...]:
        width = len(source.series_id_cols)
        if width == 0:
            if request.series_ids:
                raise ValueError(f"local source {source.name!r} cannot serve a multi-series request")
            return ((),)
        if not request.series_ids:
            raise ValueError(f"series source {source.name!r} requires request.series_ids")
        normalized = []
        for value in request.series_ids:
            if width == 1:
                if isinstance(value, tuple):
                    if len(value) != 1:
                        raise ValueError(f"series identity for source {source.name!r} must have width 1")
                    normalized.append(value[0])
                else:
                    normalized.append(value)
            else:
                if not isinstance(value, tuple) or len(value) != width:
                    raise ValueError(
                        f"series identity for source {source.name!r} must have width {width}"
                    )
                normalized.append(value)
        return tuple(normalized)

    @staticmethod
    def _frame_identities(source: DataSourceSpec, frame: pd.DataFrame) -> list[Any]:
        if not source.series_id_cols:
            return [()] * len(frame)
        if len(source.series_id_cols) == 1:
            return frame[source.series_id_cols[0]].tolist()
        return list(frame.loc[:, list(source.series_id_cols)].itertuples(index=False, name=None))

    def _require_history_per_series(
        self,
        source: DataSourceSpec,
        frame: pd.DataFrame,
        request: InformationSetRequest,
    ) -> None:
        expected = set(self._request_identities(source, request))
        actual = set(self._frame_identities(source, frame))
        missing = expected - actual
        if missing:
            raise ValueError(f"history source {source.name!r} is missing requested series")

    def _expected_temporal_keys(
        self,
        source: DataSourceSpec,
        request: InformationSetRequest,
    ) -> set[tuple[Any, ...]]:
        identities = self._request_identities(source, request)
        if not source.series_id_cols:
            return {(time,) for time in request.forecast_times}
        if len(source.series_id_cols) == 1:
            return {(identity, time) for identity in identities for time in request.forecast_times}
        return {
            (*identity, time)
            for identity in identities
            for time in request.forecast_times
        }

    @staticmethod
    def _actual_temporal_keys(
        source: DataSourceSpec,
        frame: pd.DataFrame,
    ) -> set[tuple[Any, ...]]:
        columns = [*source.series_id_cols, source.time_col]
        return set(frame.loc[:, columns].itertuples(index=False, name=None))

    def _order_temporal(
        self,
        source: DataSourceSpec,
        frame: pd.DataFrame,
        request: InformationSetRequest,
    ) -> pd.DataFrame:
        if not source.series_id_cols:
            return frame.sort_values(source.time_col, kind="stable").reset_index(drop=True)
        order = {identity: index for index, identity in enumerate(self._request_identities(source, request))}
        ordered = frame.copy()
        ordered["__series_order"] = [order[identity] for identity in self._frame_identities(source, ordered)]
        ordered = ordered.sort_values(["__series_order", source.time_col], kind="stable")
        return ordered.drop(columns="__series_order").reset_index(drop=True)

    def _order_static(
        self,
        source: DataSourceSpec,
        frame: pd.DataFrame,
        request: InformationSetRequest,
    ) -> pd.DataFrame:
        order = {identity: index for index, identity in enumerate(self._request_identities(source, request))}
        ordered = frame.copy()
        ordered["__series_order"] = [order[identity] for identity in self._frame_identities(source, ordered)]
        return ordered.sort_values("__series_order", kind="stable").drop(columns="__series_order").reset_index(drop=True)

    @staticmethod
    def _role_frame(
        source: DataSourceSpec,
        frame: pd.DataFrame,
        role: ColumnRole,
    ) -> pd.DataFrame:
        columns = list(source.series_id_cols)
        if source.time_col is not None:
            columns.append(source.time_col)
        available_at_col = SourceRegistry._available_at_col(source)
        if available_at_col is not None:
            columns.append(available_at_col)
        columns.extend(column.name for column in source.columns if column.role is role)
        return frame.loc[:, list(dict.fromkeys(columns))].reset_index(drop=True)


__all__ = ["FrameReader", "SourceGenerator", "SourceRegistry"]
