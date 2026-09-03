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
        # 性能（2026-09-01）：known_future property 每次访问都对全部帧做 deep copy，
        # 而编译按 (identity, horizon_step) 逐行进入 _compile_known_future——
        # 5 折 × 96 步 = 480 次全帧拷贝是多外生列场景的主要耗时。
        # 按 information_set 身份缓存一次取值（帧不可变，copy 一次即足够）。
        # compile 作用域的帧缓存：compile() 进入时按当前 information_set 取一次
        # 三角色帧（property 会 deep copy），作用域内全部逐行查询共享这一份。
        self._compile_scope_frames: dict[str, dict[str, Any]] = {}
        self._compile_scope_aux: dict[str, Any] = {}

    def _prime_compile_scope(self, information_set: MaterializedInformationSet) -> None:
        """compile() 进入时调用：刷新当前信息集的帧与派生缓存。"""
        self._compile_scope_frames = {
            ColumnRole.TARGET.value: information_set.target_history,
            ColumnRole.OBSERVED_PAST.value: information_set.observed_past,
            ColumnRole.KNOWN_FUTURE.value: information_set.known_future,
        }
        # Compiler 会跨 backtest origin 复用；派生缓存只在单次 compile 内有效。
        self._compile_scope_aux = {}

    def _role_frames(
        self,
        information_set: MaterializedInformationSet,
        role: ColumnRole,
    ) -> dict[str, Any]:
        """返回 compile 作用域内的角色帧（避免逐行 property 访问触发 deep copy）。"""
        frames = self._compile_scope_frames.get(role.value)
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
    ) -> CompiledFeatures:
        if not isinstance(information_set, MaterializedInformationSet):
            raise TypeError("information_set must be a MaterializedInformationSet")
        if not isinstance(request, InformationSetRequest):
            raise TypeError("request must be an InformationSetRequest")
        self._prime_compile_scope(information_set)
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

    def compile_batch(
        self,
        information_sets: Sequence[MaterializedInformationSet],
        requests: Sequence[InformationSetRequest],
        *,
        horizon_steps: Sequence[int] | None = None,
        visibility_cutoffs: Sequence[pd.Timestamp] | None = None,
    ) -> tuple[CompiledFeatures, ...]:
        """批量编译无 provider 依赖的 origins，保持逐行审计合同不变。

        lag/known-future 按时间数组定位，rolling/difference 只对共享历史计算
        一次，再广播到该 origin 的全部 horizon 行。依赖逐步 provider 的策略与
        暂未批量化的历史变换显式回退 :meth:`compile`。
        """
        if len(information_sets) != len(requests):
            raise ValueError(
                "compile_batch requires information_sets and requests of equal length"
            )
        if visibility_cutoffs is not None and len(visibility_cutoffs) != len(requests):
            raise ValueError(
                "compile_batch requires one visibility cutoff per request"
            )
        batch_cutoffs = (
            tuple(visibility_cutoffs)
            if visibility_cutoffs is not None
            else (None,) * len(requests)
        )
        for information_set, request in zip(information_sets, requests):
            if not isinstance(information_set, MaterializedInformationSet):
                raise TypeError("information_set must be a MaterializedInformationSet")
            if not isinstance(request, InformationSetRequest):
                raise TypeError("request must be an InformationSetRequest")
        selected_steps = tuple(
            self._normalize_horizon_steps(horizon_steps, request.H)
            for request in requests
        )
        for request in requests:
            if request.H != self.problem.horizon:
                raise ValueError(
                    "information-set horizon does not match ForecastProblemSpec: "
                    f"request={request.H}, problem={self.problem.horizon}"
                )
            if request.target_access != "history_only":
                raise ValueError("feature compilation requires target_access='history_only'")

        if self._batch_requires_fallback(requests, selected_steps):
            warnings.warn(
                "compile_batch is falling back to per-request compile because the "
                "strategy or feature set requires sequential history/provider semantics",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._compile_batch_fallback(
                information_sets,
                requests,
                selected_steps,
                batch_cutoffs,
            )

        previous_scope = self._compile_scope_frames
        previous_aux = self._compile_scope_aux
        try:
            items = []
            frame_cache: dict[
                int,
                tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame]],
            ] = {}
            for information_set, request, steps, cutoff in zip(
                information_sets,
                requests,
                selected_steps,
                batch_cutoffs,
            ):
                cache_key = id(information_set)
                cached_frames = frame_cache.get(cache_key)
                if cached_frames is None:
                    cached_frames = (
                        {
                            ColumnRole.TARGET.value: information_set.target_history,
                            ColumnRole.OBSERVED_PAST.value: information_set.observed_past,
                            ColumnRole.KNOWN_FUTURE.value: information_set.known_future,
                        },
                        information_set.static,
                    )
                    frame_cache[cache_key] = cached_frames
                items.append(
                    self._prepare_batch_item(
                        information_set,
                        request,
                        steps,
                        cutoff,
                        role_frames=cached_frames[0],
                        static_frames=cached_frames[1],
                    )
                )
            self._compile_batch_lag_mapping(
                items,
                ColumnRole.TARGET,
                self.features.target_lags,
            )
            self._compile_batch_lag_mapping(
                items,
                ColumnRole.OBSERVED_PAST,
                self.features.observed_past_lags,
            )
            self._compile_batch_known_future(items)
            self._compile_batch_static(items)
            self._compile_batch_datetime(items)
            self._compile_batch_transformations(items)
            return tuple(self._finish_batch_item(item) for item in items)
        finally:
            self._compile_scope_frames = previous_scope
            self._compile_scope_aux = previous_aux

    def _batch_requires_fallback(
        self,
        requests: Sequence[InformationSetRequest],
        selected_steps: Sequence[tuple[int, ...]],
    ) -> bool:
        advanced = self.features.transformations.get("advanced", {})
        if isinstance(advanced, Mapping) and any(
            advanced.get(kind) is not None
            for kind in ("percent_change", "time_since", "ewm")
        ):
            return True
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
                for lags in lag_mapping.values():
                    for lag in lags:
                        source_times = anchors - lag * pd.tseries.frequencies.to_offset(
                            self.problem.freq
                        )
                        if bool((source_times > request.forecast_origin).any()):
                            return True
                if role is ColumnRole.OBSERVED_PAST:
                    continue
        return False

    def _compile_batch_fallback(
        self,
        information_sets: Sequence[MaterializedInformationSet],
        requests: Sequence[InformationSetRequest],
        selected_steps: Sequence[tuple[int, ...]],
        visibility_cutoffs: Sequence[pd.Timestamp | None],
    ) -> tuple[CompiledFeatures, ...]:
        compiled_items = []
        for information_set, request, steps, cutoff in zip(
            information_sets,
            requests,
            selected_steps,
            visibility_cutoffs,
        ):
            compiled_items.append(
                self.compile(
                    information_set,
                    request,
                    horizon_steps=tuple(step + 1 for step in steps),
                    visibility_cutoff=cutoff,
                )
            )
        return tuple(compiled_items)

    def _prepare_batch_item(
        self,
        information_set: MaterializedInformationSet,
        request: InformationSetRequest,
        selected_steps: tuple[int, ...],
        visibility_cutoff: pd.Timestamp | None,
        *,
        role_frames: dict[str, dict[str, Any]],
        static_frames: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        identities = self._request_identities(request)
        steps = np.asarray(selected_steps, dtype=np.int64)
        step_count = len(steps)
        row_identities = tuple(
            identity for identity in identities for _ in range(step_count)
        )
        selected_target_times = request.forecast_times.take(steps)
        target_times = pd.DatetimeIndex(
            np.tile(selected_target_times.asi8, len(identities))
        )
        history_anchors = pd.DatetimeIndex(
            [
                self._history_anchor_time(request, pd.Timestamp(target_time))
                for target_time in selected_target_times
            ]
        )
        frame_data: dict[str, Any] = {}
        for column in self.problem.series_id_cols:
            frame_data[column] = [
                self._identity_payload(identity)[column]
                for identity in identities
                for _ in range(step_count)
            ]
        frame_data["target_time"] = target_times
        frame_data["horizon_step"] = np.tile(steps + 1, len(identities))
        return {
            "information_set": information_set,
            "request": request,
            "selected_steps": selected_steps,
            "identities": identities,
            "row_identities": row_identities,
            "target_times": target_times,
            "history_anchors": history_anchors,
            "frames": role_frames,
            "static_frames": static_frames,
            "columns": frame_data,
            "proof_columns": {},
            "visibility_cutoff": visibility_cutoff,
        }

    def _compile_batch_lag_mapping(
        self,
        items: Sequence[dict[str, Any]],
        role: ColumnRole,
        lag_mapping: Mapping[str, tuple[int, ...]],
    ) -> None:
        offset = pd.tseries.frequencies.to_offset(self.problem.freq)
        for column_name, lags in lag_mapping.items():
            source = self._source_for_column(column_name, role)
            shared_frame = None
            shared_lookup = None
            if (
                items
                and not source.series_id_cols
                and source.availability is AvailabilityPolicy.SOURCE_TIME
            ):
                master = max(
                    items,
                    key=lambda item: pd.Timestamp(item["request"].forecast_origin),
                )
                shared_frame = master["frames"][role.value].get(source.name)
                if shared_frame is None:
                    raise ValueError(
                        f"history source {source.name!r} was not materialized"
                    )
                cache_key = self._row_cache_key(source, role)
                information_set = cast(
                    MaterializedInformationSet, master["information_set"]
                )
                try:
                    _frames, shared_lookup = information_set.row_position_lookup(
                        cache_key
                    )
                except KeyError:
                    information_set.register_row_position_lookup(
                        cache_key,
                        {cache_key: shared_frame},
                        cast(str, source.time_col),
                    )
                    _frames, shared_lookup = information_set.row_position_lookup(
                        cache_key
                    )
            for lag in lags:
                feature_name = f"{column_name}__lag_{lag}"
                for item in items:
                    steps = cast(tuple[int, ...], item["selected_steps"])
                    anchors = cast(pd.DatetimeIndex, item["history_anchors"])
                    source_times = pd.DatetimeIndex(
                        np.tile((anchors - lag * offset).asi8, len(item["identities"]))
                    )
                    values = np.empty(len(source_times), dtype=object)
                    available_at = np.empty(len(source_times), dtype=object)
                    step_count = len(steps)
                    frame = item["frames"][role.value].get(source.name)
                    if frame is None:
                        raise ValueError(
                            f"history source {source.name!r} was not materialized"
                        )
                    for identity_index, identity in enumerate(item["identities"]):
                        row_slice = slice(
                            identity_index * step_count,
                            (identity_index + 1) * step_count,
                        )
                        if shared_frame is not None and shared_lookup is not None:
                            selected = shared_frame
                            positions = np.fromiter(
                                (
                                    shared_lookup.get(int(timestamp_ns), -1)
                                    for timestamp_ns in source_times[row_slice].asi8
                                ),
                                dtype=np.int64,
                                count=step_count,
                            )
                            if bool(
                                ((positions < 0) | (positions >= len(selected))).any()
                            ):
                                missing = int(np.flatnonzero(positions < 0)[0])
                                raise ValueError(
                                    f"source {source.name!r} requires exactly one row at "
                                    f"{source_times[row_slice][missing]} for series "
                                    f"{identity!r}"
                                )
                        else:
                            selected = self._filter_identity(source, frame, identity)
                            positions = self._batch_temporal_positions(
                                source,
                                selected,
                                source_times[row_slice],
                                item["information_set"],
                                role,
                            )
                        source_values = selected[column_name].to_numpy(copy=False)
                        values[row_slice] = source_values[positions]
                        if source.availability is AvailabilityPolicy.SOURCE_TIME:
                            available_at[row_slice] = source_times[row_slice].to_pydatetime()
                        else:
                            available_at_col = (
                                source.available_at_col
                                if source.availability is AvailabilityPolicy.COLUMN
                                else "available_at"
                            )
                            available_at[row_slice] = selected[
                                available_at_col
                            ].to_numpy(copy=False)[positions]
                    item["columns"][feature_name] = values
                    self._add_batch_proof_column(
                        item,
                        feature_name,
                        source.name,
                        role.value,
                        source_times,
                        available_at,
                    )

    def _batch_temporal_positions(
        self,
        source: DataSourceSpec,
        frame: pd.DataFrame,
        source_times: pd.DatetimeIndex,
        information_set: MaterializedInformationSet,
        role: ColumnRole,
    ) -> np.ndarray:
        if source.time_col is None:
            raise ValueError(f"temporal source {source.name!r} has no time_col")
        if not source.series_id_cols:
            cache_key = self._row_cache_key(source, role)
            try:
                _frames, lookup = information_set.row_position_lookup(cache_key)
            except KeyError:
                information_set.register_row_position_lookup(
                    cache_key,
                    {cache_key: frame},
                    source.time_col,
                )
                _frames, lookup = information_set.row_position_lookup(cache_key)
            positions = np.fromiter(
                (lookup.get(int(timestamp_ns), -1) for timestamp_ns in source_times.asi8),
                dtype=np.int64,
                count=len(source_times),
            )
        else:
            time_index = pd.DatetimeIndex(
                pd.to_datetime(frame[source.time_col].to_numpy())
            )
            if time_index.has_duplicates:
                positions = np.full(len(source_times), -1, dtype=np.int64)
            else:
                positions = time_index.get_indexer(source_times)
        if bool(((positions < 0) | (positions >= len(frame))).any()):
            missing_position = int(np.flatnonzero(positions < 0)[0])
            raise ValueError(
                f"source {source.name!r} requires exactly one row at "
                f"{source_times[missing_position]} for series"
            )
        return positions

    def _compile_batch_known_future(
        self,
        items: Sequence[dict[str, Any]],
    ) -> None:
        for source in self.data.sources:
            columns = tuple(
                column
                for column in source.columns
                if column.role is ColumnRole.KNOWN_FUTURE
            )
            if not columns:
                continue
            for item in items:
                request = cast(InformationSetRequest, item["request"])
                step_count = len(item["selected_steps"])
                frame = item["frames"][ColumnRole.KNOWN_FUTURE.value].get(
                    source.name
                )
                if frame is None:
                    raise ValueError(
                        f"known_future source {source.name!r} was not materialized"
                    )
                positions = np.empty(len(item["target_times"]), dtype=np.int64)
                available_at = np.empty(len(positions), dtype=object)
                for identity_index, identity in enumerate(item["identities"]):
                    row_slice = slice(
                        identity_index * step_count,
                        (identity_index + 1) * step_count,
                    )
                    selected = self._filter_identity(source, frame, identity)
                    time_index = pd.DatetimeIndex(
                        pd.to_datetime(selected[source.time_col].to_numpy())
                    )
                    if time_index.has_duplicates:
                        selected_positions = np.full(step_count, -1, dtype=np.int64)
                    else:
                        selected_positions = time_index.get_indexer(
                            item["target_times"][row_slice]
                        )
                    if bool(
                        (
                            (selected_positions < 0)
                            | (selected_positions >= len(selected))
                        ).any()
                    ):
                        missing = int(np.flatnonzero(selected_positions < 0)[0])
                        raise ValueError(
                            f"source {source.name!r} requires exactly one row at "
                            f"{item['target_times'][row_slice][missing]} for series "
                            f"{identity!r}"
                        )
                    positions[row_slice] = selected_positions
                    available_at_col = (
                        source.available_at_col
                        if source.availability is AvailabilityPolicy.COLUMN
                        else None
                    )
                    if available_at_col is None:
                        # generated known-future 天然在各自 forecast origin 可得；
                        # batch 可共享 union materialization，但 proof 仍按请求原点。
                        available_at[row_slice] = request.forecast_origin
                    else:
                        available_at[row_slice] = selected[
                            available_at_col
                        ].to_numpy(copy=False)[selected_positions]
                    for column in columns:
                        values = item["columns"].get(column.name)
                        if values is None:
                            values = np.empty(len(positions), dtype=object)
                        values[row_slice] = selected[column.name].to_numpy(
                            copy=False
                        )[selected_positions]
                        item["columns"][column.name] = values
                for column in columns:
                    self._add_batch_proof_column(
                        item,
                        column.name,
                        source.name,
                        ColumnRole.KNOWN_FUTURE.value,
                        item["target_times"],
                        available_at,
                    )

    def _compile_batch_static(self, items: Sequence[dict[str, Any]]) -> None:
        for source in self.data.sources:
            columns = tuple(
                column for column in source.columns if column.role is ColumnRole.STATIC
            )
            if not columns:
                continue
            for item in items:
                request = cast(InformationSetRequest, item["request"])
                frame = item["static_frames"].get(source.name)
                if frame is None:
                    raise ValueError(f"static source {source.name!r} was not materialized")
                step_count = len(item["selected_steps"])
                values_by_column = {
                    column.name: np.empty(len(item["target_times"]), dtype=object)
                    for column in columns
                }
                for identity_index, identity in enumerate(item["identities"]):
                    selected = self._filter_identity(source, frame, identity)
                    if len(selected) != 1:
                        raise ValueError(
                            f"static source {source.name!r} requires exactly one row "
                            f"for {identity!r}"
                        )
                    row_slice = slice(
                        identity_index * step_count,
                        (identity_index + 1) * step_count,
                    )
                    source_row = selected.iloc[0]
                    for column in columns:
                        values_by_column[column.name][row_slice] = source_row[column.name]
                for column in columns:
                    item["columns"][column.name] = values_by_column[column.name]
                    self._add_batch_proof_column(
                        item,
                        column.name,
                        source.name,
                        ColumnRole.STATIC.value,
                        (None,) * len(item["target_times"]),
                        (request.forecast_origin,) * len(item["target_times"]),
                    )

    def _compile_batch_datetime(self, items: Sequence[dict[str, Any]]) -> None:
        for item in items:
            request = cast(InformationSetRequest, item["request"])
            for name in self.datetime_features:
                feature_name = f"dt_{name}"
                item["columns"][feature_name] = [
                    self._DATETIME_FEATURES[name](pd.Timestamp(target_time))
                    for target_time in item["target_times"]
                ]
                self._add_batch_proof_column(
                    item,
                    feature_name,
                    "calendar",
                    ColumnRole.KNOWN_FUTURE.value,
                    item["target_times"],
                    (request.forecast_origin,) * len(item["target_times"]),
                )

    def _compile_batch_transformations(
        self,
        items: Sequence[dict[str, Any]],
    ) -> None:
        transformations = self.features.transformations
        unknown = sorted(set(transformations) - self._TRANSFORMATION_KEYS)
        if unknown:
            raise ValueError(f"unsupported feature transformations: {unknown}")
        direct = transformations.get("direct")
        if direct is not None:
            if not isinstance(direct, Mapping):
                raise TypeError("transformations.direct must be a mapping")
            layout = direct.get("layout")
            if layout not in {"independent_models", "single_model_horizon"}:
                raise ValueError(f"unsupported direct layout: {layout!r}")
            if layout == "single_model_horizon":
                horizon_feature = direct.get("horizon_feature", {})
                if not isinstance(horizon_feature, Mapping):
                    raise TypeError(
                        "transformations.direct.horizon_feature must be a mapping"
                    )
                name = str(horizon_feature.get("name", "forecast_horizon_idx"))
                for item in items:
                    values = np.asarray(
                        item["columns"]["horizon_step"], dtype=float
                    )
                    item["columns"][name] = values
                    if bool(horizon_feature.get("cyclical", False)):
                        period = float(self.problem.horizon)
                        item["columns"][f"{name}_sin"] = np.sin(
                            2.0 * np.pi * values / period
                        )
                        item["columns"][f"{name}_cos"] = np.cos(
                            2.0 * np.pi * values / period
                        )

        advanced = transformations.get("advanced", {})
        if not isinstance(advanced, Mapping):
            raise TypeError("transformations.advanced must be a mapping")
        supported = {
            "rolling",
            "expanding",
            "difference",
            "percent_change",
            "time_since",
            "ewm",
            "cyclical",
            "interaction",
            "polynomial",
        }
        unknown_advanced = sorted(set(advanced) - supported)
        if unknown_advanced:
            raise ValueError(
                f"unsupported advanced transformations: {unknown_advanced}"
            )
        self._compile_batch_history_transformations(items, advanced)
        for item in items:
            self._compile_batch_row_transformations(item["columns"], advanced)
            self._compile_batch_named_interactions(
                item["columns"], transformations.get("interactions", {})
            )
            request = cast(InformationSetRequest, item["request"])
            for feature_name in item["columns"]:
                if feature_name in item["proof_columns"] or feature_name in {
                    *self.problem.series_id_cols,
                    "target_time",
                    "horizon_step",
                }:
                    continue
                self._add_batch_proof_column(
                    item,
                    feature_name,
                    "derived_visible_features",
                    "derived",
                    (request.forecast_origin,) * len(item["target_times"]),
                    (request.forecast_origin,) * len(item["target_times"]),
                )

    def _compile_batch_history_transformations(
        self,
        items: Sequence[dict[str, Any]],
        advanced: Mapping[str, Any],
    ) -> None:
        rolling_spec = advanced.get("rolling")
        if rolling_spec is not None:
            if not isinstance(rolling_spec, Mapping):
                raise TypeError("transformations.advanced.rolling must be a mapping")
            columns = self._string_sequence(
                rolling_spec.get("columns", ()), "advanced.rolling.columns"
            )
            windows = self._positive_int_sequence(
                rolling_spec.get("windows", ()), "rolling.windows"
            )
            stats = self._string_sequence(
                rolling_spec.get("stats", ()), "rolling.stats"
            )
            for column in columns:
                histories = self._batch_master_histories(items, column)
                for window in windows:
                    rolled_by_identity = {
                        identity: self._batch_rolling_series(history, window, stats)
                        for identity, history in histories.items()
                    }
                    for stat in stats:
                        feature_name = f"{column}_rolling_{stat}_{window}"
                        for item in items:
                            values = np.empty(len(item["target_times"]), dtype=float)
                            step_count = len(item["selected_steps"])
                            for identity_index, identity in enumerate(
                                item["identities"]
                            ):
                                history = histories[identity]
                                position = int(
                                    np.searchsorted(
                                        history.index.asi8,
                                        pd.Timestamp(
                                            item["request"].forecast_origin
                                        ).value,
                                        side="right",
                                    )
                                    - 1
                                )
                                if position < 0:
                                    raise ValueError(
                                        f"history column {column!r} has no visible values"
                                    )
                                value = float(
                                    rolled_by_identity[identity][stat].iloc[position]
                                )
                                # pandas rolling 的增量算法与逐片 Series 统计可能有
                                # 浮点末位差异；逐片复算只发生在 origin 粒度，用于
                                # 保持 compile() 的逐值合同，不再按 horizon 重算。
                                exact = self._statistic(
                                    history.iloc[
                                        max(0, position - window + 1) : position + 1
                                    ],
                                    stat,
                                )
                                if value != exact:
                                    value = exact
                                row_slice = slice(
                                    identity_index * step_count,
                                    (identity_index + 1) * step_count,
                                )
                                values[row_slice] = value
                            item["columns"][feature_name] = values

        expanding_spec = advanced.get("expanding")
        if expanding_spec is not None:
            if not isinstance(expanding_spec, Mapping):
                raise TypeError("transformations.advanced.expanding must be a mapping")
            columns = self._string_sequence(
                expanding_spec.get("columns", ()), "advanced.expanding.columns"
            )
            stats = self._string_sequence(
                expanding_spec.get("stats", ()), "expanding.stats"
            )
            for column in columns:
                histories = self._batch_master_histories(items, column)
                for item in items:
                    values_by_stat = {
                        stat: np.empty(len(item["target_times"]), dtype=float)
                        for stat in stats
                    }
                    step_count = len(item["selected_steps"])
                    for identity_index, identity in enumerate(item["identities"]):
                        history = histories[identity]
                        position = int(
                            np.searchsorted(
                                pd.DatetimeIndex(history.index).asi8,
                                pd.Timestamp(item["request"].forecast_origin).value,
                                side="right",
                            )
                            - 1
                        )
                        if position < 0:
                            raise ValueError(
                                f"history column {column!r} has no visible values"
                            )
                        visible = history.iloc[: position + 1]
                        row_slice = slice(
                            identity_index * step_count,
                            (identity_index + 1) * step_count,
                        )
                        for stat in stats:
                            # 保留逐行 compile() 的 pandas 统计合同，但只在
                            # origin 粒度计算一次，再广播到该 origin 的 calls。
                            values_by_stat[stat][row_slice] = self._statistic(
                                visible,
                                stat,
                            )
                    for stat, values in values_by_stat.items():
                        item["columns"][
                            f"{column}_expanding_{stat}"
                        ] = values

        difference_spec = advanced.get("difference")
        if difference_spec is not None:
            if not isinstance(difference_spec, Mapping):
                raise TypeError("transformations.advanced.difference must be a mapping")
            columns = self._string_sequence(
                difference_spec.get("columns", ()), "advanced.difference.columns"
            )
            periods = self._positive_int_sequence(
                difference_spec.get("periods", ()), "difference.periods"
            )
            for column in columns:
                histories = self._batch_master_histories(items, column)
                for period in periods:
                    feature_name = f"{column}_diff_{period}"
                    for item in items:
                        values = np.empty(len(item["target_times"]), dtype=float)
                        step_count = len(item["selected_steps"])
                        for identity_index, identity in enumerate(item["identities"]):
                            history = histories[identity]
                            position = int(
                                np.searchsorted(
                                    history.index.asi8,
                                    pd.Timestamp(item["request"].forecast_origin).value,
                                    side="right",
                                )
                                - 1
                            )
                            if position < period:
                                raise ValueError(
                                    f"{column!r} has insufficient visible history "
                                    f"for diff {period}"
                                )
                            row_slice = slice(
                                identity_index * step_count,
                                (identity_index + 1) * step_count,
                            )
                            values[row_slice] = float(
                                history.iloc[position]
                                - history.iloc[position - period]
                            )
                        item["columns"][feature_name] = values

    def _batch_master_histories(
        self,
        items: Sequence[dict[str, Any]],
        column_name: str,
    ) -> dict[Any, pd.Series]:
        matches = [
            (source, column.role)
            for source in self.data.sources
            for column in source.columns
            if column.name == column_name
            and column.role in {ColumnRole.TARGET, ColumnRole.OBSERVED_PAST}
        ]
        if len(matches) != 1:
            raise ValueError(
                f"advanced history column {column_name!r} must resolve to one "
                "visible history source"
            )
        source, role = matches[0]
        if source.availability is not AvailabilityPolicy.SOURCE_TIME:
            raise ValueError(
                "batch history transformations require availability=source_time; "
                f"source {source.name!r} uses {source.availability.value!r}"
            )
        identities = tuple(
            dict.fromkeys(
                identity for item in items for identity in item["identities"]
            )
        )
        histories: dict[Any, pd.Series] = {}
        for identity in identities:
            candidates = [
                item
                for item in items
                if identity in item["identities"]
            ]
            master = max(
                candidates,
                key=lambda item: pd.Timestamp(item["request"].forecast_origin),
            )
            frame = master["frames"][role.value].get(source.name)
            if frame is None:
                raise ValueError(
                    f"history source {source.name!r} was not materialized"
                )
            selected = self._filter_identity(source, frame, identity)
            if source.time_col is None:
                raise ValueError(f"history source {source.name!r} has no time_col")
            times = pd.DatetimeIndex(
                pd.to_datetime(selected[source.time_col].to_numpy())
            )
            order = np.argsort(times.asi8, kind="stable")
            times = times[order]
            values = pd.to_numeric(
                selected[column_name].iloc[order], errors="raise"
            ).astype(float)
            visible = times <= pd.Timestamp(master["request"].forecast_origin)
            times = times[visible]
            values = values.iloc[np.flatnonzero(visible)].reset_index(drop=True)
            if len(values) == 0:
                raise ValueError(
                    f"history column {column_name!r} has no visible values"
                )
            if not np.isfinite(values.to_numpy()).all():
                raise ValueError(
                    f"history column {column_name!r} must be finite"
                )
            values.index = times
            histories[identity] = values
        return histories

    def _batch_rolling_series(
        self,
        history: pd.Series,
        window: int,
        stats: Sequence[str],
    ) -> dict[str, pd.Series]:
        results: dict[str, pd.Series] = {}
        rolling = history.rolling(window, min_periods=1)
        for stat in stats:
            if stat in {"max_diff", "min_diff"}:
                if window < 2:
                    results[stat] = pd.Series(0.0, index=history.index)
                else:
                    diffs = history.diff()
                    method = "max" if stat == "max_diff" else "min"
                    results[stat] = getattr(
                        diffs.rolling(window - 1, min_periods=1), method
                    )().fillna(0.0)
                continue
            if stat not in {"mean", "std", "min", "max", "median", "skew", "kurt"}:
                raise ValueError(f"unsupported history statistic: {stat!r}")
            results[stat] = getattr(rolling, stat)().fillna(0.0)
        return results

    def _compile_batch_row_transformations(
        self,
        frame: dict[str, Any],
        advanced: Mapping[str, Any],
    ) -> None:
        cyclical = advanced.get("cyclical")
        if cyclical is not None:
            if not isinstance(cyclical, Mapping):
                raise TypeError("transformations.advanced.cyclical must be a mapping")
            period = cyclical.get("period")
            if (
                isinstance(period, bool)
                or not isinstance(period, (int, float))
                or period <= 0
            ):
                raise ValueError("cyclical.period must be positive")
            for configured in self._string_sequence(
                cyclical.get("columns", ()), "cyclical.columns"
            ):
                column = configured if configured in frame else f"dt_{configured}"
                if column not in frame:
                    raise ValueError(
                        f"cyclical references unknown feature: {configured!r}"
                    )
                values = np.asarray(frame[column], dtype=float)
                frame[f"{column}_sin"] = np.sin(
                    2.0 * np.pi * values / float(period)
                )
                frame[f"{column}_cos"] = np.cos(
                    2.0 * np.pi * values / float(period)
                )

        interaction = advanced.get("interaction")
        if interaction is not None:
            if not isinstance(interaction, Mapping):
                raise TypeError(
                    "transformations.advanced.interaction must be a mapping"
                )
            pairs = interaction.get("column_pairs", ())
            if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
                raise TypeError("interaction.column_pairs must be a sequence")
            operations = set(
                self._string_sequence(
                    interaction.get("operations", ()), "interaction.operations"
                )
            )
            for pair in pairs:
                if (
                    isinstance(pair, (str, bytes))
                    or not isinstance(pair, Sequence)
                    or len(pair) != 2
                ):
                    raise TypeError(
                        "interaction column pairs must contain two feature names"
                    )
                left, right = pair
                if left not in frame or right not in frame:
                    raise ValueError(
                        f"interaction references unknown features: {pair}"
                    )
                left_values = np.asarray(frame[left], dtype=float)
                right_values = np.asarray(frame[right], dtype=float)
                if "add" in operations:
                    frame[f"{left}_add_{right}"] = left_values + right_values
                if "subtract" in operations:
                    frame[f"{left}_substract_{right}"] = left_values - right_values
                if "multiply" in operations:
                    frame[f"{left}_multiply_{right}"] = left_values * right_values
                if "divide" in operations:
                    frame[f"{left}_divide_{right}"] = left_values / (
                        right_values + 1e-8
                    )

        polynomial = advanced.get("polynomial")
        if polynomial is not None:
            if not isinstance(polynomial, Mapping):
                raise TypeError(
                    "transformations.advanced.polynomial must be a mapping"
                )
            degree = polynomial.get("degree", 2)
            if (
                isinstance(degree, bool)
                or not isinstance(degree, int)
                or degree < 2
            ):
                raise ValueError("polynomial.degree must be an integer >= 2")
            for column in self._string_sequence(
                polynomial.get("columns", ()), "polynomial.columns"
            ):
                if column not in frame:
                    raise ValueError(
                        f"polynomial references unknown feature: {column!r}"
                    )
                values = np.asarray(frame[column], dtype=float)
                for current_degree in range(2, degree + 1):
                    frame[f"{column}_pow_{current_degree}"] = (
                        values ** current_degree
                    )

    def _compile_batch_named_interactions(
        self,
        frame: dict[str, Any],
        interactions: Any,
    ) -> None:
        if not isinstance(interactions, Mapping):
            raise TypeError("transformations.interactions must be a mapping")
        for name, members in interactions.items():
            if isinstance(members, (str, bytes)) or not isinstance(members, Sequence):
                raise TypeError(f"interaction {name!r} must list feature names")
            if len(members) < 2:
                raise ValueError(f"interaction {name!r} requires at least two features")
            missing = [member for member in members if member not in frame]
            if missing:
                raise ValueError(
                    f"interaction {name!r} references unknown features: {missing}"
                )
            values = np.column_stack(
                [np.asarray(frame[member], dtype=float) for member in members]
            )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"interaction {name!r} requires finite numeric features"
                )
            frame[str(name)] = np.prod(values, axis=1)

    @staticmethod
    def _add_batch_proof_column(
        item: dict[str, Any],
        feature_name: str,
        source_name: str,
        role: str,
        source_times: Any,
        available_at: Any,
    ) -> None:
        normalized_source_times = (
            source_times
            if len(source_times) == 0 or source_times[0] is None
            else tuple(pd.DatetimeIndex(source_times))
        )
        item["proof_columns"][feature_name] = (
            source_name,
            role,
            normalized_source_times,
            tuple(pd.DatetimeIndex(available_at)),
        )

    def _finish_batch_item(self, item: dict[str, Any]) -> CompiledFeatures:
        columns = cast(dict[str, Any], item["columns"])
        frame = pd.DataFrame(columns).infer_objects(copy=False)
        request = cast(InformationSetRequest, item["request"])
        key_columns = [*self.problem.series_id_cols, "target_time", "horizon_step"]
        feature_names = tuple(
            column for column in frame.columns if column not in key_columns
        )
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
        proofs = []
        target_times = cast(pd.DatetimeIndex, item["target_times"])
        horizon_values = np.asarray(columns["horizon_step"], dtype=np.int64)
        for row_index, (target_time, horizon_step) in enumerate(
            zip(target_times, horizon_values)
        ):
            for feature_name in feature_names:
                source_name, role, source_times, available_at = item[
                    "proof_columns"
                ][feature_name]
                # compile() 的 derived proof 集合跨行累积，因此派生特征只在
                # 第一行登记一次；批路径必须保留这一既有审计布局。
                if role == "derived" and row_index > 0:
                    continue
                proofs.append(
                    VisibilityProof(
                        feature_name=feature_name,
                        source_name=source_name,
                        role=role,
                        target_time=target_time,
                        source_time=source_times[row_index],
                        forecast_origin=request.forecast_origin,
                        horizon_step=int(horizon_step),
                        available_at=available_at[row_index],
                    )
                )
        normalized_cutoff = pd.Timestamp(
            request.forecast_origin
            if item["visibility_cutoff"] is None
            else item["visibility_cutoff"]
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
            source_lineage=item["information_set"].lineage,
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
            "ewm",
            "cyclical",
            "interaction",
            "polynomial",
        }
        unknown = sorted(set(advanced) - supported)
        if unknown:
            raise ValueError(f"unsupported advanced transformations: {unknown}")
        for kind in (
            "rolling",
            "expanding",
            "difference",
            "percent_change",
            "time_since",
            "ewm",
        ):
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
                elif kind == "ewm":
                    # 指数加权统计（近期行为权重更高）：按半衰期计。
                    # ewm 半衰期以样本步数计；与 rolling 窗口一样只消费可见历史。
                    halflives = self._positive_number_sequence(
                        spec.get("halflives", ()), "ewm.halflives"
                    )
                    ewm_stats = self._string_sequence(spec.get("stats", ()), "ewm.stats")
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
        frames = self._role_frames(information_set, role)
        frame = frames.get(source.name)
        if frame is None:
            raise ValueError(f"history source {source.name!r} was not materialized")
        # 性能（2026-09-02 两级缓存）：
        # L1（instance 级，跨 origin 共享）：解析后的时间 ns 数组 + 排序位置。
        # 时间列解析与排序不依赖 forecast_origin，整个回测只需一次。
        # L2（compile 作用域级）：按当前 origin 过滤后的可见序列。
        cache_key = f"visible_history::{source.name}::{source.time_col}"
        parsed = self._compile_scope_aux.get(f"{cache_key}::parsed")
        order = self._compile_scope_aux.get(f"{cache_key}::order")
        if parsed is None or order is None:
            selected = self._filter_identity(source, frame, identity)
            if source.time_col is None:
                raise ValueError(f"history source {source.name!r} has no time_col")
            parsed = pd.DatetimeIndex(
                pd.to_datetime(selected[source.time_col].to_numpy())
            )
            order = np.argsort(parsed.asi8, kind="stable")
            parsed = parsed[order]
            self._compile_scope_aux[f"{cache_key}::parsed"] = parsed
            self._compile_scope_aux[f"{cache_key}::order"] = order
            self._compile_scope_aux[f"{cache_key}::selected"] = selected
        selected = self._compile_scope_aux[f"{cache_key}::selected"]
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

    @staticmethod
    def _statistic(values: pd.Series, stat: str) -> float:
        if stat in {"max_diff", "min_diff"}:
            # 爬坡统计：窗内相邻步最大/最小变化量。窗口 < 2 时无相邻对，
            # 与其他统计的防御回退一致（minimum_history_rows 已保证
            # window >= 2 才会启用 diff 类特征，此处为纵深防御）。
            if len(values) < 2:
                warnings.warn(
                    f"{stat!r} requires at least 2 samples (got {len(values)}); "
                    "falling back to 0.0",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return 0.0
            diffs = values.diff().dropna()
            return float(diffs.max() if stat == "max_diff" else diffs.min())
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
