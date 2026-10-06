"""批编译执行器；共享规划由 compiler 提供，快照隔离在批入口检查。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any, cast
import warnings

import numpy as np
import pandas as pd
from feature_engineering.kernels.seasonal import normalize_same_slot_spec, normalize_recent_state_spec

from data_loading import InformationSetRequest, MaterializedInformationSet
from forecasting_core.specs import AvailabilityPolicy, ColumnRole, DataSourceSpec
from feature_engineering.kernels.spectral import (
    fourier_features,
    normalize_band_periods,
    wavelet_energy_features,
)
from feature_engineering.kernels.history import expanding_statistics, rolling_statistics


from feature_engineering.compilation.contracts import CompiledFeatures, FeatureSchema, VisibilityProof, ProofMode
from forecasting_core.specs.feature_transformations import STAT_MIN_SAMPLES

class BatchExecutor:
    def __init__(self, compiler: Any):
        self.compiler = compiler

    def compile_batch(
        self,
        information_sets: Sequence[MaterializedInformationSet],
        requests: Sequence[InformationSetRequest],
        *,
        horizon_steps: Sequence[int] | None = None,
        visibility_cutoffs: Sequence[pd.Timestamp] | None = None,
        proof_mode: ProofMode = "materialize",
    ) -> tuple[CompiledFeatures, ...]:
        """批量编译无 provider 依赖的 origins。

        lag/known-future 按时间数组定位，rolling/difference 只对共享历史计算
        一次，再广播到该 origin 的全部 horizon 行。依赖逐步 provider 的策略与
        暂未批量化的历史变换显式回退 :meth:`compile`。默认 materialize 模式
        保持逐行审计合同；内部监督训练可用 validate-only 做同边界列式校验而不
        返回 proof 对象。
        """
        if len(information_sets) != len(requests):
            raise ValueError(
                "compile_batch requires information_sets and requests of equal length"
            )
        if visibility_cutoffs is not None and len(visibility_cutoffs) != len(requests):
            raise ValueError(
                "compile_batch requires one visibility cutoff per request"
            )
        if proof_mode not in {"materialize", "validate_only"}:
            raise ValueError(f"unsupported proof_mode: {proof_mode!r}")
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
            self.compiler._normalize_horizon_steps(horizon_steps, request.H)
            for request in requests
        )
        for request in requests:
            if request.H != self.compiler.problem.horizon:
                raise ValueError(
                    "information-set horizon does not match ForecastProblemSpec: "
                    f"request={request.H}, problem={self.compiler.problem.horizon}"
                )
            if request.target_access != "history_only":
                raise ValueError("feature compilation requires target_access='history_only'")

        eligibility = self.compiler._batch_eligibility(requests, selected_steps)
        if not eligibility.eligible:
            if proof_mode == "validate_only":
                raise ValueError(
                    "proof_mode='validate_only' requires vectorized batch compilation"
                )
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

        # 仅共享同一不可变物化快照且历史下界相同的请求；同名源不代表
        # 同一版本。生产训练的 union materialization 仍能整批共享。
        groups: dict[tuple[int, Any], list[int]] = {}
        for index, (information_set, request) in enumerate(zip(information_sets, requests)):
            groups.setdefault((id(information_set), request.history_start), []).append(index)
        if len(groups) > 1:
            results: dict[int, CompiledFeatures] = {}
            totals = dict.fromkeys(self.compiler.last_batch_stage_wall_seconds, 0.0)
            for indices in groups.values():
                compiled = self.compile_batch(
                    tuple(information_sets[index] for index in indices),
                    tuple(requests[index] for index in indices),
                    horizon_steps=horizon_steps,
                    visibility_cutoffs=None if visibility_cutoffs is None else tuple(
                        visibility_cutoffs[index] for index in indices
                    ),
                    proof_mode=proof_mode,
                )
                results.update(zip(indices, compiled))
                for name, elapsed in self.compiler.last_batch_stage_wall_seconds.items():
                    totals[name] += elapsed
            self.compiler.last_batch_stage_wall_seconds = totals
            return tuple(results[index] for index in range(len(requests)))

        with self.compiler._compilation_scope():
            prepare_started = perf_counter()
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
                        proof_mode,
                        role_frames=cached_frames[0],
                        static_frames=cached_frames[1],
                    )
                )
            self.compiler.last_batch_stage_wall_seconds["batch_prepare"] = (
                perf_counter() - prepare_started
            )
            features_started = perf_counter()
            self._compile_batch_lag_mapping(
                items,
                ColumnRole.TARGET,
                self.compiler.features.target_lags,
            )
            self._compile_batch_lag_mapping(
                items,
                ColumnRole.OBSERVED_PAST,
                self.compiler.features.observed_past_lags,
            )
            self._compile_batch_known_future(items)
            self._compile_batch_static(items)
            self._compile_batch_datetime(items)
            self._compile_batch_transformations(items)
            self.compiler.last_batch_stage_wall_seconds["batch_feature_columns"] = (
                perf_counter() - features_started
            )
            finish_started = perf_counter()
            compiled = tuple(self._finish_batch_item(item) for item in items)
            self.compiler.last_batch_stage_wall_seconds[
                "finish_and_proof_validation"
            ] = perf_counter() - finish_started
            return compiled

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
                self.compiler.compile(
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
        proof_mode: ProofMode,
        *,
        role_frames: dict[str, dict[str, Any]],
        static_frames: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        identities = self.compiler._request_identities(request)
        steps = np.asarray(selected_steps, dtype=np.int64)
        step_count = len(steps)
        row_identities = tuple(
            identity for identity in identities for _ in range(step_count)
        )
        selected_target_times = request.forecast_times.take(steps)
        target_times = pd.DatetimeIndex(
            selected_target_times.take(np.tile(np.arange(step_count), len(identities)))
        )
        history_anchors = pd.DatetimeIndex(
            [
                self.compiler._history_anchor_time(request, pd.Timestamp(target_time))
                for target_time in selected_target_times
            ]
        )
        frame_data: dict[str, Any] = {}
        for column in self.compiler.problem.series_id_cols:
            frame_data[column] = [
                self.compiler._identity_payload(identity)[column]
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
            "proof_mode": proof_mode,
            "visibility_cutoff": visibility_cutoff,
        }

    def _compile_batch_lag_mapping(
        self,
        items: Sequence[dict[str, Any]],
        role: ColumnRole,
        lag_mapping: Mapping[str, tuple[int, ...]],
    ) -> None:
        offset = pd.tseries.frequencies.to_offset(self.compiler.problem.freq)
        for column_name, lags in lag_mapping.items():
            source = self.compiler._source_for_column(column_name, role)
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
                cache_key = self.compiler._row_cache_key(source, role)
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
                        (anchors - lag * offset).take(
                            np.tile(np.arange(len(anchors)), len(item["identities"]))
                        )
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
                            selected = self.compiler._filter_identity(source, frame, identity)
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
            cache_key = self.compiler._row_cache_key(source, role)
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
        for source in self.compiler.data.sources:
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
                    selected = self.compiler._filter_identity(source, frame, identity)
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
                        else (
                            "available_at"
                            if source.availability is AvailabilityPolicy.GENERATOR_DEFINED
                            else None
                        )
                    )
                    if available_at_col is None:
                        # 只有无逐行可得时间的原点情景使用请求原点。
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
        for source in self.compiler.data.sources:
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
                    selected = self.compiler._filter_identity(source, frame, identity)
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
            for name in self.compiler.datetime_features:
                feature_name = f"dt_{name}"
                item["columns"][feature_name] = [
                    self.compiler._DATETIME_FEATURES[name](pd.Timestamp(target_time))
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
        transformations = self.compiler.features.transformations
        unknown = sorted(set(transformations) - self.compiler._TRANSFORMATION_KEYS)
        if unknown:
            raise ValueError(f"unsupported feature transformations: {unknown}")
        direct = transformations.get("direct")
        if direct is not None:
            # 空 batch 也保留原有 direct 校验，不能因无行而静默放行。
            self.compiler._direct_horizon_feature(direct)
            for item in items:
                self.compiler._compile_direct_transformations(
                    item["columns"], direct, vectorized=True,
                )

        advanced = transformations.get("advanced", {})
        if not isinstance(advanced, Mapping):
            raise TypeError("transformations.advanced must be a mapping")
        self.compiler._validate_advanced_transformations(advanced)
        self._compile_batch_history_transformations(items, advanced)
        for kind, normalize in (("same_slot", normalize_same_slot_spec), ("recent_state", normalize_recent_state_spec)):
            if kind not in advanced:
                continue
            spec = normalize(advanced[kind])
            for column in spec["columns"]:
                histories = self._batch_master_histories(items, column)
                for item in items:
                    per_row = []
                    for identity, anchor in zip(item["row_identities"], item["target_times"]):
                        per_row.append(self.compiler._causal_statistics(kind, spec, column, histories[identity],
                                                              item["request"].forecast_origin, anchor))
                    for name in per_row[0]:
                        item["columns"][name] = np.asarray([row[name] for row in per_row])
        for item in items:
            columns = item["columns"]
            if "block_weather" in advanced:
                with self.compiler._compilation_scope(frames=item["frames"]):
                    rows = [self.compiler._block_weather_values(identity, int(step), item["request"], item["information_set"])
                            for identity, step in zip(item["row_identities"], columns["horizon_step"])]
                for name in rows[0][0]:
                    columns[name] = np.asarray([values[name] for values, _ in rows])
                    self._add_batch_proof_column(item, name, "block_known_future", "known_future",
                                                 item["target_times"], [available for _, available in rows])
            self.compiler._compile_cyclical(columns, advanced.get("cyclical"), vectorized=True)
            self.compiler._compile_interaction_spec(columns, advanced.get("interaction"), vectorized=True)
            self.compiler._compile_polynomial(columns, advanced.get("polynomial"), vectorized=True)
            self.compiler._compile_named_interactions(
                columns, transformations.get("interactions", {}), vectorized=True,
            )
            request = cast(InformationSetRequest, item["request"])
            for feature_name in item["columns"]:
                if feature_name in item["proof_columns"] or feature_name in {
                    *self.compiler.problem.series_id_cols,
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
            columns = self.compiler._string_sequence(
                rolling_spec.get("columns", ()), "advanced.rolling.columns"
            )
            windows = self.compiler._positive_int_sequence(
                rolling_spec.get("windows", ()), "rolling.windows"
            )
            stats = self.compiler._validated_stats(
                rolling_spec.get("stats", ()), "rolling.stats", self.compiler.ROLLING_STATS
            )
            for column in columns:
                histories = self._batch_master_histories(items, column)
                for window in windows:
                    rolled_by_identity = {
                        identity: rolling_statistics(history, window, stats)
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
                                if position + 1 < window:
                                    raise ValueError(f"{column!r} has insufficient visible history for rolling window {window}")
                                value = float(
                                    rolled_by_identity[identity][stat].iloc[position]
                                )
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
            columns = self.compiler._string_sequence(
                expanding_spec.get("columns", ()), "advanced.expanding.columns"
            )
            stats = self.compiler._validated_stats(
                expanding_spec.get("stats", ()), "expanding.stats", self.compiler.ROLLING_STATS
            )
            for column in columns:
                histories = self._batch_master_histories(items, column)
                expanded_by_identity = {
                    identity: expanding_statistics(history, stats)
                    for identity, history in histories.items()
                }
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
                        row_slice = slice(
                            identity_index * step_count,
                            (identity_index + 1) * step_count,
                        )
                        for stat in stats:
                            if position + 1 < STAT_MIN_SAMPLES.get(stat, 1):
                                raise ValueError(f"history statistic {stat!r} has insufficient samples")
                            values_by_stat[stat][row_slice] = expanded_by_identity[identity][stat].iloc[position]
                    for stat, values in values_by_stat.items():
                        item["columns"][
                            f"{column}_expanding_{stat}"
                        ] = values

        difference_spec = advanced.get("difference")
        if difference_spec is not None:
            if not isinstance(difference_spec, Mapping):
                raise TypeError("transformations.advanced.difference must be a mapping")
            columns = self.compiler._string_sequence(
                difference_spec.get("columns", ()), "advanced.difference.columns"
            )
            periods = self.compiler._positive_int_sequence(
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

        self._compile_batch_origin_statistics(items, advanced)

        fourier_spec = advanced.get("fourier")
        if fourier_spec is not None:
            if not isinstance(fourier_spec, Mapping):
                raise TypeError("transformations.advanced.fourier must be a mapping")
            columns = self.compiler._string_sequence(
                fourier_spec.get("columns", ()), "advanced.fourier.columns"
            )
            windows = self.compiler._positive_int_sequence(
                fourier_spec.get("windows", ()), "fourier.windows"
            )
            band_periods = normalize_band_periods(fourier_spec.get("band_periods"))
            for column in columns:
                histories = self._batch_master_histories(items, column)
                for window in windows:
                    # 频域特征无增量算法，直接按 (identity, origin) 缓存窗内复算，
                    # 与逐行 compile() 的数值合同天然一致。
                    cache: dict[Any, dict[str, float]] = {}
                    for item in items:
                        step_count = len(item["selected_steps"])
                        feature_values: dict[str, np.ndarray] = {}
                        for identity_index, identity in enumerate(item["identities"]):
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
                            if position + 1 < window:
                                raise ValueError(
                                    f"{column!r} has insufficient visible history "
                                    f"for fourier window {window}"
                                )
                            key = (identity, position)
                            if key not in cache:
                                cache[key] = fourier_features(
                                    history.iloc[
                                        position - window + 1 : position + 1
                                    ].to_numpy(),
                                    top_k=int(fourier_spec.get("top_k", 5)),
                                    band_periods=band_periods,
                                )
                            row_slice = slice(
                                identity_index * step_count,
                                (identity_index + 1) * step_count,
                            )
                            for suffix, value in cache[key].items():
                                name = f"{column}_fft_{suffix}_{window}"
                                values = feature_values.get(name)
                                if values is None:
                                    values = np.empty(
                                        len(item["target_times"]), dtype=float
                                    )
                                    feature_values[name] = values
                                values[row_slice] = value
                        item["columns"].update(feature_values)

        wavelet_spec = advanced.get("wavelet")
        if wavelet_spec is not None:
            if not isinstance(wavelet_spec, Mapping):
                raise TypeError("transformations.advanced.wavelet must be a mapping")
            columns = self.compiler._string_sequence(
                wavelet_spec.get("columns", ()), "advanced.wavelet.columns"
            )
            windows = self.compiler._positive_int_sequence(
                wavelet_spec.get("windows", ()), "wavelet.windows"
            )
            wavelet_name = str(wavelet_spec.get("wavelet", "db4"))
            wavelet_level = wavelet_spec.get("level", 3)
            if (
                isinstance(wavelet_level, bool)
                or not isinstance(wavelet_level, int)
                or wavelet_level < 1
            ):
                raise ValueError("wavelet.level must be a positive integer")
            for column in columns:
                histories = self._batch_master_histories(items, column)
                for window in windows:
                    cache = {}
                    for item in items:
                        step_count = len(item["selected_steps"])
                        feature_values = {}
                        for identity_index, identity in enumerate(item["identities"]):
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
                            if position + 1 < window:
                                raise ValueError(
                                    f"{column!r} has insufficient visible history "
                                    f"for wavelet window {window}"
                                )
                            key = (identity, position)
                            if key not in cache:
                                cache[key] = wavelet_energy_features(
                                    history.iloc[
                                        position - window + 1 : position + 1
                                    ].to_numpy(),
                                    wavelet=wavelet_name,
                                    level=wavelet_level,
                                )
                            row_slice = slice(
                                identity_index * step_count,
                                (identity_index + 1) * step_count,
                            )
                            for suffix, value in cache[key].items():
                                name = f"{column}_wavelet_energy_{suffix}_{window}"
                                values = feature_values.get(name)
                                if values is None:
                                    values = np.empty(
                                        len(item["target_times"]), dtype=float
                                    )
                                    feature_values[name] = values
                                values[row_slice] = value
                        item["columns"].update(feature_values)

    def _compile_batch_origin_statistics(
        self,
        items: Sequence[dict[str, Any]],
        advanced: Mapping[str, Any],
    ) -> None:
        """按 (信息集, origin, identity) 计算一次，再广播至当前 horizon 行。

        不借用其他原点的最长历史：EWM/事件距离依赖完整前缀，信息集可能
        有不同起点或发布版本。复用 single 的统计规则，保持精确数值及异常。
        """
        selected = {
            kind: advanced[kind]
            for kind in ("percent_change", "time_since", "ewm", "rolling_quantile", "lagged_rolling")
            if advanced.get(kind) is not None
        }
        if not selected:
            return
        for item in items:
            step_count = len(item["selected_steps"])
            with self.compiler._compilation_scope(frames=item["frames"]):
                for index, identity in enumerate(item["identities"]):
                    row: dict[str, Any] = {}
                    self.compiler._compile_history_transformations(
                        row, selected, identity, item["request"], item["information_set"],
                    )
                    row_slice = slice(index * step_count, (index + 1) * step_count)
                    for name, value in row.items():
                        if name not in item["columns"]:
                            item["columns"][name] = np.empty(len(item["target_times"]), dtype=float)
                        item["columns"][name][row_slice] = value

    def _batch_master_histories(
        self,
        items: Sequence[dict[str, Any]],
        column_name: str,
    ) -> dict[Any, pd.Series]:
        matches = [
            (source, column.role)
            for source in self.compiler.data.sources
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
            selected = self.compiler._filter_identity(source, frame, identity)
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
            else pd.DatetimeIndex(source_times)
        )
        item["proof_columns"][feature_name] = (
            source_name,
            role,
            normalized_source_times,
            pd.DatetimeIndex(available_at),
        )

    @staticmethod
    def _validate_batch_proof_columns(
        proof_columns: Mapping[str, tuple[Any, Any, Any, Any]],
        feature_names: Sequence[str],
        forecast_origin: pd.Timestamp,
    ) -> None:
        """列式校验可见时间，不构造逐行 ``VisibilityProof``。"""
        for feature_name in feature_names:
            _source_name, _role, _source_times, available_at = proof_columns[
                feature_name
            ]
            available_index = pd.DatetimeIndex(available_at)
            if available_index.hasnans:
                raise ValueError(
                    f"visibility proof for {feature_name!r} requires available_at"
                )
            if bool((available_index > forecast_origin).any()):
                raise ValueError(
                    f"feature {feature_name!r} is available after forecast_origin"
                )

    def _finish_batch_item(self, item: dict[str, Any]) -> CompiledFeatures:
        columns = cast(dict[str, Any], item["columns"])
        frame = self.compiler._ordered_frame(pd.DataFrame(columns).infer_objects(copy=False))
        request = cast(InformationSetRequest, item["request"])
        key_columns = [*self.compiler.problem.series_id_cols, "target_time", "horizon_step"]
        feature_names = tuple(
            column for column in frame.columns if column not in key_columns
        )
        categorical_names = tuple(
            column
            for source in self.compiler.data.sources
            for column_spec in source.columns
            if column_spec.categorical and column_spec.name in feature_names
            for column in (column_spec.name,)
        )
        categorical_names = (
            *categorical_names,
            *(
                f"dt_{name}"
                for name in self.compiler.datetime_categorical
                if f"dt_{name}" in feature_names
            ),
        )
        proofs = []
        target_times = cast(pd.DatetimeIndex, item["target_times"])
        horizon_values = np.asarray(columns["horizon_step"], dtype=np.int64)
        if item["proof_mode"] == "materialize":
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
        if item["proof_mode"] == "materialize":
            self.compiler._validate_visibility_proofs(proofs, cutoff)
        else:
            self._validate_batch_proof_columns(
                item["proof_columns"],
                feature_names,
                cutoff,
            )
        return CompiledFeatures(
            frame=frame,
            schema=FeatureSchema(
                feature_names=feature_names,
                categorical_names=tuple(dict.fromkeys(categorical_names)),
            ),
            source_lineage=item["information_set"].lineage,
            visibility_proof=proofs,
        )
