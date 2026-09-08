"""Canonical data-source registry and strict as-of materialization."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from data_loading.information.information_set import (
    InformationSetRequest,
    MaterializedInformationSet,
    SourceLineage,
    WeatherSourceLineage,
)
from data_loading.information.providers import ObservedFutureProvider, collect_observed_provider
from data_loading.sources.source_io import FrameReader, SourceFrames, SourceGenerator
from data_loading.sources.discovery import TargetHistoryCoverage, latest_target_time, target_history_coverage
from data_loading.processing.validation import validate_frame
from data_loading.processing.visibility import select_as_of_vintage, history_frame
from data_loading.processing.alignment import known_future_frame, static_frame
from forecasting_core.specs.data import AvailabilityPolicy, ColumnRole, DataSourceSpec, DataSpec
from data_loading.weather_generator.generator import WeatherGenerator, weather_generator


_PATH_FIELDS = (
    ("history", "history_path"),
    ("backtest", "backtest_path"),
    ("future", "future_path"),
)

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
        if any(source.generator == 'weather' for source in data_spec.sources) and normalized_generators.get('weather') in (None, weather_generator):
            normalized_generators['weather'] = WeatherGenerator(base_dir)
        if any(not isinstance(name, str) or not callable(generator) for name, generator in normalized_generators.items()):
            raise TypeError("generators must map names to callables")
        self._data_spec = data_spec
        self._base_dir = Path(base_dir)
        self._generators = normalized_generators
        self._frames = SourceFrames(self._base_dir, reader)

    def materialize(
        self, request: InformationSetRequest, *, source_names: tuple[str, ...] | None = None,
    ) -> MaterializedInformationSet:
        if not isinstance(request, InformationSetRequest):
            raise TypeError("request must be an InformationSetRequest")
        sources = self._data_spec.sources
        if source_names is not None:
            if not isinstance(source_names, tuple) or any(not isinstance(name, str) for name in source_names):
                raise TypeError("source_names must be a tuple of source names")
            selected = set(source_names)
            configured = {source.name for source in sources}
            if not selected or len(selected) != len(source_names) or not selected <= configured:
                raise ValueError("source_names must be nonempty, unique, and configured")
            sources = tuple(source for source in sources if source.name in selected)
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

        for source in sources:
            frame, source_lineage = self._load_source(source, request)
            lineage.extend(source_lineage)
            roles = {column.role for column in source.columns}
            if ColumnRole.TARGET in roles:
                target_history[source.name] = history_frame(
                    source,
                    frame,
                    request,
                    role=ColumnRole.TARGET,
                )
            if ColumnRole.OBSERVED_PAST in roles:
                observed_frame = history_frame(
                    source,
                    frame,
                    request,
                    role=ColumnRole.OBSERVED_PAST,
                )
                observed_past[source.name] = observed_frame
                collect_observed_provider(
                    self._frames, source,
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
                known_future[source.name] = known_future_frame(source, frame, request)
            if ColumnRole.STATIC in roles:
                static[source.name] = static_frame(source, frame, request)

        return MaterializedInformationSet(
            target_history=target_history,
            observed_past=observed_past,
            known_future=known_future,
            static=static,
            observed_future_providers={
                identity: ObservedFutureProvider(
                    horizon=request.H,
                    trajectories=trajectories,
                    methods=observed_provider_methods[identity],
                    available_at=observed_provider_available_at[identity],
                )
                for identity, trajectories in observed_provider_values.items()
            },
            lineage=lineage,
        )

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def generators(self) -> dict[str, SourceGenerator]:
        """返回注册表副本，外部修改不影响本次运行。"""
        return dict(self._generators)

    def target_history_coverage(self) -> tuple[TargetHistoryCoverage, ...]:
        return target_history_coverage(self._data_spec, self._frames)

    def latest_target_time(self) -> pd.Timestamp:
        return latest_target_time(self._data_spec, self._frames)

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
            frame = validate_frame(
                source,
                generated,
                generated=True,
                path_version=None,
            )
            lineage = [self._lineage(source, "generated", None, request)]
            if source.generator == 'weather':
                evidence = generated.attrs.get('weather_evidence')
                if not isinstance(evidence, str) or not evidence:
                    raise ValueError('weather generator did not provide dependency evidence')
                lineage = [WeatherSourceLineage(
                    source_name=source.name, path_version='generated', path=None,
                    availability_policy=source.availability.value,
                    includes_target_labels=False, weather_evidence=evidence,
                )]
        else:
            frames = []
            lineage = []
            roles = {column.role for column in source.columns}
            path_fields = (
                _PATH_FIELDS
                if ColumnRole.KNOWN_FUTURE in roles
                else (("history", "history_path"),)
            )
            if source.inference_columns:
                version = "future" if request.data_phase == "future" else "history"
                path_fields = ((version, f"{version}_path"),)
                if getattr(source, f"{version}_path") is None:
                    raise ValueError(f"source {source.name!r} requires {version}_path for {request.data_phase}")
            for version, field_name in path_fields:
                configured_path = getattr(source, field_name)
                if configured_path is None:
                    continue
                cached = self._frames.read_validated(source, configured_path, version)
                frames.append(cached)
                lineage.append(self._lineage(source, version, configured_path, request))
            frame = pd.concat(frames, ignore_index=True)

        if source.availability in {AvailabilityPolicy.COLUMN, AvailabilityPolicy.GENERATOR_DEFINED}:
            roles = {column.role for column in source.columns}
            includes_target_labels = (
                ColumnRole.TARGET in roles
                and request.target_access == "supervised_labels"
            )
            frame = select_as_of_vintage(
                source,
                frame,
                request,
                include_target_labels=includes_target_labels,
            )
        if source.inference_columns and request.target_access == "history_only":
            # 训练请求（supervised_labels）读 rt_ 实测；推理请求读 pred_ 预报物理列。
            mapping = dict(source.inference_columns)
            for model_column, physical_column in mapping.items():
                if physical_column not in frame.columns:
                    raise ValueError(
                        f"source {source.name!r} inference column {physical_column!r} missing from file"
                    )
                frame[model_column] = frame[physical_column]
            if source.time_col is not None:
                requested = frame.loc[frame[source.time_col].isin(request.forecast_times)]
                if not requested.empty and requested[list(mapping)].isna().any(axis=None):
                    raise ValueError(
                        f"source {source.name!r} inference columns contain missing values "
                        "at requested forecast times"
                    )
        return frame, lineage

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
            includes_target_labels=(
                any(column.role is ColumnRole.TARGET for column in source.columns)
                and request.target_access == "supervised_labels"
            ),
        )


__all__ = ["FrameReader", "SourceGenerator", "SourceRegistry"]
