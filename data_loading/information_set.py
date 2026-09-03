"""Immutable request and defensive information-set result containers."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import pandas as pd

from data_loading.providers import EndogenousFutureProvider


TargetAccess: TypeAlias = Literal["history_only", "supervised_labels"]
_TARGET_ACCESS_VALUES = frozenset({"history_only", "supervised_labels"})


def _normalize_series_id(value: Any) -> Any:
    if isinstance(value, list):
        value = tuple(value)
    if isinstance(value, tuple):
        value = tuple(_normalize_series_id(item) for item in value)
    if value is None or bool(pd.isna(value)):
        raise ValueError("series_ids entries must not be missing")
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError("series_ids entries must be hashable") from exc
    return value


@dataclass(frozen=True, slots=True, init=False)
class InformationSetRequest:
    forecast_origin: pd.Timestamp
    forecast_times: pd.DatetimeIndex
    series_ids: tuple[Any, ...]
    target_access: TargetAccess

    def __init__(
        self,
        forecast_origin: Any,
        forecast_times: pd.DatetimeIndex,
        series_ids: Sequence[Any],
        target_access: TargetAccess = "history_only",
    ) -> None:
        try:
            normalized_origin = pd.Timestamp(forecast_origin)
        except (TypeError, ValueError) as exc:
            raise ValueError("forecast_origin must be a valid Timestamp") from exc
        if pd.isna(normalized_origin):
            raise ValueError("forecast_origin must not be NaT")
        if not isinstance(forecast_times, pd.DatetimeIndex):
            raise TypeError("forecast_times must be a DatetimeIndex")
        normalized_times = forecast_times.copy(deep=True)
        if normalized_times.empty:
            raise ValueError("forecast_times must not be empty")
        if normalized_times.hasnans:
            raise ValueError("forecast_times must not contain NaT")
        if not normalized_times.is_monotonic_increasing or normalized_times.has_duplicates:
            raise ValueError("forecast_times must be strictly increasing")
        if isinstance(series_ids, (str, bytes)) or not isinstance(series_ids, Sequence):
            raise TypeError("series_ids must be a sequence")
        normalized_series_ids = tuple(_normalize_series_id(value) for value in series_ids)
        if len(normalized_series_ids) != len(set(normalized_series_ids)):
            raise ValueError("series_ids must not contain duplicates")
        if target_access not in _TARGET_ACCESS_VALUES:
            raise ValueError(f"unknown target_access: {target_access!r}")

        object.__setattr__(self, "forecast_origin", normalized_origin)
        object.__setattr__(self, "forecast_times", normalized_times)
        object.__setattr__(self, "series_ids", normalized_series_ids)
        object.__setattr__(self, "target_access", target_access)

    @property
    def N(self) -> int:
        return len(self.series_ids) or 1

    @property
    def H(self) -> int:
        return len(self.forecast_times)


@dataclass(frozen=True, slots=True)
class SourceLineage:
    source_name: str
    path_version: str
    path: str | None
    availability_policy: str | None
    includes_target_labels: bool


class MaterializedInformationSet:
    """Source-qualified frames with defensive copies at every access boundary."""

    __slots__ = (
        "_target_history",
        "_observed_past",
        "_known_future",
        "_static",
        "_observed_future_providers",
        "_lineage",
        "_sealed",
        "_row_position_caches",
    )

    def __init__(
        self,
        *,
        target_history: Mapping[str, pd.DataFrame],
        observed_past: Mapping[str, pd.DataFrame],
        known_future: Mapping[str, pd.DataFrame],
        static: Mapping[str, pd.DataFrame],
        observed_future_providers: Mapping[Any, EndogenousFutureProvider] | None = None,
        lineage: Sequence[SourceLineage],
    ) -> None:
        object.__setattr__(self, "_target_history", self._copy_frames(target_history))
        object.__setattr__(self, "_observed_past", self._copy_frames(observed_past))
        object.__setattr__(self, "_known_future", self._copy_frames(known_future))
        object.__setattr__(self, "_static", self._copy_frames(static))
        object.__setattr__(
            self,
            "_observed_future_providers",
            dict(observed_future_providers or {}),
        )
        object.__setattr__(self, "_lineage", tuple(lineage))
        object.__setattr__(self, "_sealed", True)
        # 性能缓存（2026-08-30 方案 A）：(cache_key) -> {ts_ns: row_position}。
        # 同一物理 source 可按角色投影为不同时间范围/列集，调用方须用
        # 角色限定 cache_key 隔离这些 frame。
        # 本对象不可变、帧内容终身不变，缓存可安全共享给同一 origin 的全部查询；
        # 相比在 Compiler 侧按 id(frame) 缓存，这里不受防御性 copy 影响。
        object.__setattr__(self, "_row_position_caches", {})

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("MaterializedInformationSet is immutable")
        object.__setattr__(self, name, value)

    @staticmethod
    def _copy_frames(frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        return {name: frame.copy(deep=True) for name, frame in frames.items()}

    def row_position_lookup(
        self,
        cache_key: str,
    ) -> tuple[Mapping[str, pd.DataFrame], Mapping[int, int]]:
        """返回 ``(frames, lookup)``：指定缓存域的帧字典与时间→行位置映射。

        性能契约（2026-08-30 方案 A）：同一 information_set 内帧内容不变，
        时间→位置映射只解析一次并共享给全部查询方（compiler 的逐 lag 取值）。
        ``frames`` 仍是防御性 copy（调用方对其做行提取），lookup 的 position
        直接索引 copy 后的帧是安全的——copy 保持行序与行数不变。
        """
        cached = self._row_position_caches.get(cache_key)
        if cached is not None:
            return cached
        raise KeyError(
            f"no row-position cache registered for key {cache_key!r}"
        )

    def register_row_position_lookup(
        self,
        cache_key: str,
        frames: Mapping[str, pd.DataFrame],
        time_col: str,
        *,
        frame_name: str | None = None,
    ) -> None:
        """为指定缓存域构建并登记时间→行位置映射。"""
        if cache_key in self._row_position_caches:
            return
        frame = frames[frame_name or cache_key]
        parsed = pd.DatetimeIndex(pd.to_datetime(frame[time_col].to_numpy()))
        lookup: dict[int, int] = {}
        for position, value in enumerate(parsed.asi8):
            if value in lookup:
                # 重复时间戳：标记为 -1，查询时走「非恰好一行」RAISE 路径
                lookup[value] = -1
            else:
                lookup[value] = position
        object.__setattr__(
            self, "_row_position_caches",
            {**self._row_position_caches, cache_key: (frames, lookup)},
        )

    @property
    def target_history(self) -> dict[str, pd.DataFrame]:
        return self._copy_frames(self._target_history)

    @property
    def observed_past(self) -> dict[str, pd.DataFrame]:
        return self._copy_frames(self._observed_past)

    @property
    def known_future(self) -> dict[str, pd.DataFrame]:
        return self._copy_frames(self._known_future)

    @property
    def static(self) -> dict[str, pd.DataFrame]:
        return self._copy_frames(self._static)

    @property
    def observed_future_providers(self) -> dict[Any, EndogenousFutureProvider]:
        return dict(self._observed_future_providers)

    @property
    def lineage(self) -> tuple[SourceLineage, ...]:
        return self._lineage


__all__ = [
    "InformationSetRequest",
    "MaterializedInformationSet",
    "SourceLineage",
    "TargetAccess",
]
