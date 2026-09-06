"""角色限定的时间→行位置索引；状态仍由信息集持有。"""
from collections.abc import Mapping

import pandas as pd

RowPositionCaches = dict[str, tuple[Mapping[str, pd.DataFrame], Mapping[int, int]]]


def row_position_lookup(caches: RowPositionCaches, cache_key: str) -> tuple[Mapping[str, pd.DataFrame], Mapping[int, int]]:
    cached = caches.get(cache_key)
    if cached is not None:
        return cached
    raise KeyError(f"no row-position cache registered for key {cache_key!r}")


def register_row_position_lookup(
    caches: RowPositionCaches,
    cache_key: str,
    frames: Mapping[str, pd.DataFrame],
    time_col: str,
    *,
    frame_name: str | None = None,
) -> RowPositionCaches:
    if cache_key in caches:
        return caches
    frame = frames[frame_name or cache_key]
    parsed = pd.DatetimeIndex(pd.to_datetime(frame[time_col].to_numpy()))
    lookup: dict[int, int] = {}
    for position, value in enumerate(parsed.asi8):
        if value in lookup:
            # 重复时间戳标记 -1，查询方仍走非恰好一行的 RAISE 路径。
            lookup[value] = -1
        else:
            lookup[value] = position
    return {**caches, cache_key: (frames, lookup)}
