"""预测原点解析与数据源能力协议。"""
from __future__ import annotations

from typing import Any, Protocol, cast
import pandas as pd


class SupportsLatestTargetTime(Protocol):
    """提供目标历史最后已知时刻的数据源视图（如 data_loading.SourceRegistry）。"""

    def latest_target_time(self) -> pd.Timestamp: ...


def resolve_origin(registry: SupportsLatestTargetTime, raw_origin: Any) -> pd.Timestamp:
    """预测原点解析：None = 数据最后已知时刻，否则严格 Timestamp。"""
    origin = pd.Timestamp(registry.latest_target_time() if raw_origin is None else raw_origin)
    if bool(pd.isna(origin)):
        raise ValueError("forecast origin must be a finite timestamp")
    return cast(pd.Timestamp, origin)
