"""预测原点解析合同：部署与回测共用的 origin 归一化。

自 model_testing.contracts.primitives 迁入（2026-09-27 边界审计）：origin 解析服务于
部署/final fit 路径，非回测专属；以 Protocol 描述数据源能力，本层不 import
data_loading。
"""
from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


class SupportsLatestTargetTime(Protocol):
    """提供目标历史最后已知时刻的数据源视图（如 data_loading.SourceRegistry）。"""

    def latest_target_time(self) -> pd.Timestamp: ...


def resolve_origin(registry: SupportsLatestTargetTime, raw_origin: Any) -> pd.Timestamp:
    """预测原点解析：None = 数据最后已知时刻，否则严格 Timestamp。"""
    if raw_origin is None:
        return registry.latest_target_time()
    return pd.Timestamp(raw_origin)
