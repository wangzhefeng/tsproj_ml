"""全前缀历史统计的只读注入合同；编译器不拥有状态更新生命周期。"""
from typing import Any, Protocol

import pandas as pd

from data_loading.information.information_set import SourceLineage


class HistoryStatisticsProvider(Protocol):
    def require_binding(self, config_fingerprint: str, origin: pd.Timestamp) -> None:
        """快照必须属于当前配置与发报原点，不能借未来状态。"""
        ...

    @property
    def source_lineage(self) -> SourceLineage:
        """完整前缀的内容身份，不仅是保留的尾部历史。"""
        ...

    def value(self, kind: str, column: str, stat: str, *, origin: pd.Timestamp,
              identity: Any, parameter: float | None = None) -> float:
        """读取原点可见统计，不更新状态、不接纳预测值。"""
        ...
