"""编译结果与作用域数据；不依赖任何执行后端。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal
from collections.abc import Sequence
import pandas as pd
from data_loading import SourceLineage
from feature_engineering.statistics.provider import HistoryStatisticsProvider

ProofMode = Literal["materialize", "validate_only"]

@dataclass(frozen=True, slots=True)
class FeatureSchema:
    feature_names: tuple[str, ...]
    categorical_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BatchEligibility:
    """Structured decision for one supervised compiler batch."""

    eligible: bool
    reason_codes: tuple[str, ...]
    trigger_fields: tuple[str, ...]
    origin_count: int
    call_count: int
    estimated_origin_call_count: int

    @property
    def reason_code(self) -> str:
        return self.reason_codes[0] if self.reason_codes else "eligible"

    def to_payload(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "reason_codes": list(self.reason_codes),
            "trigger_fields": list(self.trigger_fields),
            "origin_count": self.origin_count,
            "call_count": self.call_count,
            "estimated_origin_call_count": self.estimated_origin_call_count,
        }


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


@dataclass(slots=True)
class CompilationContext:
    """一次编译的帧与派生缓存；退出作用域即释放，不进入持久化状态。"""

    frames: dict[str, dict[str, Any]]
    auxiliary: dict[str, Any] = field(default_factory=dict)
    statistics_provider: HistoryStatisticsProvider | None = None
