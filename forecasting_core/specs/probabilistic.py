"""Typed YAML-level probabilistic configuration contract.

The deployment artifact uses
``forecasting_core.probability.spec.ProbabilisticSpec`` instead.
Legacy flat keys (``crossing_method``, ``conformal``) were swept from all
active YAMLs on 2026-09-01 and are rejected here; crossing behaviour is
declared via ``crossing:`` and CQR via ``calibration:``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from forecasting_core.specs._mapping import (
    FrozenMappingSpec,
    freeze_json_value,
    strict_mapping,
)
from forecasting_core.probability.spec import PROBABILISTIC_FIELDS, probabilistic_spec_from_mapping


class ProbabilisticConfigSpec(FrozenMappingSpec):
    """Strict typed probabilistic YAML section with Mapping compatibility."""

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | "ProbabilisticConfigSpec",
        *,
        source: str = "<constructor>",
    ) -> "ProbabilisticConfigSpec":
        if isinstance(value, cls):
            return value
        payload = strict_mapping(
            value,
            path="probabilistic",
            source=source,
            allowed=PROBABILISTIC_FIELDS,
        )
        # 唯一运行时解析器负责语义；默认值不写回原始 payload。
        try:
            probabilistic_spec_from_mapping(payload)
        except ValueError as exc:
            if str(exc).startswith("Unknown "):
                raise ValueError(f"Unknown fields in probabilistic from {source}: {exc}") from exc
            raise
        return cls(freeze_json_value(payload, "probabilistic"))


__all__ = ["ProbabilisticConfigSpec", "PROBABILISTIC_FIELDS"]
