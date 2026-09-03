"""Typed output configuration contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from forecasting_core.specs._mapping import (
    FrozenMappingSpec,
    freeze_json_value,
    strict_mapping,
    validate_nested_mappings,
)


OUTPUT_FIELDS = frozenset(
    {
        "identity",
        "directories",
        "overlay",
        "scenario_subpath",
        "results_root",
        "checkpoints_dir",
        "test_results_dir",
        "pred_results_dir",
    }
)

_OUTPUT_NESTED_FIELDS: dict[str, frozenset[str]] = {
    "output.identity": frozenset({"scenario_subpath"}),
    "output.directories": frozenset({"checkpoints", "tests", "forecast"}),
    "output.overlay": frozenset({"path", "column"}),
}


class OutputSpec(FrozenMappingSpec):
    """Strict typed output section with immutable Mapping compatibility."""

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | "OutputSpec",
        *,
        source: str = "<constructor>",
    ) -> "OutputSpec":
        if isinstance(value, cls):
            return value
        payload = strict_mapping(
            value,
            path="output",
            source=source,
            allowed=OUTPUT_FIELDS,
        )
        validate_nested_mappings(
            payload,
            source=source,
            schemas=_OUTPUT_NESTED_FIELDS,
        )
        return cls(freeze_json_value(payload, "output"))


__all__ = ["OutputSpec", "OUTPUT_FIELDS"]
