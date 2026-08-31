"""Typed YAML-level probabilistic configuration contract.

This type preserves the canonical YAML payload, including historical conformal
intent that the current runtime does not execute.  The deployment artifact uses
``forecasting_core.probabilistic_spec.ProbabilisticSpec`` instead.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from forecasting_core.specs._mapping import (
    FrozenMappingSpec,
    freeze_json_value,
    strict_mapping,
    validate_nested_mappings,
)


PROBABILISTIC_FIELDS = frozenset(
    {
        "mode",
        "schema_version",
        "quantiles",
        "point_quantile",
        "recursive_propagation",
        "crossing",
        "crossing_method",
        "intervals",
        "calibration",
        "conformal",
    }
)

_PROBABILISTIC_NESTED_FIELDS: dict[str, frozenset[str]] = {
    "probabilistic.crossing": frozenset({"method", "report_raw"}),
    "probabilistic.conformal": frozenset(
        {
            "method",
            "interval",
            "target_coverage",
            "alpha",
            "calibration_windows",
            "min_windows",
            "min_scores",
            "label_availability_delay_steps",
            "allow_interval_shrink",
            "grouping",
            "coverage",
        }
    ),
    "probabilistic.calibration": frozenset(
        {
            "method",
            "interval",
            "target_coverage",
            "calibration_windows",
            "min_windows",
            "min_scores",
            "label_availability_delay_steps",
            "allow_interval_shrink",
            "grouping",
        }
    ),
}
_INTERVAL_FIELDS = frozenset({"name", "lower_quantile", "upper_quantile"})


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
        validate_nested_mappings(
            payload,
            source=source,
            schemas=_PROBABILISTIC_NESTED_FIELDS,
        )
        if "crossing" in payload and "crossing_method" in payload:
            raise ValueError(
                "probabilistic.crossing and probabilistic.crossing_method are mutually exclusive"
            )
        _validate_intervals(payload.get("intervals"), source=source)
        _validate_probability_semantics(payload)
        return cls(freeze_json_value(payload, "probabilistic"))


def _validate_intervals(value: Any, *, source: str) -> None:
    if value is None:
        return
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"probabilistic.intervals must be a sequence in {source}")
    for index, item in enumerate(value):
        strict_mapping(
            item,
            path=f"probabilistic.intervals[{index}]",
            source=source,
            allowed=_INTERVAL_FIELDS,
            required=_INTERVAL_FIELDS,
        )


def _validate_probability_semantics(payload: Mapping[str, Any]) -> None:
    mode = str(payload.get("mode", "point")).lower()
    if mode not in {"point", "quantile"}:
        raise ValueError("probabilistic.mode must be point or quantile")
    if mode == "point":
        return
    raw_levels = payload.get("quantiles")
    if isinstance(raw_levels, (str, bytes)) or not isinstance(raw_levels, Sequence):
        raise TypeError("probabilistic.quantiles must be a sequence in quantile mode")
    levels = tuple(float(level) for level in raw_levels)
    if not levels:
        raise ValueError("probabilistic.quantiles must not be empty in quantile mode")
    if any(not math.isfinite(level) or not 0.0 < level < 1.0 for level in levels):
        raise ValueError("probabilistic.quantiles must be finite and inside (0, 1)")
    if tuple(sorted(set(levels))) != levels:
        raise ValueError("probabilistic.quantiles must be unique and strictly increasing")
    point = float(payload.get("point_quantile", 0.5))
    if point not in levels:
        raise ValueError("probabilistic.point_quantile must be present in quantiles")


__all__ = ["ProbabilisticConfigSpec", "PROBABILISTIC_FIELDS"]
