"""Typed YAML-level probabilistic configuration contract.

The deployment artifact uses
``forecasting_core.probabilistic_spec.ProbabilisticSpec`` instead.
Legacy flat keys (``crossing_method``, ``conformal``) were swept from all
active YAMLs on 2026-09-01 and are rejected here; crossing behaviour is
declared via ``crossing:`` and CQR via ``calibration:``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from forecasting_core.specs._mapping import (
    FrozenMappingSpec,
    freeze_json_value,
    strict_mapping,
    validate_nested_mappings,
)
from forecasting_core.probabilistic_spec import validate_quantile_grid


PROBABILISTIC_FIELDS = frozenset(
    {
        "mode",
        "quantiles",
        "point_quantile",
        "crossing",
        "intervals",
        "calibration",
    }
)

_CROSSING_METHODS = frozenset(
    {"none", "rearrangement", "median_preserving_isotonic"}
)

_PROBABILISTIC_NESTED_FIELDS: dict[str, frozenset[str]] = {
    "probabilistic.crossing": frozenset({"method", "report_raw"}),
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
        _validate_crossing(payload.get("crossing"), source=source)
        _validate_intervals(payload.get("intervals"), source=source)
        _validate_probability_semantics(payload)
        return cls(freeze_json_value(payload, "probabilistic"))


def _validate_crossing(value: Any, *, source: str) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise TypeError(f"probabilistic.crossing must be a mapping in {source}")
    method = str(value.get("method", "median_preserving_isotonic")).lower()
    if method not in _CROSSING_METHODS:
        raise ValueError(
            f"probabilistic.crossing.method must be one of "
            f"{sorted(_CROSSING_METHODS)} in {source}; got {method!r}"
        )


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
    # 网格规则统一走合同层唯一实现（2026-09-01 去重）
    validate_quantile_grid(
        raw_levels,
        point_quantile=float(payload.get("point_quantile", 0.5)),
    )


__all__ = ["ProbabilisticConfigSpec", "PROBABILISTIC_FIELDS"]
