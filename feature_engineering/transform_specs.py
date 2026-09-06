# -*- coding: utf-8 -*-
"""Strict feature/target transformation configuration normalization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from decomposition.configuration.spec import normalize_decomposition_config

_FEATURE_SCALING_METHODS = frozenset({"none", "minmax", "standard", "robust"})
_CALENDAR_METHODS = frozenset({"none", "per_calendar_day"})
_TARGET_SCALING_METHODS = _FEATURE_SCALING_METHODS


def normalize_feature_scaling(value: Any) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError("transformations.feature_scaling must be a mapping")
    unknown = set(value) - {"method", "grouped", "encode_categorical"}
    if unknown:
        raise ValueError(
            f"unknown transformations.feature_scaling keys: {sorted(unknown)}"
        )
    method = str(value.get("method", "none") or "none").lower()
    if method not in _FEATURE_SCALING_METHODS:
        raise ValueError(
            f"unsupported feature scaling method {method!r}; "
            f"expected one of {sorted(_FEATURE_SCALING_METHODS)}"
        )
    grouped = value.get("grouped", False)
    encode_categorical = value.get("encode_categorical", False)
    if not isinstance(grouped, bool):
        raise TypeError("transformations.feature_scaling.grouped must be bool")
    if not isinstance(encode_categorical, bool):
        raise TypeError(
            "transformations.feature_scaling.encode_categorical must be bool"
        )
    return {
        "method": method,
        "grouped": grouped,
        "encode_categorical": encode_categorical,
    }


def normalize_target_transformations(value: Any) -> dict[str, dict[str, Any]]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError("transformations.target must be a mapping")
    unknown = set(value) - {"calendar_normalization", "decomposition", "scaling"}
    if unknown:
        raise ValueError(f"unknown transformations.target keys: {sorted(unknown)}")

    calendar = value.get("calendar_normalization", {})
    if not isinstance(calendar, Mapping):
        raise TypeError("transformations.target.calendar_normalization must be a mapping")
    calendar_unknown = set(calendar) - {"method"}
    if calendar_unknown:
        raise ValueError(
            "unknown transformations.target.calendar_normalization keys: "
            f"{sorted(calendar_unknown)}"
        )
    calendar_method = str(calendar.get("method", "none") or "none").lower()
    if calendar_method not in _CALENDAR_METHODS:
        raise ValueError(
            f"unsupported calendar normalization method {calendar_method!r}; "
            f"expected one of {sorted(_CALENDAR_METHODS)}"
        )

    decomposition_payload = normalize_decomposition_config(value.get("decomposition", {}))

    scaling = value.get("scaling", {})
    if not isinstance(scaling, Mapping):
        raise TypeError("transformations.target.scaling must be a mapping")
    scaling_unknown = set(scaling) - {"method", "inverse"}
    if scaling_unknown:
        raise ValueError(
            f"unknown transformations.target.scaling keys: {sorted(scaling_unknown)}"
        )
    scaling_method = str(scaling.get("method", "none") or "none").lower()
    if scaling_method not in _TARGET_SCALING_METHODS:
        raise ValueError(
            f"unsupported target scaling method {scaling_method!r}; "
            f"expected one of {sorted(_TARGET_SCALING_METHODS)}"
        )
    inverse = scaling.get("inverse", False)
    if not isinstance(inverse, bool):
        raise TypeError("transformations.target.scaling.inverse must be bool")
    return {
        "calendar_normalization": {"method": calendar_method},
        "decomposition": decomposition_payload,
        "scaling": {"method": scaling_method, "inverse": inverse},
    }


__all__ = ["normalize_feature_scaling", "normalize_target_transformations"]
