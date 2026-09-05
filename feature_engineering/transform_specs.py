# -*- coding: utf-8 -*-
"""Strict feature/target transformation configuration normalization."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from decomposition import resolve_decomposition_spec

_FEATURE_SCALING_METHODS = frozenset({"none", "minmax", "standard", "robust"})
_CALENDAR_METHODS = frozenset({"none", "per_calendar_day"})
_DECOMPOSITION_METHODS = frozenset(
    {"none", "linear", "quadratic", "damped", "stl", "mstl"}
)
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

    decomposition = value.get("decomposition", {})
    if not isinstance(decomposition, Mapping):
        raise TypeError("transformations.target.decomposition must be a mapping")
    decomposition_payload = dict(decomposition)
    fit_history_steps = decomposition_payload.pop("fit_history_steps", None)
    if fit_history_steps is not None and (
        isinstance(fit_history_steps, bool) or not isinstance(fit_history_steps, int) or fit_history_steps < 1
    ):
        raise ValueError("decomposition.fit_history_steps must be a positive integer")
    decomposition_method = str(
        decomposition_payload.get("method", "none") or "none"
    ).lower()
    if decomposition_method not in _DECOMPOSITION_METHODS:
        raise ValueError(
            f"unsupported decomposition method {decomposition_method!r}; "
            f"expected one of {sorted(_DECOMPOSITION_METHODS)}"
        )
    if decomposition_method == "quadratic":
        configured_degree = int(decomposition_payload.get("trend_degree", 2))
        if configured_degree != 2:
            raise ValueError("quadratic decomposition requires trend_degree=2")
        decomposition_payload["method"] = "linear"
        decomposition_payload["trend_degree"] = 2
    elif decomposition_method == "damped":
        configured_forecast = str(
            decomposition_payload.get("trend_forecast", "damped") or "damped"
        ).lower()
        if configured_forecast != "damped":
            raise ValueError("damped decomposition requires trend_forecast='damped'")
        decomposition_payload["method"] = "linear"
        decomposition_payload["trend_forecast"] = "damped"
    else:
        decomposition_payload["method"] = decomposition_method
    resolve_decomposition_spec(
        SimpleNamespace(decomposition=dict(decomposition_payload))
    )
    if fit_history_steps is not None:
        if decomposition_method == "none":
            raise ValueError("decomposition.fit_history_steps requires an enabled decomposition")
        decomposition_payload["fit_history_steps"] = fit_history_steps

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
