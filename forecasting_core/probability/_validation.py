"""概率子包内部共享的严格标量与 mapping 校验。"""
from __future__ import annotations

from numbers import Real
from typing import Any, Mapping
import math


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _strict_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _is_close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _validate_unknown_keys(mapping: Mapping[str, Any], allowed: set[str] | frozenset[str], path: str) -> None:
    if any(not isinstance(key, str) for key in mapping):
        raise TypeError(f"{path} keys must be strings")
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown {path} key(s): {unknown}")


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value
