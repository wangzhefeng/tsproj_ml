"""Shared immutable mapping support for canonical runtime specs."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeVar


T = TypeVar("T", bound="FrozenMappingSpec")


def freeze_json_value(value: Any, path: str) -> Any:
    """Recursively freeze one JSON-like value and reject unsupported values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite floats")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            normalized[key] = freeze_json_value(item, f"{path}.{key}")
        return MappingProxyType(dict(sorted(normalized.items())))
    raise TypeError(f"{path} must contain only JSON-like values")


def thaw_json_value(value: Any) -> Any:
    """Return a mutable JSON-like copy of a frozen value."""
    if isinstance(value, tuple):
        return [thaw_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: thaw_json_value(item) for key, item in value.items()}
    return value


def strict_mapping(
    value: Any,
    *,
    path: str,
    source: str,
    allowed: frozenset[str],
    required: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping in {source}")
    keys = set(value)
    if any(not isinstance(key, str) for key in keys):
        raise TypeError(f"{path} keys must be strings in {source}")
    unknown = sorted(keys - allowed)
    missing = sorted(required - keys)
    if unknown:
        raise ValueError(f"Unknown fields in {path} from {source}: {unknown}")
    if missing:
        raise ValueError(f"Missing required fields in {path} from {source}: {missing}")
    return value


def validate_nested_mappings(
    payload: Mapping[str, Any],
    *,
    source: str,
    schemas: Mapping[str, frozenset[str]],
) -> None:
    """Validate every present dotted-path mapping against its allowed keys."""
    for dotted_path, allowed in schemas.items():
        value: Any = payload
        for part in dotted_path.split(".")[1:]:
            if not isinstance(value, Mapping) or part not in value:
                value = None
                break
            value = value[part]
        if value is None:
            continue
        strict_mapping(
            value,
            path=dotted_path,
            source=source,
            allowed=allowed,
        )


@dataclass(frozen=True, slots=True)
class FrozenMappingSpec(Mapping[str, Any]):
    """Typed immutable boundary that remains compatible with Mapping consumers."""

    _values: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls: type[T],
        value: Mapping[str, Any] | T,
        *,
        source: str = "<constructor>",
    ) -> T:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"{cls.__name__} requires a mapping")
        return cls(freeze_json_value(value, cls.__name__))

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def canonical_payload(self) -> dict[str, Any]:
        return thaw_json_value(self._values)


__all__ = [
    "FrozenMappingSpec",
    "freeze_json_value",
    "strict_mapping",
    "thaw_json_value",
    "validate_nested_mappings",
]
