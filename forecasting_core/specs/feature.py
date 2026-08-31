"""Canonical feature-generation specifications."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


def _required_name(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be blank")
    if value != stripped:
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    return value


def _normalize_lag_mapping(value: Any, field_name: str) -> Mapping[str, tuple[int, ...]]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")

    normalized: dict[str, tuple[int, ...]] = {}
    for raw_name, raw_lags in value.items():
        name = _required_name(raw_name, f"{field_name} keys")
        if isinstance(raw_lags, (str, bytes)) or not isinstance(raw_lags, Sequence):
            raise TypeError(f"{field_name}.{name} must be a sequence of integers")
        lags = []
        for lag in raw_lags:
            if isinstance(lag, bool) or not isinstance(lag, int):
                raise TypeError(f"{field_name}.{name} entries must be integers")
            if lag <= 0:
                raise ValueError(f"{field_name}.{name} entries must be positive")
            lags.append(lag)
        normalized[name] = tuple(sorted(set(lags)))
    return MappingProxyType(dict(sorted(normalized.items())))


def _normalize_datetime_features(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("datetime_features must be a sequence of strings")
    features = tuple(
        _required_name(feature, "datetime_features entries") for feature in value
    )
    if len(features) != len(set(features)):
        raise ValueError("datetime_features must not contain duplicates")
    return features


def _freeze_json_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite floats")
        return value
    if isinstance(value, list):
        return tuple(
            _freeze_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = _required_name(key, f"{path} keys")
            if normalized_key.startswith("enable_"):
                raise ValueError(f"{path}.{normalized_key} is forbidden")
            normalized[normalized_key] = _freeze_json_value(
                item,
                f"{path}.{normalized_key}",
            )
        return MappingProxyType(dict(sorted(normalized.items())))
    raise TypeError(f"{path} must contain only JSON-like values")


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True, init=False)
class FeatureSpec:
    target_lags: Mapping[str, tuple[int, ...]]
    observed_past_lags: Mapping[str, tuple[int, ...]]
    datetime_features: tuple[str, ...]
    transformations: Mapping[str, Any]
    selection: Mapping[str, Any] | None

    def __init__(
        self,
        target_lags: Mapping[str, Sequence[int]],
        observed_past_lags: Mapping[str, Sequence[int]],
        datetime_features: Sequence[str],
        transformations: Mapping[str, Any],
        selection: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(transformations, Mapping):
            raise TypeError("transformations must be a mapping")
        if selection is not None and not isinstance(selection, Mapping):
            raise TypeError("selection must be a mapping when present")

        object.__setattr__(
            self,
            "target_lags",
            _normalize_lag_mapping(target_lags, "target_lags"),
        )
        object.__setattr__(
            self,
            "observed_past_lags",
            _normalize_lag_mapping(observed_past_lags, "observed_past_lags"),
        )
        object.__setattr__(
            self,
            "datetime_features",
            _normalize_datetime_features(datetime_features),
        )
        object.__setattr__(
            self,
            "transformations",
            _freeze_json_value(transformations, "transformations"),
        )
        object.__setattr__(
            self,
            "selection",
            _freeze_json_value(selection, "features.selection")
            if selection is not None
            else None,
        )

    def canonical_payload(self) -> dict[str, object]:
        payload = {
            "target_lags": {
                name: list(lags) for name, lags in self.target_lags.items()
            },
            "observed_past_lags": {
                name: list(lags) for name, lags in self.observed_past_lags.items()
            },
            "datetime_features": list(self.datetime_features),
            "transformations": _thaw_json_value(self.transformations),
        }
        # 未配置 selection 时不进 payload，保持存量配置 fingerprint 不变
        if self.selection is not None:
            payload["selection"] = _thaw_json_value(self.selection)
        return payload


__all__ = ["FeatureSpec"]
