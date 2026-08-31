"""Estimator selection specification independent of forecasting strategy."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum
from types import MappingProxyType
from typing import Any


def _required_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool")
    return value


@dataclass(frozen=True, slots=True, init=False)
class EstimatorCapabilities:
    """Estimator 能力声明合同（spec 层数据类型，行为探测实现见
    `model_training/estimators/capabilities.py`；2026-08-30 流水线阶段重排迁入，
    消除 specs → estimators 的反向依赖）。"""

    scalar_target: bool
    scalar_quantile: bool
    native_multi_target_point: bool
    native_multi_target_quantile: bool
    sample_weight: bool
    categorical: bool
    nan_support: bool

    def __init__(
        self,
        scalar_target: bool,
        scalar_quantile: bool,
        native_multi_target_point: bool,
        native_multi_target_quantile: bool,
        sample_weight: bool,
        categorical: bool,
        nan_support: bool,
    ) -> None:
        values = {
            "scalar_target": scalar_target,
            "scalar_quantile": scalar_quantile,
            "native_multi_target_point": native_multi_target_point,
            "native_multi_target_quantile": native_multi_target_quantile,
            "sample_weight": sample_weight,
            "categorical": categorical,
            "nan_support": nan_support,
        }
        for field_name, value in values.items():
            object.__setattr__(self, field_name, _required_bool(value, field_name))

    def canonical_payload(self) -> dict[str, bool]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


class TargetAdapter(str, Enum):
    INDEPENDENT = "independent"
    REGRESSOR_CHAIN = "regressor_chain"
    NATIVE = "native"


def _required_model_type(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("model_type must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError("model_type must not be blank")
    if value != stripped:
        raise ValueError("model_type must not contain surrounding whitespace")
    return value


def _normalize_target_adapter(value: Any) -> TargetAdapter:
    if isinstance(value, TargetAdapter):
        return value
    if not isinstance(value, str):
        raise TypeError("target_adapter must be a TargetAdapter or string")
    try:
        return TargetAdapter(value)
    except ValueError as exc:
        raise ValueError(f"unknown target_adapter: {value!r}") from exc


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
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            normalized[key] = _freeze_json_value(item, f"{path}.{key}")
        return MappingProxyType(dict(sorted(normalized.items())))
    raise TypeError(f"{path} must contain only JSON-like values")


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True, init=False)
class EstimatorSpec:
    model_type: str
    target_adapter: TargetAdapter
    params: Mapping[str, Any]

    def __init__(
        self,
        model_type: str,
        target_adapter: TargetAdapter | str,
        params: Mapping[str, Any] = {},
    ) -> None:
        if not isinstance(params, Mapping):
            raise TypeError("params must be a mapping")
        frozen_params = _freeze_json_value(params, "params")

        object.__setattr__(self, "model_type", _required_model_type(model_type))
        object.__setattr__(self, "target_adapter", _normalize_target_adapter(target_adapter))
        object.__setattr__(self, "params", frozen_params)

    def validate_capabilities(
        self,
        capabilities: EstimatorCapabilities,
        probabilistic_mode: str = "point",
    ) -> "EstimatorSpec":
        if not isinstance(capabilities, EstimatorCapabilities):
            raise TypeError("capabilities must be EstimatorCapabilities")
        if probabilistic_mode not in {"point", "quantile"}:
            raise ValueError(f"unknown probabilistic_mode: {probabilistic_mode!r}")

        if self.target_adapter in {
            TargetAdapter.INDEPENDENT,
            TargetAdapter.REGRESSOR_CHAIN,
        }:
            required_capability = (
                capabilities.scalar_target
                if probabilistic_mode == "point"
                else capabilities.scalar_quantile
            )
            if not required_capability:
                raise ValueError(
                    f"{self.target_adapter.value} requires "
                    f"scalar_{'target' if probabilistic_mode == 'point' else 'quantile'} "
                    "capability"
                )
            return self

        required_capability = (
            capabilities.native_multi_target_point
            if probabilistic_mode == "point"
            else capabilities.native_multi_target_quantile
        )
        if not required_capability:
            raise ValueError(
                "native target adapter lacks "
                f"native_multi_target_{probabilistic_mode} capability"
            )
        return self

    def canonical_payload(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "target_adapter": self.target_adapter.value,
            "params": _thaw_json_value(self.params),
        }
