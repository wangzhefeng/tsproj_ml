"""可读执行证据；永不进入语义配置身份。"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from importlib.metadata import PackageNotFoundError, version
import platform
import json
import math
import numpy as np
from typing import Any

from model_building.wrappers.base import BaseModel
from utils.runtime_env import RUNTIME_DEPENDENCY_PACKAGES

# 证据遍历的 artifact 宿主包前缀：实例字段递归只深入这些包的对象，
# 新增生产 artifact 宿主包时必须登记此处，否则该包节点被静默跳过（不 RAISE）。
_ARTIFACT_MODULE_PREFIXES = ("model_training.", "model_building.adapters.", "probabilistic.")


def dependency_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in RUNTIME_DEPENDENCY_PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not_installed"
    return result


def json_evidence(payload: Any) -> Any:
    """快照原生标量参数；不支持的对象显式标记而非静默丢弃。"""
    def encode(value: Any) -> Any:
        if isinstance(value, np.generic):
            return encode(value.item())
        if isinstance(value, np.ndarray):
            return encode(value.tolist())
        if isinstance(value, Mapping):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return {"nonfinite_float": str(value)}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return {"unserialized_type": f"{type(value).__module__}.{type(value).__qualname__}"}
    return json.loads(json.dumps(encode(payload), allow_nan=False))


def collect_model_evidence(artifact: Any) -> list[dict[str, Any]]:
    """只读已拟合 wrapper 状态，不执行拟合/预测；共享 booster 去重。"""
    visited: set[int] = set()
    records = []

    def walk(value: Any, path: str) -> None:
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, BaseModel):
            record = {"path": path, "wrapper": type(value).__name__, "fitted": value.is_fitted, "wrapper_params": value.get_params()}
            native = value.model
            if value.is_fitted and native is not None:
                if hasattr(native, "get_all_params"):
                    record["native_params"] = native.get_all_params()
                elif hasattr(native, "booster_"):
                    record["native_params"] = dict(native.booster_.params)
                elif hasattr(native, "get_params"):
                    record["native_params"] = native.get_params(deep=False)
                if hasattr(value, "parameter_validation"):
                    record["parameter_validation"] = getattr(value, "parameter_validation")
            records.append(record)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                walk(item, f"{path}/{key}")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                walk(item, f"{path}/{index}")
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                walk(getattr(value, field.name), f"{path}/{field.name}")
        elif type(value).__module__.startswith(_ARTIFACT_MODULE_PREFIXES):
            # slots 适配器没有 __dict__；只读已存储字段，不调用任意 property。
            attributes = dict(vars(value)) if hasattr(value, "__dict__") else {}
            for cls in type(value).__mro__:
                slots = cls.__dict__.get("__slots__", ())
                for key in ((slots,) if isinstance(slots, str) else slots):
                    if key not in {"__dict__", "__weakref__"} and hasattr(value, key):
                        attributes[key] = getattr(value, key)
            for key, item in attributes.items():
                walk(item, f"{path}/{key}")

    walk(artifact, "artifact")
    return records


__all__ = ["collect_model_evidence", "dependency_versions", "json_evidence"]
