"""Readable execution evidence; never part of semantic configuration identity."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from importlib.metadata import PackageNotFoundError, version
import platform
import json
import math
import numpy as np
from typing import Any

from models.wrappers.base import BaseModel


def dependency_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "catboost", "statsmodels", "chinese-calendar"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not_installed"
    return result


def json_evidence(payload: Any) -> Any:
    """Snapshot native scalar parameters; unsupported objects remain explicitly marked."""
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
    """Read fitted wrapper state without fitting or predicting; deduplicate shared boosters."""
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
        elif type(value).__module__.startswith(("model_training.", "probabilistic.")) and hasattr(value, "__dict__"):
            for key, item in vars(value).items():
                walk(item, f"{path}/{key}")

    walk(artifact, "artifact")
    return records
