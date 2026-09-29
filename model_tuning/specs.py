"""独立调参配方；模型候选始终重新解析为 canonical config。"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from typing import Any

import optuna
import pandas as pd

from forecasting_core.specs.config import ForecastConfigSpec, parse_model_config


def _positive_integer(value: Any, name: str, *, zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f"{name} must be a {'non-negative' if zero else 'positive'} integer")
    return value


def _choice_value(value: Any) -> None:
    if isinstance(value, list):
        for item in value:
            _choice_value(item)
    elif value is None or isinstance(value, (str, bool, int)):
        return
    elif isinstance(value, float) and math.isfinite(value):
        return
    else:
        raise ValueError("categorical choices require finite JSON scalars or lists")


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    path: str
    kind: str
    low: int | float = 0
    high: int | float = 0
    log: bool = False
    choices: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, path: str, value: Any) -> ParameterSpec:
        if not isinstance(path, str) or any(not part or part.strip() != part for part in path.split(".")):
            raise ValueError("parameter path must have nonempty exact components")
        parts = path.split(".")
        if not ((len(parts) == 3 and parts[:2] == ["estimator", "params"])
                or (len(parts) >= 2 and parts[0] == "features")):
            raise ValueError(f"forbidden tuning path: {path}")
        if not isinstance(value, Mapping):
            raise TypeError("parameter distribution must be a mapping")
        kind = value.get("type")
        allowed = {"type", "choices"} if kind == "categorical" else {"type", "low", "high", "log"}
        if set(value) - allowed:
            raise ValueError(f"unknown distribution fields for {path}")
        if kind == "categorical":
            choices = value.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError("categorical choices must be a nonempty list")
            for choice in choices:
                _choice_value(choice)
            encoded = tuple(json.dumps(choice, sort_keys=True, allow_nan=False) for choice in choices)
            if len(set(encoded)) != len(encoded):
                raise ValueError("categorical choices must be unique")
            return cls(path, kind, choices=encoded)
        if kind not in {"int", "float"}:
            raise ValueError(f"unknown distribution type: {kind!r}")
        low, high = value.get("low"), value.get("high")
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise ValueError(f"numeric bounds required for {path}")
        for bound in (low, high):
            if (isinstance(bound, bool) or not isinstance(bound, (int, float))
                    or not math.isfinite(bound) or (kind == "int" and not isinstance(bound, int))):
                raise ValueError(f"invalid {kind} bounds for {path}")
        if low > high:
            raise ValueError("distribution low must not exceed high")
        logarithmic = value.get("log", False)
        if not isinstance(logarithmic, bool) or (logarithmic and low <= 0):
            raise ValueError("log must be bool and log distributions require low > 0")
        return cls(path, kind, low, high, logarithmic)

    def sample(self, trial: optuna.trial.BaseTrial) -> Any:
        if self.kind == "categorical":
            # Optuna 类别存整数索引，列表型特征组合保存在独立的实际参数记录中。
            index = trial.suggest_categorical(self.path, list(range(len(self.choices))))
            return json.loads(self.choices[index])
        if self.kind == "int":
            return trial.suggest_int(self.path, int(self.low), int(self.high), log=self.log)
        return trial.suggest_float(self.path, float(self.low), float(self.high), log=self.log)


@dataclass(frozen=True, slots=True)
class TuningSpec:
    trials: int
    seed: int
    metric: str
    holdout_origin: str
    holdout_fold_count: int
    parameters: tuple[ParameterSpec, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> TuningSpec:
        fields = {"trials", "seed", "metric", "holdout_origin", "holdout_fold_count", "parameters"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError(f"search recipe requires exactly {sorted(fields)}")
        trials = _positive_integer(value["trials"], "trials")
        seed = _positive_integer(value["seed"], "seed", zero=True)
        folds = _positive_integer(value["holdout_fold_count"], "holdout_fold_count")
        metric = value["metric"]
        if metric not in {"MAE", "RMSE", "MAPE"}:
            raise ValueError("metric must be MAE, RMSE or MAPE")
        origin = value["holdout_origin"]
        if not isinstance(origin, str) or not origin.strip() or pd.isna(pd.Timestamp(origin)):
            raise ValueError("holdout_origin must be an explicit timestamp string")
        parameters = value["parameters"]
        if not isinstance(parameters, Mapping) or not parameters:
            raise ValueError("parameters must be a nonempty mapping")
        parsed = tuple(ParameterSpec.from_mapping(path, distribution) for path, distribution in parameters.items())
        paths = [parameter.path.split(".") for parameter in parsed]
        if any(a == b[:len(a)] for i, a in enumerate(paths) for j, b in enumerate(paths) if i != j):
            raise ValueError("parameter paths must not overlap")
        return cls(trials, seed, metric, origin, folds, parsed)

    def sample_config(self, base: ForecastConfigSpec, trial: optuna.trial.BaseTrial) -> tuple[ForecastConfigSpec, dict]:
        payload = base.canonical_payload()
        selected = {}
        for parameter in self.parameters:
            parts = parameter.path.split(".")
            parent: dict[str, Any] = payload
            for part in parts[:-1]:
                child = parent.get(part)
                if not isinstance(child, dict):
                    raise ValueError(f"tuning requires an existing feature/parameter path: {parameter.path}")
                parent = child
            if parts[0] == "features" and parts[-1] not in parent:
                raise ValueError(f"tuning requires an existing feature path: {parameter.path}")
            sampled = parameter.sample(trial)
            parent[parts[-1]] = sampled
            selected[parameter.path] = sampled
        return parse_model_config(payload, source="<tuning candidate>"), selected
