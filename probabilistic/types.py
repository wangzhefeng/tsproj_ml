# -*- coding: utf-8 -*-
"""概率模型输出的强类型数据契约。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from probabilistic.spec import (
    ProbabilisticSpec,
    validate_interval_quantiles,
    validate_quantile_grid,
)


def _as_finite_1d(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape={array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _quantile_column_token(level: float) -> str:
    percent = float(level) * 100.0
    token = format(percent, ".15g")
    return token.replace(".", "p")


@dataclass(frozen=True)
class QuantileGrid:
    """数值 quantile grid；CSV 列名只是可逆序列化表示。"""

    levels: tuple[float, ...]
    point_level: float = 0.5

    def __post_init__(self) -> None:
        levels = validate_quantile_grid(self.levels, self.point_level)
        column_names = [f"predict_q{_quantile_column_token(level)}" for level in levels]
        if len(set(column_names)) != len(column_names):
            raise ValueError(
                "quantile column codec collision; levels are too close to serialize safely"
            )
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "point_level", float(self.point_level))

    def index_of(self, level: float) -> int:
        value = float(level)
        for index, candidate in enumerate(self.levels):
            if math.isclose(candidate, value, rel_tol=0.0, abs_tol=1e-12):
                return index
        raise ValueError(f"quantile level={value:g} is not present in the grid")

    @property
    def point_index(self) -> int:
        return self.index_of(self.point_level)

    def column_name(self, level: float) -> str:
        canonical_level = self.levels[self.index_of(level)]
        return f"predict_q{_quantile_column_token(canonical_level)}"


@dataclass
class PredictionIntervalForecast:
    """独立 prediction interval；边界不冒充分位数。"""

    name: str
    lower: np.ndarray
    upper: np.ndarray
    target_coverage: float
    method: str
    base_quantiles: tuple[float, float]

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        self.method = str(self.method).strip().lower()
        if not self.name:
            raise ValueError("prediction interval name must not be empty")
        if not self.method:
            raise ValueError("prediction interval method must not be empty")
        self.lower = _as_finite_1d(self.lower, "interval lower")
        self.upper = _as_finite_1d(self.upper, "interval upper")
        if len(self.lower) != len(self.upper):
            raise ValueError(
                "prediction interval length mismatch: "
                f"lower={len(self.lower)}, upper={len(self.upper)}"
            )
        if np.any(self.lower > self.upper):
            raise ValueError("prediction interval requires lower <= upper at every point")
        coverage = float(self.target_coverage)
        if not math.isfinite(coverage) or not 0.0 < coverage < 1.0:
            raise ValueError("target_coverage must be finite and inside (0, 1)")
        if len(self.base_quantiles) != 2:
            raise ValueError("base_quantiles must contain exactly two levels")
        lower_q, upper_q = validate_interval_quantiles(
            self.base_quantiles[0],
            self.base_quantiles[1],
            self.base_quantiles,
        )
        self.target_coverage = coverage
        self.base_quantiles = (lower_q, upper_q)


@dataclass
class ForecastDistribution:
    """统一的边际分位数预测结果。"""

    point: np.ndarray
    quantile_grid: QuantileGrid
    quantile_values: np.ndarray
    intervals: dict[str, PredictionIntervalForecast]
    space: str
    quantile_stage: str
    forecast_times: pd.DatetimeIndex
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.space = str(self.space).lower()
        self.quantile_stage = str(self.quantile_stage).lower()
        if self.space not in {"model", "target"}:
            raise ValueError("space must be one of: model, target")
        if self.quantile_stage not in {"raw", "processed"}:
            raise ValueError("quantile_stage must be one of: raw, processed")

        self.point = _as_finite_1d(self.point, "point")
        values = np.asarray(self.quantile_values, dtype=float)
        expected_shape = (len(self.point), len(self.quantile_grid.levels))
        if values.shape != expected_shape:
            raise ValueError(
                f"quantile_values shape must be {expected_shape}; got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("quantile_values must contain only finite values")
        self.quantile_values = values.copy()
        point_quantile = self.quantile_values[:, self.quantile_grid.point_index]
        if not np.allclose(self.point, point_quantile, rtol=0.0, atol=1e-12):
            raise ValueError("point must equal the configured point quantile at every step")

        times = pd.DatetimeIndex(pd.to_datetime(self.forecast_times))
        if len(times) != len(self.point):
            raise ValueError(
                f"forecast_times length must equal n_steps={len(self.point)}; got {len(times)}"
            )
        if times.hasnans:
            raise ValueError("forecast_times must not contain NaT")
        if len(times) > 1 and np.any(np.diff(times.asi8) <= 0):
            raise ValueError("forecast_times must be strictly increasing")
        self.forecast_times = times.copy()

        normalized_intervals: dict[str, PredictionIntervalForecast] = {}
        for key, interval in dict(self.intervals).items():
            if not isinstance(interval, PredictionIntervalForecast):
                raise TypeError(f"interval {key!r} must be PredictionIntervalForecast")
            if str(key) != interval.name:
                raise ValueError(
                    f"interval mapping key={key!r} must equal interval.name={interval.name!r}"
                )
            if len(interval.lower) != len(self.point):
                raise ValueError(
                    f"interval {key!r} length must equal n_steps={len(self.point)}"
                )
            for level in interval.base_quantiles:
                self.quantile_grid.index_of(level)
            normalized_intervals[str(key)] = interval
        self.intervals = normalized_intervals
        self.metadata = dict(self.metadata)

    @property
    def n_steps(self) -> int:
        return len(self.point)


@dataclass(frozen=True)
class BlendQuantileModel:
    """一个 quantile 的 Direct/Recursive 显式子模型对。"""

    direct: Any
    recursive: Any

    def __post_init__(self) -> None:
        if self.direct is None or self.recursive is None:
            raise ValueError("blend quantile model requires direct and recursive models")


@dataclass
class ProbabilisticModelBundle:
    """替代 legacy nested dict 的版本化概率模型 bundle。"""

    schema_version: int
    spec: ProbabilisticSpec
    model_type: str
    pred_method: str
    models_by_quantile: dict[float, Any]
    recursive_propagation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.schema_version) != 1:
            raise ValueError(
                f"Unsupported ProbabilisticModelBundle schema_version={self.schema_version}"
            )
        if self.spec.mode != "quantile":
            raise ValueError("ProbabilisticModelBundle requires spec.mode=quantile")
        models = {float(level): model for level, model in self.models_by_quantile.items()}
        expected = tuple(self.spec.quantiles)
        actual = tuple(sorted(models))
        if actual != expected:
            raise ValueError(
                f"models_by_quantile keys must exactly equal spec.quantiles; "
                f"expected={expected}, actual={actual}"
            )
        if any(model is None for model in models.values()):
            raise ValueError("models_by_quantile must not contain None")
        propagation = str(self.recursive_propagation).lower()
        if propagation != self.spec.recursive_propagation:
            raise ValueError(
                "bundle recursive_propagation must equal spec.recursive_propagation"
            )
        self.schema_version = 1
        self.model_type = str(self.model_type).lower()
        self.pred_method = str(self.pred_method)
        self.models_by_quantile = {level: models[level] for level in expected}
        self.recursive_propagation = propagation
        self.metadata = dict(self.metadata)

    @property
    def quantile_grid(self) -> QuantileGrid:
        return QuantileGrid(self.spec.quantiles, point_level=self.spec.point_quantile)

    @property
    def point_quantile(self) -> float:
        return self.spec.point_quantile

    @property
    def is_blend(self) -> bool:
        return bool(self.models_by_quantile) and all(
            isinstance(model, BlendQuantileModel)
            for model in self.models_by_quantile.values()
        )


@dataclass
class ForecastModelBundle:
    """部署期唯一权威 bundle；模型、预处理和输入 schema 同步版本化。"""

    schema_version: int
    model: Any
    feature_scaler: Any
    target_transform: Any
    selected_features: tuple[str, ...]
    input_schema: dict[str, Any]
    probabilistic_spec: ProbabilisticSpec
    model_type: str
    pred_method: str

    def __post_init__(self) -> None:
        if int(self.schema_version) != 1:
            raise ValueError(
                f"Unsupported ForecastModelBundle schema_version={self.schema_version}"
            )
        if self.model is None:
            raise ValueError("ForecastModelBundle.model must not be None")
        self.schema_version = 1
        self.selected_features = tuple(str(value) for value in self.selected_features)
        self.input_schema = dict(self.input_schema)
        self.model_type = str(self.model_type).lower()
        self.pred_method = str(self.pred_method)

    def schema_payload(self) -> dict[str, Any]:
        """返回不含模型对象、可供人工审计的 JSON metadata。"""
        return {
            "schema_version": self.schema_version,
            "model_type": self.model_type,
            "pred_method": self.pred_method,
            "selected_features": list(self.selected_features),
            "input_schema": self.input_schema,
            "probabilistic": asdict(self.probabilistic_spec),
        }

    def write_schema_json(self, path: Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.schema_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def migrate_legacy_quantile_bundle(value: Any) -> Any:
    """把旧 ``{"predict_type":"quantile", "models":...}`` 转为 runtime bundle。"""
    if not (
        isinstance(value, dict)
        and value.get("predict_type") == "quantile"
        and isinstance(value.get("models"), dict)
    ):
        return value
    raw_models = value["models"]
    raw_levels = value.get("quantiles", list(raw_models))
    point_level = float(value.get("median_quantile", 0.5))
    grid = QuantileGrid(tuple(float(level) for level in raw_levels), point_level)
    models: dict[float, Any] = {}
    for level in grid.levels:
        model = raw_models.get(level, raw_models.get(str(level)))
        if isinstance(model, dict) and {"direct", "recursive"} <= set(model):
            model = BlendQuantileModel(
                direct=model["direct"],
                recursive=model["recursive"],
            )
        models[level] = model
    spec = ProbabilisticSpec(
        mode="quantile",
        quantiles=grid.levels,
        point_quantile=grid.point_level,
        recursive_propagation=str(value.get("recursive_propagation", "median_path")),
        crossing_method="none",
        crossing_report_raw=True,
        intervals=(),
        calibration=None,
        schema_version=1,
    )
    return ProbabilisticModelBundle(
        schema_version=1,
        spec=spec,
        model_type=str(value.get("model_type", "legacy")),
        pred_method=str(value.get("pred_method", "legacy")),
        models_by_quantile=models,
        recursive_propagation=spec.recursive_propagation,
        metadata={"migrated_from_legacy_dict": True},
    )
