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

from forecasting_core.tensors import (
    MarginalQuantileForecastTensor,
    PointForecastTensor,
)
from forecasting_core.probabilistic_spec import (
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
class MarginalForecastDistribution:
    """Canonical per-target marginal quantiles; no joint dependence model."""

    point: PointForecastTensor
    quantiles: MarginalQuantileForecastTensor
    dependence_model: None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.point, PointForecastTensor):
            raise TypeError("point must be a PointForecastTensor")
        if not isinstance(self.quantiles, MarginalQuantileForecastTensor):
            raise TypeError(
                "quantiles must be a MarginalQuantileForecastTensor"
            )
        if self.dependence_model is not None:
            raise ValueError(
                "joint dependence models are unsupported; dependence_model must be None"
            )
        quantile_point = self.quantiles.point()
        if (
            self.point.series_ids != quantile_point.series_ids
            or self.point.targets != quantile_point.targets
            or not self.point.forecast_times.equals(quantile_point.forecast_times)
        ):
            raise ValueError("point and quantiles must have identical axes")
        if not np.allclose(
            self.point.values,
            quantile_point.values,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("point must equal the configured point quantile")
        self.metadata = {
            **dict(self.metadata),
            "distribution_kind": "marginal_quantile",
            "dependence_model": None,
        }

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.quantiles.shape


def generate_joint_samples(
    quantiles: MarginalQuantileForecastTensor,
    *,
    n_samples: int,
):
    if not isinstance(quantiles, MarginalQuantileForecastTensor):
        raise TypeError("quantiles must be a MarginalQuantileForecastTensor")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    raise NotImplementedError(
        "joint sample generation is not implemented; marginal quantiles have "
        "dependence_model=None"
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
    pred_method: str | None
    canonical_problem: dict[str, Any] | None = None
    strategy_spec: dict[str, Any] | None = None
    ensemble_spec: dict[str, Any] | None = None
    estimator_spec: dict[str, Any] | None = None
    dimensions: tuple[int, int, int] | None = None
    series_ids: tuple[Any, ...] = ()
    target_order: tuple[str, ...] = ()
    feature_lineage: tuple[dict[str, Any], ...] = ()
    source_lineage: tuple[dict[str, Any], ...] = ()
    training_scope: str | None = None
    result_schema_version: int | None = None
    config_fingerprint: str | None = None
    # CQR 校准状态（2026-09-01 激活）：final fit 时由回测折的 as-of 校准池
    # 计算，部署期据此产出 predict_pi 区间；未配置 calibration 时为 None。
    calibration_state: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        version = int(self.schema_version)
        if version not in {1, 2}:
            raise ValueError(
                f"Unsupported ForecastModelBundle schema_version={self.schema_version}"
            )
        if self.model is None:
            raise ValueError("ForecastModelBundle.model must not be None")
        self.schema_version = version
        self.selected_features = tuple(str(value) for value in self.selected_features)
        self.input_schema = dict(self.input_schema)
        self.model_type = str(self.model_type).lower()
        if version == 1:
            if self.pred_method is None:
                raise ValueError("schema v1 ForecastModelBundle requires pred_method")
            self.pred_method = str(self.pred_method)
            return

        if self.pred_method is not None:
            raise ValueError("canonical ForecastModelBundle forbids legacy pred_method")
        if not isinstance(self.canonical_problem, dict):
            raise TypeError("canonical ForecastModelBundle requires canonical_problem")
        if (self.strategy_spec is None) == (self.ensemble_spec is None):
            raise ValueError(
                "canonical ForecastModelBundle requires exactly one strategy_spec or ensemble_spec"
            )
        if self.ensemble_spec is not None:
            # ensemble bundle: fusion state lives in ensemble_spec and member
            # bundles; top-level strategy/estimator specs must be absent (v4 §8.2)
            if self.estimator_spec is not None or self.strategy_spec is not None:
                raise ValueError(
                    "canonical ensemble ForecastModelBundle forbids top-level "
                    "strategy_spec/estimator_spec; they live in member bundles"
                )
        elif not isinstance(self.estimator_spec, dict):
            raise TypeError("canonical ForecastModelBundle requires estimator_spec")
        if (
            not isinstance(self.dimensions, tuple)
            or len(self.dimensions) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.dimensions
            )
        ):
            raise ValueError("canonical ForecastModelBundle dimensions must be positive (N,H,K)")
        if len(self.target_order) != self.dimensions[2]:
            raise ValueError("canonical ForecastModelBundle target_order must match K")
        self.series_ids = tuple(self.series_ids)
        if not self.series_ids:
            panel = self.input_schema.get("panel", {})
            known = panel.get("known_series_ids", ()) if isinstance(panel, dict) else ()
            self.series_ids = tuple(
                tuple(value) if isinstance(value, list) else value for value in known
            )
        if not self.series_ids and self.dimensions[0] == 1:
            self.series_ids = ("__local__",)
        if self.series_ids and len(self.series_ids) != self.dimensions[0]:
            raise ValueError("canonical ForecastModelBundle series_ids must match N")
        if self.training_scope not in {"local", "global"}:
            raise ValueError("canonical ForecastModelBundle training_scope must be local or global")
        if self.result_schema_version != 2:
            raise ValueError("canonical ForecastModelBundle result_schema_version must be 2")
        if (
            not isinstance(self.config_fingerprint, str)
            or len(self.config_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.config_fingerprint
            )
        ):
            raise ValueError(
                "canonical ForecastModelBundle requires a SHA-256 config_fingerprint"
            )
        self.canonical_problem = dict(self.canonical_problem)
        self.strategy_spec = (
            dict(self.strategy_spec) if self.strategy_spec is not None else None
        )
        self.ensemble_spec = (
            dict(self.ensemble_spec) if self.ensemble_spec is not None else None
        )
        self.estimator_spec = (
            dict(self.estimator_spec) if self.estimator_spec is not None else None
        )
        self.target_order = tuple(str(target) for target in self.target_order)
        self.feature_lineage = tuple(dict(item) for item in self.feature_lineage)
        self.source_lineage = tuple(dict(item) for item in self.source_lineage)
        if self.calibration_state is not None:
            if not isinstance(self.calibration_state, dict):
                raise TypeError("calibration_state must be a dict or None")
            if self.calibration_state.get("status") == "applied" and not isinstance(
                self.calibration_state.get("correction"), (int, float)
            ):
                raise ValueError(
                    "applied calibration_state requires a numeric correction"
                )
            self.calibration_state = dict(self.calibration_state)

    def schema_payload(self) -> dict[str, Any]:
        """返回不含模型对象、可供人工审计的 JSON metadata。"""
        payload = {
            "schema_version": self.schema_version,
            "model_type": self.model_type,
            "selected_features": list(self.selected_features),
            "input_schema": self.input_schema,
            "probabilistic": asdict(self.probabilistic_spec),
        }
        if self.schema_version == 1:
            payload["pred_method"] = self.pred_method
            return payload
        payload.update(
            {
                "canonical_problem": self.canonical_problem,
                "strategy": self.strategy_spec,
                "ensemble": self.ensemble_spec,
                "estimator": self.estimator_spec,
                "dimensions": list(self.dimensions or ()),
                "series_ids": [
                    list(value) if isinstance(value, tuple) else value
                    for value in self.series_ids
                ],
                "target_order": list(self.target_order),
                "feature_lineage": list(self.feature_lineage),
                "source_lineage": list(self.source_lineage),
                "training_scope": self.training_scope,
                "result_schema_version": self.result_schema_version,
                "config_fingerprint": self.config_fingerprint,
                "calibration_state": self.calibration_state,
            }
        )
        return payload

    def write_schema_json(self, path: Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.schema_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
