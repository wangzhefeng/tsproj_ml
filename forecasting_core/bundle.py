"""部署模型的唯一自包含合同；不执行文件写入。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from forecasting_core.probability.calibration import (
    CalibrationSpec,
    ResidualCalibrationSpec,
    validate_cqr_state,
    validate_residual_state,
)
from forecasting_core.probability.spec import ProbabilisticSpec
from typing import Any


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
    execution_mode: str = 'strict'

    def __post_init__(self) -> None:
        version = self.schema_version
        if self.execution_mode not in {'strict', 'research_replay'}:
            raise ValueError('invalid bundle execution_mode')
        if type(version) is not int or version != 2:
            raise ValueError(
                f"Unsupported ForecastModelBundle schema_version={self.schema_version}"
            )
        if self.model is None:
            raise ValueError("ForecastModelBundle.model must not be None")
        self.schema_version = version
        self.selected_features = tuple(str(value) for value in self.selected_features)
        self.input_schema = dict(self.input_schema)
        self.model_type = str(self.model_type).lower()
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
        self.validate_calibration_state()
        if self.calibration_state is not None:
            self.calibration_state = dict(self.calibration_state)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__post_init__()

    def validate_calibration_state(self) -> None:
        """保存/部署也须重验，防对象在构造或反序列化之后被修改。"""
        if not isinstance(self.probabilistic_spec, ProbabilisticSpec):
            raise TypeError("bundle probabilistic_spec must be ProbabilisticSpec")
        calibration = self.probabilistic_spec.calibration
        if isinstance(calibration, CalibrationSpec):
            validate_cqr_state(self.calibration_state, calibration)
            return
        if isinstance(calibration, ResidualCalibrationSpec) and self.calibration_state is None:
            raise ValueError("absolute_residual bundle requires calibration state")
        if self.calibration_state is not None:
            if not isinstance(self.calibration_state, dict):
                raise TypeError("calibration_state must be a dict or None")
            if isinstance(self.probabilistic_spec.calibration, ResidualCalibrationSpec):
                if self.dimensions is None or self.canonical_problem is None:
                    raise ValueError("residual calibration requires canonical dimensions/problem")
                validate_residual_state(self.calibration_state, series_ids=self.series_ids, targets=self.target_order,
                                        shape=self.dimensions, coverage=self.probabilistic_spec.calibration.target_coverage,
                                        freq=self.canonical_problem["freq"])
            else:
                raise ValueError("calibration state requires configured calibration")

    def schema_payload(self) -> dict[str, Any]:
        """返回不含模型对象、可供人工审计的 JSON metadata。"""
        payload = {
            "schema_version": self.schema_version,
            "model_type": self.model_type,
            "selected_features": list(self.selected_features),
            "input_schema": self.input_schema,
            "probabilistic": asdict(self.probabilistic_spec),
        }
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
                "execution_mode": self.execution_mode,
            }
        )
        return payload
