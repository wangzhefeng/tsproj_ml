"""自包含 canonical 策略 bundle 的部署预测。

部署调用方按 ``bundle.input_schema`` 列序提供编译好的特征行。本模块负责
全部持久化预处理、selected-feature 对齐、策略执行、分位组装与目标空间
逆变换；不读模型 YAML 与训练/OOF 缓存。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from feature_engineering.selection import selected_indices_for_artifact
from forecasting_core.bundle import ForecastModelBundle
from forecasting_core.probability.distribution import MarginalForecastDistribution
from forecasting_core.specs import ForecastStrategySpec
from forecasting_core.tensors.point import PointForecastTensor
from forecasting_core.execution.strategy import FeatureProvider
from model_predicting.loops.predictor import assemble_marginal_quantile_distribution
from probabilistic.calibration import pi_column_names
from probabilistic.residual import apply_residual_state
from forecasting_core.probability.intervals import PointIntervalForecast
from forecasting_core.probability.calibration import ResidualCalibrationSpec
from model_training.strategies import (
    CanonicalStrategyArtifact,
    get_standard_executor,
)
from model_training.quantile import CanonicalMarginalQuantileArtifact


def predict_strategy_bundle(
    bundle: ForecastModelBundle,
    raw_design: np.ndarray,
    *,
    forecast_times: pd.DatetimeIndex,
    series_ids: tuple[Any, ...] | None = None,
    raw_feature_provider: FeatureProvider | None = None,
    purpose: str = 'production',
) -> PointForecastTensor | MarginalForecastDistribution | PointIntervalForecast:
    """从单个 schema-2 策略 bundle 预测，不做 config 或缓存 IO。"""
    if not isinstance(bundle, ForecastModelBundle) or bundle.schema_version != 2:
        raise TypeError("bundle must be a schema-2 ForecastModelBundle")
    bundle.validate_calibration_state()
    if purpose not in {'production', 'research_replay'}:
        raise ValueError('invalid deployment purpose')
    if bundle.execution_mode not in {'strict', 'research_replay'}:
        raise ValueError('invalid bundle execution_mode')
    if bundle.execution_mode == 'research_replay' and purpose != 'research_replay':
        raise ValueError('research bundle is not eligible for production deployment')
    if bundle.strategy_spec is None or bundle.ensemble_spec is not None:
        raise ValueError("predict_strategy_bundle requires a single-model bundle")
    strategy = ForecastStrategySpec(**bundle.strategy_spec)
    artifact = bundle.model
    artifact_schema = _artifact_schema(artifact)
    design, provider = _prepare_features(
        bundle,
        raw_design,
        artifact_schema=artifact_schema,
        raw_feature_provider=raw_feature_provider,
    )
    resolved_series_ids = tuple(series_ids or bundle.series_ids)
    times = pd.DatetimeIndex(forecast_times)
    if isinstance(artifact, CanonicalStrategyArtifact):
        transformed = _predict_point(
            strategy,
            artifact,
            design,
            series_ids=resolved_series_ids,
            forecast_times=times,
            feature_provider=provider,
        )
    elif isinstance(artifact, CanonicalMarginalQuantileArtifact):
        transformed = _predict_quantiles(
            strategy,
            artifact,
            design,
            series_ids=resolved_series_ids,
            forecast_times=times,
            feature_provider=provider,
            crossing_method=bundle.probabilistic_spec.crossing_method,
            crossing_report_raw=bundle.probabilistic_spec.crossing_report_raw,
        )
    else:
        raise TypeError(
            "strategy bundle model must be CanonicalStrategyArtifact or "
            "CanonicalMarginalQuantileArtifact"
        )
    if bundle.target_transform is None:
        restored = transformed
    elif isinstance(transformed, PointForecastTensor):
        restored = bundle.target_transform.restore_point(transformed)
    else:
        restored = bundle.target_transform.restore_distribution(transformed)
    if isinstance(restored, MarginalForecastDistribution):
        attach_bundle_prediction_intervals(bundle, restored)
    elif isinstance(bundle.probabilistic_spec.calibration, ResidualCalibrationSpec):
        if not isinstance(restored, PointForecastTensor) or bundle.calibration_state is None:
            raise ValueError("absolute_residual deployment requires saved point calibration state")
        return apply_residual_state(restored, bundle.calibration_state)
    return restored


def attach_bundle_prediction_intervals(
    bundle: ForecastModelBundle,
    distribution: MarginalForecastDistribution,
) -> None:
    """把 bundle 内 CQR 校准状态应用为部署期 predict_pi 区间（就地写 metadata）。

    修正量在 final fit 时由回测校准池冻结；部署不重新校准，保证
    bundle 自包含、不读训练期数据。数组形状 (N,H,K)，与 quantiles 张量同轴。
    """
    bundle.validate_calibration_state()
    state = bundle.calibration_state
    if not state or state.get("status") != "applied":
        return
    if state.get("forecast_origin") is not None and distribution.point.forecast_times[0] <= pd.Timestamp(state["forecast_origin"]):
        raise ValueError("saved calibration cannot predict at or before its calibration origin")
    interval = bundle.probabilistic_spec.calibration_interval
    if interval is None:
        return
    correction = float(state["correction"])
    levels = list(distribution.quantiles.levels)
    lower_index = levels.index(interval.lower_quantile)
    upper_index = levels.index(interval.upper_quantile)
    lower = distribution.quantiles.values[..., lower_index] - correction
    upper = distribution.quantiles.values[..., upper_index] + correction
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
        raise ValueError("CQR deployment bounds must be finite and ordered")
    lower_col, upper_col = pi_column_names(float(state["target_coverage"]))
    distribution.metadata["prediction_intervals"] = {
        interval.name: {
            "method": "cqr",
            "target_coverage": float(state["target_coverage"]),
            "correction": correction,
            "lower": lower,
            "upper": upper,
            "columns": [lower_col, upper_col],
        }
    }


def _artifact_schema(
    artifact: CanonicalStrategyArtifact | CanonicalMarginalQuantileArtifact,
) -> tuple[str, ...]:
    if isinstance(artifact, CanonicalStrategyArtifact):
        return tuple(artifact.feature_schema)
    if isinstance(artifact, CanonicalMarginalQuantileArtifact):
        point = artifact.artifacts_by_level[artifact.point_level]
        return tuple(point.feature_schema)
    raise TypeError("unsupported canonical bundle artifact")


def _prepare_features(
    bundle: ForecastModelBundle,
    raw_design: np.ndarray,
    *,
    artifact_schema: tuple[str, ...],
    raw_feature_provider: FeatureProvider | None,
) -> tuple[np.ndarray, FeatureProvider | None]:
    columns = bundle.input_schema.get("columns")
    if not isinstance(columns, list) or any(not isinstance(value, str) for value in columns):
        raise ValueError("strategy bundle input_schema.columns must be a string list")
    # 原始 Global 设计可含字符串类别；必须先经过保存的编码器再转数值。
    design = np.asarray(raw_design)
    if design.ndim != 2 or design.shape[1] != len(columns):
        raise ValueError(
            "raw deployment design must be two-dimensional and match "
            "bundle.input_schema.columns"
        )
    if bundle.feature_scaler is not None:
        design = bundle.feature_scaler.transform(design)
    design = np.asarray(design, dtype=float)
    indices = selected_indices_for_artifact(tuple(columns), artifact_schema)
    if indices is not None:
        design = design[:, indices]

    if raw_feature_provider is None:
        return design, None

    def provider(call_index, coordinates, dependencies, predicted):
        values = np.asarray(
            raw_feature_provider(call_index, coordinates, dependencies, predicted),
        )
        if bundle.feature_scaler is not None:
            values = bundle.feature_scaler.transform(values)
        values = np.asarray(values, dtype=float)
        if indices is not None:
            values = values[:, indices]
        return values

    return design, provider


def _predict_point(
    strategy: ForecastStrategySpec,
    artifact: CanonicalStrategyArtifact,
    design: np.ndarray,
    *,
    series_ids: tuple[Any, ...],
    forecast_times: pd.DatetimeIndex,
    feature_provider: FeatureProvider | None,
) -> PointForecastTensor:
    resolved = strategy.resolve(artifact.H)
    if artifact.target_plan.strategy_name is not strategy.name:
        raise ValueError("bundle strategy does not match strategy artifact")
    executor_type = get_standard_executor(strategy)
    executor = executor_type(strategy, artifact.target_plan, artifact.predictors)
    if resolved.consumes_previous and feature_provider is None:
        raise ValueError("recursive strategy deployment requires raw_feature_provider")
    return executor.predict(
        design,
        series_ids=series_ids,
        forecast_times=forecast_times,
        feature_provider=feature_provider,
    )


def _predict_quantiles(
    strategy: ForecastStrategySpec,
    artifact: CanonicalMarginalQuantileArtifact,
    design: np.ndarray,
    *,
    series_ids: tuple[Any, ...],
    forecast_times: pd.DatetimeIndex,
    feature_provider: FeatureProvider | None,
    crossing_method: str = "median_preserving_isotonic",
    crossing_report_raw: bool = True,
) -> MarginalForecastDistribution:
    def predict_level(level_artifact, provider):
        return _predict_point(
            strategy,
            level_artifact,
            design,
            series_ids=series_ids,
            forecast_times=forecast_times,
            feature_provider=provider,
        )

    # 交叉修复以 bundle 内概率规格为准（部署期不读 YAML），与训练期同口径；
    # median path 组装段与训练期共享 assemble_marginal_quantile_distribution。
    return assemble_marginal_quantile_distribution(
        artifact,
        crossing_method=crossing_method,
        report_raw=crossing_report_raw,
        feature_provider=feature_provider,
        predict_level=predict_level,
    )


__all__ = ["predict_strategy_bundle", "attach_bundle_prediction_intervals"]
