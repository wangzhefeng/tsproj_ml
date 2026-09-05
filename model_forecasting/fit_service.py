"""Canonical fold/final-fit and prediction services."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from forecasting_core.checkpoints import FitCheckpoint
from forecasting_core.artifacts import MarginalForecastDistribution
from forecasting_core.runtime_resources import RuntimeExecutionPlan
from forecasting_core.specs import ForecastConfigSpec, TargetAdapter
from forecasting_core.tensors import PointForecastTensor
from model_forecasting.design import _RegistryDesignBuilder
from model_forecasting.forecaster import (
    CanonicalForecaster,
    CanonicalMarginalQuantileForecaster,
)
from model_forecasting.transforms import CanonicalFeatureScaler, CanonicalTargetTransform
from model_forecasting.transform_windows import select_transform_history
from model_training.estimators import (
    SharedMultiQuantilePool,
    make_model_factory,
    resolve_model_capabilities,
    supports_native_multi_quantile,
)
from model_training.trainer import CanonicalTrainer
from model_forecasting.resource_planner import (
    build_runtime_workload,
    plan_runtime_execution,
    runtime_estimator_params as _planned_estimator_params,
)
from probabilistic.training import CanonicalMarginalQuantileTrainer


def _fit_runtime_transforms(
    config: ForecastConfigSpec,
    builder: _RegistryDesignBuilder,
    X_by_call: tuple[np.ndarray, ...],
    Y: np.ndarray,
    origins: tuple[pd.Timestamp, ...],
    sample_series_ids: tuple[Any, ...],
    history_cutoff: pd.Timestamp,
    *,
    target_history: PointForecastTensor | None = None,
) -> tuple[
    CanonicalFeatureScaler,
    CanonicalTargetTransform,
    tuple[np.ndarray, ...],
    np.ndarray,
]:
    feature_scaler = CanonicalFeatureScaler.from_config(
        config,
        feature_names=builder.feature_schema,
        categorical_names=builder.categorical_schema,
        category_orders={
            column: tuple(
                dict.fromkeys(
                    (
                        identity[index]
                        if isinstance(identity, tuple)
                        else identity
                    )
                    for identity in builder.series_ids
                )
            )
            for index, column in enumerate(config.problem.series_id_cols)
        },
    )
    transformed_X = feature_scaler.fit_transform_calls(X_by_call)
    target_transform = CanonicalTargetTransform.from_config(config)
    if target_transform.is_identity:
        target_transform.fit_identity(builder.series_ids, config.problem.targets)
    else:
        context, scaling_times, window_audit = select_transform_history(
            target_history
            if target_history is not None
            else builder.target_history(history_cutoff),
            origins, horizon=config.problem.horizon, freq=config.problem.freq,
            decomposition_history_steps=target_transform.transformations["decomposition"].get("fit_history_steps"),
        )
        target_transform.fit_transform(context, scaling_times=scaling_times)
        target_transform.fit_window_metadata = window_audit
    transformed_Y = target_transform.transform_training(
        Y,
        origins,
        series_ids=sample_series_ids,
    )
    return feature_scaler, target_transform, transformed_X, transformed_Y


def _forecast_designs_with_scaler(
    builder: _RegistryDesignBuilder,
    origin: pd.Timestamp,
    feature_scaler: CanonicalFeatureScaler,
    target_transform: CanonicalTargetTransform,
):
    raw_designs, raw_provider = builder.forecast_designs(
        origin,
        target_transform=target_transform,
    )

    def provider(call_index, coordinates, dependencies, predicted):
        return feature_scaler.transform(
            raw_provider(call_index, coordinates, dependencies, predicted)
        )

    return (feature_scaler.transform(raw_designs[0]),), provider


def _restore_prediction(
    prediction: PointForecastTensor | MarginalForecastDistribution,
    target_transform: CanonicalTargetTransform,
) -> PointForecastTensor | MarginalForecastDistribution:
    if isinstance(prediction, PointForecastTensor):
        return target_transform.restore_point(prediction)
    if isinstance(prediction, MarginalForecastDistribution):
        return target_transform.restore_distribution(prediction)
    raise TypeError(f"unsupported canonical prediction type: {type(prediction).__name__}")


def _runtime_execution_plan(config: ForecastConfigSpec) -> RuntimeExecutionPlan:
    """未显式传入执行计划的拟合调用方使用的回退计划。"""
    workload = build_runtime_workload(
        config,
        training_rows=0,
        feature_count=1,
        design_bytes=0,
    )
    return plan_runtime_execution(config, workload)


def _fit_point(
    config: ForecastConfigSpec,
    feature_schema: tuple[str, ...],
    X_by_call: tuple[np.ndarray, ...],
    Y: np.ndarray,
    *,
    n_series: int,
    execution_plan: RuntimeExecutionPlan | None = None,
    max_workers: int | None = None,
    checkpoint: FitCheckpoint | None = None,
):
    resolved_plan = execution_plan or _runtime_execution_plan(config)
    runtime_params = _planned_estimator_params(config, resolved_plan)
    if checkpoint is not None:
        checkpoint = checkpoint.child(
            config=config.fingerprint(), feature_schema=feature_schema,
            n_series=n_series, estimator_params=runtime_params,
        )
    capabilities = resolve_model_capabilities(
        config.estimator.model_type,
        runtime_params,
        feature_names=feature_schema,
        probe_native=config.estimator.target_adapter is TargetAdapter.NATIVE,
    )
    trainer = CanonicalTrainer(
        config,
        estimator_factory=make_model_factory(
            config.estimator.model_type,
            runtime_params,
            feature_names=feature_schema,
        ),
        capabilities=capabilities,
        feature_schema=feature_schema,
        checkpoint=checkpoint,
    )
    return trainer, trainer.train(
        X_by_call,
        Y,
        n_series=n_series,
        max_workers=(
            resolved_plan.output_workers
            if max_workers is None
            else max_workers
        ),
    )


def _fit_quantile(
    config: ForecastConfigSpec,
    feature_schema: tuple[str, ...],
    X_by_call: tuple[np.ndarray, ...],
    Y: np.ndarray,
    *,
    n_series: int,
    execution_plan: RuntimeExecutionPlan | None = None,
    worker_plan: tuple[int, int] | None = None,
    checkpoint: FitCheckpoint | None = None,
):
    resolved_plan = execution_plan or _runtime_execution_plan(config)
    runtime_params = _planned_estimator_params(config, resolved_plan)
    if checkpoint is not None:
        checkpoint = checkpoint.child(
            config=config.fingerprint(), feature_schema=feature_schema,
            n_series=n_series, estimator_params=runtime_params,
        )
    capabilities = resolve_model_capabilities(
        config.estimator.model_type,
        runtime_params,
        feature_names=feature_schema,
        probe_native=config.estimator.target_adapter is TargetAdapter.NATIVE,
    )
    if not capabilities.scalar_quantile:
        raise ValueError(
            f"model_type {config.estimator.model_type!r} does not support scalar quantiles"
        )
    # xgb 原生多分位（2026-09-01）：单 booster 输出整个 grid，训练成本
    # ≈1× 而非 Q×；共享位置对齐要求逐 level 串行。其余模型走逐 level
    # 独立训练 + 线程并行（数值与历史串行完全一致）。
    if supports_native_multi_quantile(config.estimator.model_type):
        levels = tuple(
            float(level) for level in config.probabilistic.get("quantiles", ())
        )
        pool = SharedMultiQuantilePool(
            config.estimator.model_type,
            runtime_params,
            levels,
            feature_schema,
        )
        # Persist the completed shared booster, not level slices holding a pool.
        # Every level still creates every position, preserving positional alignment.
        pool.checkpoint = checkpoint
        trainer = CanonicalMarginalQuantileTrainer(
            config,
            estimator_factory_for_level=lambda level: pool.factory_for_level(
                levels.index(level)
            ),
            capabilities=capabilities,
            feature_schema=feature_schema,
        )
        return (
            trainer,
            trainer.train(X_by_call, Y, n_series=n_series, max_workers=1),
            capabilities,
        )
    trainer = CanonicalMarginalQuantileTrainer(
        config,
        estimator_factory_for_level=lambda level: make_model_factory(
            config.estimator.model_type,
            runtime_params,
            feature_names=feature_schema,
            quantile=level,
        ),
        capabilities=capabilities,
        feature_schema=feature_schema,
        checkpoint=checkpoint,
    )
    level_workers, output_workers = (
        (resolved_plan.quantile_workers, resolved_plan.output_workers)
        if worker_plan is None
        else worker_plan
    )
    return (
        trainer,
        trainer.train(
            X_by_call,
            Y,
            n_series=n_series,
            max_workers=level_workers,
            output_workers=output_workers,
        ),
        capabilities,
    )


def _predict(
    config: ForecastConfigSpec,
    artifact,
    base_design: np.ndarray,
    provider,
    forecast_times: pd.DatetimeIndex,
    series_ids: tuple[Any, ...],
):
    kwargs = {
        "series_ids": series_ids,
        "forecast_times": forecast_times,
        "feature_provider": provider,
    }
    if str(config.probabilistic.get("mode", "point")) == "point":
        return CanonicalForecaster(config, artifact).predict(base_design, **kwargs)
    return CanonicalMarginalQuantileForecaster(config, artifact).predict(
        base_design,
        **kwargs,
    )
