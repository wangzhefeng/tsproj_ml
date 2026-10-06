"""Canonical fold/final-fit and prediction services."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from forecasting_core.execution.checkpoints import FitCheckpoint
from forecasting_core.probability.distribution import MarginalForecastDistribution
from forecasting_core.execution.resources import RuntimeExecutionPlan
from forecasting_core.specs import ForecastConfigSpec, TargetAdapter
from forecasting_core.tensors.point import PointForecastTensor
from pipeline.supervised_design import SupervisedDesignBuilder
from model_predicting.loops.predictor import (
    CanonicalForecaster,
    CanonicalMarginalQuantileForecaster,
)
from feature_engineering.transforms import CanonicalFeatureScaler, CanonicalTargetTransform
from feature_engineering.transforms.windows import select_transform_history
from model_training.estimators import (
    SharedMultiQuantilePool,
    make_model_factory,
    resolve_model_capabilities,
    supports_native_multi_quantile,
)
from model_training.trainer import CanonicalTrainer
from model_performance.resource_planner import (
    build_runtime_workload,
    plan_runtime_execution,
    runtime_estimator_params as _planned_estimator_params,
)
from model_training.quantile import CanonicalMarginalQuantileTrainer
from model_training.weights import resolve_training_sample_weight


def _fit_runtime_transforms(
    config: ForecastConfigSpec,
    builder: SupervisedDesignBuilder,
    X_by_call: tuple[np.ndarray, ...],
    Y: np.ndarray,
    training_origins: tuple[pd.Timestamp, ...],
    training_sample_series_ids: tuple[Any, ...],
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
            training_origins, horizon=config.problem.horizon, freq=config.problem.freq,
            decomposition_history_steps=target_transform.transformations["decomposition"].get("fit_history_steps"),
        )
        target_transform.fit_transform(context, scaling_times=scaling_times)
        target_transform.fit_window_metadata = window_audit
    transformed_Y = target_transform.transform_training(
        Y,
        training_origins,
        series_ids=training_sample_series_ids,
    )
    return feature_scaler, target_transform, transformed_X, transformed_Y


def _forecast_designs_with_scaler(
    builder: SupervisedDesignBuilder,
    origin: pd.Timestamp,
    feature_scaler: CanonicalFeatureScaler,
    target_transform: CanonicalTargetTransform,
    *,
    data_phase: str = "historical",
):
    raw_designs, raw_provider = builder.forecast_designs(
        origin,
        target_transform=target_transform,
        data_phase=data_phase,
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
    sample_weight: np.ndarray | None = None,
):
    """点预测拟合：构造 CanonicalTrainer 并在给定训练窗上拟合。

    estimator 运行参数由执行计划推导（线程档等）；checkpoint 开启时以
    config 指纹 + 特征 schema + 参数为子节点身份，命中即复用完成拟合。
    sample_weight 为 None 时不加权。
    """
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
        sample_weight=sample_weight,
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
    sample_weight: np.ndarray | None = None,
):
    """分位预测拟合：逐 quantile level 训练边际分位模型。

    两条路径：支持原生多分位的模型（xgboost）走共享 booster 池，单次训练
    输出全 level；其余模型逐 level 独立构造 estimator，level 间线程并行。
    worker_plan 为 (level_workers, output_workers) 显式覆盖执行计划。
    """
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
    # 原生多分位（xgboost）：单 booster 输出整个分位 grid，训练成本 ≈1×
    # 而非 Q×；共享 booster 的位置对齐要求逐 level 串行。其余模型走逐
    # level 独立训练 + 线程并行（数值与串行完全一致）。
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
            trainer.train(X_by_call, Y, sample_weight=sample_weight,
                          n_series=n_series, max_workers=1),
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
            sample_weight=sample_weight,
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
    """按 probabilistic.mode 分派 forecaster 执行递归预测。

    point 走 CanonicalForecaster，quantile 走 CanonicalMarginalQuantileForecaster；
    递归策略的第 2+ 步特征经 provider 重新编译（消费已预测目标值）。
    """
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
