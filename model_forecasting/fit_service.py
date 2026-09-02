"""Canonical fold/final-fit and prediction services."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from forecasting_core.artifacts import MarginalForecastDistribution
from forecasting_core.specs import ForecastConfigSpec, TargetAdapter
from forecasting_core.tensors import PointForecastTensor
from model_forecasting.design import _RegistryDesignBuilder
from model_forecasting.forecaster import (
    CanonicalForecaster,
    CanonicalMarginalQuantileForecaster,
)
from model_forecasting.transforms import CanonicalFeatureScaler, CanonicalTargetTransform
from model_training.estimators import (
    SharedMultiQuantilePool,
    make_model_factory,
    resolve_model_capabilities,
    supports_native_multi_quantile,
)
from model_training.trainer import CanonicalTrainer
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
    target_transform.fit_transform(
        target_history
        if target_history is not None
        else builder.target_history(history_cutoff)
    )
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


_THREAD_PARAM_BY_MODEL = {
    "lgb": "n_jobs",
    "lightgbm": "n_jobs",
    "xgb": "n_jobs",
    "xgboost": "n_jobs",
    "rf": "n_jobs",
    "randomforest": "n_jobs",
    "cat": "thread_count",
    "catboost": "thread_count",
}


_DEFAULT_OUTPUT_WORKERS_BY_MODEL = {
    "lgb": 4,
    "lightgbm": 4,
    "xgb": 4,
    "xgboost": 4,
    "cat": 2,
    "catboost": 2,
    "rf": 2,
    "randomforest": 2,
    "ridge": 4,
    "enet": 4,
    "elasticnet": 4,
    "lasso": 4,
    "qr": 2,
    "quantileregressor": 2,
}


def _runtime_scalar_fit_count(config: ForecastConfigSpec) -> int:
    if config.strategy is None:
        return 1
    resolved = config.strategy.resolve(config.problem.horizon)
    count = resolved.model_count
    if config.estimator.target_adapter is TargetAdapter.INDEPENDENT:
        count *= resolved.steps_per_call * len(config.problem.targets)
    return count


def _runtime_estimator_params(config: ForecastConfigSpec) -> dict[str, Any]:
    """Resolve non-semantic estimator threading controls for one fit."""
    params = dict(config.estimator.params)
    performance = config.validation.get("performance", {})
    if not isinstance(performance, Mapping):
        raise TypeError("validation.performance must be a mapping")
    configured = performance.get("model_thread_count")
    normalized_model = config.estimator.model_type.strip().lower()
    thread_param = _THREAD_PARAM_BY_MODEL.get(normalized_model)

    if configured is None:
        if (
            _runtime_scalar_fit_count(config) > 1
            and thread_param is not None
            and thread_param not in params
        ):
            # 多输出/多 horizon 策略由外层保序任务池并行；每个子模型
            # 固定单线程，避免 estimator 内外层线程乘法放大。
            params[thread_param] = 1
        return params

    if isinstance(configured, bool) or not isinstance(configured, int) or configured <= 0:
        raise ValueError("validation.performance.model_thread_count must be positive")
    if thread_param is None:
        raise ValueError(
            "validation.performance.model_thread_count is unsupported for "
            f"model_type {normalized_model!r}"
        )
    existing = params.get(thread_param)
    if existing is not None and existing != configured:
        raise ValueError(
            f"estimator.params.{thread_param} conflicts with "
            "validation.performance.model_thread_count"
        )
    params[thread_param] = configured
    return params


def _runtime_model_workers(config: ForecastConfigSpec) -> int:
    """Resolve workers for independent strategy estimator fit tasks."""
    if config.strategy is None:
        return 1
    task_count = _runtime_scalar_fit_count(config)
    performance = config.validation.get("performance", {})
    if not isinstance(performance, Mapping):
        raise TypeError("validation.performance must be a mapping")
    configured = performance.get("multi_output_n_jobs")
    if configured is None:
        normalized_model = config.estimator.model_type.strip().lower()
        default_workers = _DEFAULT_OUTPUT_WORKERS_BY_MODEL.get(normalized_model, 1)
        return min(default_workers, task_count)
    if isinstance(configured, bool) or not isinstance(configured, int) or configured <= 0:
        raise ValueError("validation.performance.multi_output_n_jobs must be positive")
    return min(configured, task_count)


def _runtime_fit_worker_plan(config: ForecastConfigSpec) -> tuple[int, int]:
    """Resolve mutually exclusive quantile-level and estimator-output workers."""
    output_workers = _runtime_model_workers(config)
    if str(config.probabilistic.get("mode", "point")) != "quantile":
        return 1, output_workers
    if supports_native_multi_quantile(config.estimator.model_type):
        return 1, 1

    levels = tuple(config.probabilistic.get("quantiles", ()))
    if not levels:
        raise ValueError("quantile worker planning requires a nonempty quantile grid")
    performance = config.validation.get("performance", {})
    if not isinstance(performance, Mapping):
        raise TypeError("validation.performance must be a mapping")
    configured = performance.get("quantile_parallel_workers")
    if configured is not None:
        if isinstance(configured, bool) or not isinstance(configured, int) or configured <= 0:
            raise ValueError(
                "validation.performance.quantile_parallel_workers must be positive"
            )
        level_workers = min(configured, len(levels))
        return level_workers, 1 if level_workers > 1 else output_workers
    if output_workers > 1:
        return 1, output_workers
    return min(4, len(levels)), 1


def _fit_point(
    config: ForecastConfigSpec,
    feature_schema: tuple[str, ...],
    X_by_call: tuple[np.ndarray, ...],
    Y: np.ndarray,
    *,
    n_series: int,
    max_workers: int | None = None,
):
    runtime_params = _runtime_estimator_params(config)
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
    )
    return trainer, trainer.train(
        X_by_call,
        Y,
        n_series=n_series,
        max_workers=(
            _runtime_model_workers(config)
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
    worker_plan: tuple[int, int] | None = None,
):
    runtime_params = _runtime_estimator_params(config)
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
    )
    level_workers, output_workers = (
        _runtime_fit_worker_plan(config)
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
