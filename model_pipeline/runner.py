"""Executable canonical runtime without legacy configuration translation."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from model_testing import geometry as validation
from data_loading import (
    BUILTIN_GENERATORS,
    SourceRegistry,
)
from model_training.estimators import make_model_factory
from feature_engineering import cache as compiled_cache
from feature_engineering.selection import (
    CanonicalFeatureSelector,
    normalize_feature_selection,
    selected_indices_for_artifact,
)
from utils.log_util import logger
from forecasting_core.specs import CalendarMonthBacktestSpec, FixedStepBacktestSpec, ForecastConfigSpec
from model_testing.primitives import resolve_origin, seasonal_naive_tensor
from model_forecasting.persistence import (
    build_strategy_model_bundle,
    persist_model_bundle,
)
from model_training.strategies import CanonicalStrategyArtifact
from forecasting_core.tensors import PointForecastTensor
from feature_engineering.transforms import CanonicalFeatureScaler, CanonicalTargetTransform
from model_performance.checkpoints import (
    FileFitCheckpoint, implementation_fingerprint, runtime_checkpoint_errors,
)
from model_performance.performance_profiles import resolve_performance_profile
from model_performance.transform_cache import (
    FoldTransformCache,
    fold_transform_fingerprint,
)
from model_training.trainer import CanonicalTrainer
from forecasting_core.artifacts import ForecastModelBundle, MarginalForecastDistribution
from forecasting_core.runtime_resources import (
    RuntimeExecutionPlan,
    RuntimeResourceBudget,
)
from model_forecasting.evidence import collect_model_evidence, dependency_versions, json_evidence
from model_pipeline.lifecycle import CanonicalRuntimeResult, run_lifecycle
from model_testing.contracts import BacktestWindow
from model_pipeline.supervised_design import (
    _BacktestWindow,
    SupervisedDesignBuilder,
    _actual_at_origin,
    _label_end,
    _rolling_backtest_windows,
    _sample_indices,
    _supervised_arrays,
)
from model_pipeline.fold_fit import (
    _fit_point,
    _fit_quantile,
    _fit_runtime_transforms,
    _forecast_designs_with_scaler,
    _predict,
    _restore_prediction,
)
from model_performance.resource_planner import (
    build_runtime_workload,
    ensemble_member_execution_plan,
    plan_runtime_execution,
    runtime_budget_for_config,
)

# P3/D3：回测原语已公开化至 model_testing/backtest.py（2026-09-06 R1 清扫：
# 删除写而不用的 _positive_validation_int/_actual_tensor 别名，仍用的两处改公开名）。


def _sample_selector(
    origin_indices: tuple[int, ...],
    *,
    n_series: int,
) -> slice | tuple[int, ...]:
    """Use a zero-copy slice when origin-major sample rows are contiguous."""
    indices = _sample_indices(origin_indices, n_series)
    if not indices:
        return ()
    if indices == tuple(range(indices[0], indices[-1] + 1)):
        return slice(indices[0], indices[-1] + 1)
    return indices


class CanonicalBaseModelRunner:
    """Public narrow facade over the validated single-model runtime path.

    Extracted in E1 (v4 §6.1). Single-model `run_canonical_config` delegates to
    `run`; ensemble members (E5+) depend only on this facade, never on runtime
    private helpers.
    """

    @runtime_checkpoint_errors
    def __init__(
        self,
        config: ForecastConfigSpec,
        registry: SourceRegistry,
        origin: pd.Timestamp,
        *,
        compiled_cache_root: str | Path | None = None,
        resource_budget: RuntimeResourceBudget | None = None,
        precompiled_payload: Mapping[str, Any] | None = None,
        precompiled_fingerprint: str | None = None,
        fold_transform_cache: FoldTransformCache | None = None,
        checkpoint_root: str | Path | None = None,
    ) -> None:
        if not isinstance(config, ForecastConfigSpec):
            raise TypeError("config must be a ForecastConfigSpec")
        if config.strategy is None:
            raise ValueError("CanonicalBaseModelRunner requires a strategy")
        self.config = config
        self.calendar_runner_factory: Any = CanonicalBaseModelRunner
        self.registry = registry
        self.origin = origin
        self.builder = SupervisedDesignBuilder(config, registry)
        self.checkpoint_root = checkpoint_root
        self.checkpoint: FileFitCheckpoint | None = None
        if checkpoint_root is not None:
            self.checkpoint = FileFitCheckpoint(checkpoint_root, {
                "config": config.fingerprint(),
                "raw_design": compiled_cache.compute_raw_design_fingerprint(
                    config, base_dir=registry._base_dir, origin=origin,
                    generators=registry._generators),
                "implementation": implementation_fingerprint(),
            })
        self.fold_transform_cache = fold_transform_cache
        self.compiled_cache_hit = False
        self.compiled_cache_fingerprint: str | None = None
        self.X_all: tuple[np.ndarray, ...]
        self.Y_all: np.ndarray
        self.supervised_origins: tuple[pd.Timestamp, ...]
        self.supervised_sample_origins: tuple[pd.Timestamp, ...]
        self.supervised_sample_series_ids: tuple[Any, ...]
        design_started = perf_counter()
        cache_status = "disabled"
        self.cache_stage_wall_seconds = {
            "load": 0.0,
            "compile": 0.0,
            "write": 0.0,
        }
        if precompiled_payload is not None:
            if not precompiled_fingerprint:
                raise ValueError(
                    "precompiled_fingerprint is required with precompiled_payload"
                )
            expected_fingerprint = compiled_cache.compute_raw_design_fingerprint(
                config,
                base_dir=registry._base_dir,
                origin=origin,
                generators=registry._generators,
            )
            if precompiled_fingerprint != expected_fingerprint:
                raise ValueError(
                    "precompiled fingerprint mismatch: "
                    f"got {precompiled_fingerprint}, expected {expected_fingerprint}"
                )
            self._restore_compiled_cache(precompiled_payload)
            self.compiled_cache_hit = True
            self.compiled_cache_fingerprint = precompiled_fingerprint
            cache_status = "memory_hit"
        elif compiled_cache_root is None:
            compile_started = perf_counter()
            self._compile_supervised_arrays()
            self.cache_stage_wall_seconds["compile"] = (
                perf_counter() - compile_started
            )
        else:
            cache_root = Path(compiled_cache_root)
            fingerprint = compiled_cache.compute_raw_design_fingerprint(
                config,
                base_dir=registry._base_dir,
                origin=origin,
                generators=registry._generators,
            )
            self.compiled_cache_fingerprint = fingerprint
            started = perf_counter()
            with compiled_cache.raw_design_cache_lock(cache_root, fingerprint):
                load_started = perf_counter()
                try:
                    payload = compiled_cache.load_compiled_cache(cache_root, fingerprint)
                except FileNotFoundError:
                    self.cache_stage_wall_seconds["load"] = (
                        perf_counter() - load_started
                    )
                    compile_started = perf_counter()
                    self._compile_supervised_arrays()
                    self.cache_stage_wall_seconds["compile"] = (
                        perf_counter() - compile_started
                    )
                    cache_status = "miss"
                    write_started = perf_counter()
                    compiled_cache.save_compiled_cache(
                        cache_root,
                        fingerprint,
                        self._compiled_cache_payload(),
                    )
                    self.cache_stage_wall_seconds["write"] = (
                        perf_counter() - write_started
                    )
                    logger.info(
                        "[CompiledFeaturesCache] miss key=%s compile_seconds=%.3f",
                        fingerprint[:12],
                        perf_counter() - started,
                    )
                else:
                    self.cache_stage_wall_seconds["load"] = (
                        perf_counter() - load_started
                    )
                    self._restore_compiled_cache(payload)
                    self.compiled_cache_hit = True
                    cache_status = "hit"
                    logger.info(
                        "[CompiledFeaturesCache] hit key=%s load_seconds=%.3f",
                        fingerprint[:12],
                        perf_counter() - started,
                    )
        self.workload = build_runtime_workload(
            config,
            training_rows=len(self.Y_all),
            feature_count=len(self.builder.feature_schema),
            design_bytes=sum(design.nbytes for design in self.X_all) + self.Y_all.nbytes,
            series_count=self.builder.n_series,
        )
        self.resource_budget = runtime_budget_for_config(
            config,
            parent_budget=resource_budget,
        )
        self.execution_plan = plan_runtime_execution(
            config,
            self.workload,
            budget=self.resource_budget,
            base_dir=registry._base_dir,
            feature_schema=self.builder.feature_schema,
        )
        self.cache_status = cache_status
        self.training_compile = self.builder.training_compile_summary()
        if self.compiled_cache_hit:
            self.training_compile["mode"] = "cache_hit"
        logger.info(
            "[TrainingCompile] mode=%s origins=%s calls=%s reasons=%s wall=%s",
            self.training_compile["mode"],
            self.training_compile["origin_count"],
            self.training_compile["call_count"],
            self.training_compile["reason_codes"],
            self.training_compile["stage_wall_seconds"],
        )
        self.lifecycle_started = design_started
        self.stage_wall_seconds: dict[str, float] = {
            "raw_design": perf_counter() - design_started,
        }

    def _compile_supervised_arrays(self) -> None:
        (
            self.X_all,
            self.Y_all,
            self.supervised_origins,
            self.supervised_sample_origins,
            self.supervised_sample_series_ids,
        ) = _supervised_arrays(self.builder, self.origin)

    def _compiled_cache_payload(self) -> dict[str, Any]:
        return {
            "X_all": self.X_all,
            "Y_all": self.Y_all,
            "supervised_origins": self.supervised_origins,
            "supervised_sample_origins": self.supervised_sample_origins,
            "supervised_sample_series_ids": self.supervised_sample_series_ids,
            "feature_schema": self.builder.feature_schema,
            "categorical_schema": self.builder.categorical_schema,
        }

    def raw_design_payload(self) -> dict[str, Any]:
        """Expose one group's raw arrays for in-memory batch reuse."""
        return self._compiled_cache_payload()

    def _restore_compiled_cache(self, payload: Mapping[str, Any]) -> None:
        required = {
            "X_all",
            "Y_all",
            "supervised_origins",
            "supervised_sample_origins",
            "supervised_sample_series_ids",
            "feature_schema",
            "categorical_schema",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(
                f"compiled feature cache payload is missing fields: {missing}"
            )
        X_all = tuple(np.asarray(design) for design in payload["X_all"])
        Y_all = np.asarray(payload["Y_all"])
        sample_origins = tuple(
            cast(pd.Timestamp, pd.Timestamp(value))
            for value in payload["supervised_sample_origins"]
        )
        if any(len(design) != len(Y_all) for design in X_all):
            raise ValueError("compiled feature cache X/Y sample counts differ")
        if len(sample_origins) != len(Y_all):
            raise ValueError("compiled feature cache origin/Y sample counts differ")
        self.X_all = X_all
        self.Y_all = Y_all
        self.supervised_origins = tuple(
            cast(pd.Timestamp, pd.Timestamp(value))
            for value in payload["supervised_origins"]
        )
        self.supervised_sample_origins = sample_origins
        self.supervised_sample_series_ids = tuple(
            payload["supervised_sample_series_ids"]
        )
        self.builder.feature_schema = tuple(payload["feature_schema"])
        self.builder.categorical_schema = tuple(payload["categorical_schema"])

    @property
    def geometry(self) -> validation.TimeGeometry:
        return validation.TimeGeometry(
            offset=self.builder.offset,
            horizon=self.config.problem.horizon,
        )

    @property
    def series_ids(self) -> tuple[Any, ...]:
        return self.builder.series_ids

    @property
    def feature_schema(self) -> tuple[str, ...]:
        return self.builder.feature_schema

    def runtime_resources_payload(self) -> dict[str, Any]:
        return {
            "performance_profile": resolve_performance_profile(
                self.config, self.workload, budget=self.resource_budget,
                base_dir=self.registry._base_dir, feature_schema=self.feature_schema,
                execution_plan=self.execution_plan,
            ),
            "workload": self.workload.payload(),
            "budget": self.resource_budget.payload(),
            "execution_plan": self.execution_plan.payload(),
            "cache": {
                "status": self.cache_status,
                "fingerprint": self.compiled_cache_fingerprint,
                "stage_wall_seconds": dict(self.cache_stage_wall_seconds),
            },
            "training_compile": dict(self.training_compile),
            "stage_wall_seconds": dict(self.stage_wall_seconds),
        }

    def apply_ensemble_member_plan(self, parent_plan: RuntimeExecutionPlan) -> None:
        """Apply one parent-budgeted serial plan before Ensemble member pools start."""
        self.execution_plan = ensemble_member_execution_plan(
            parent_plan,
            self.workload,
        )

    def backtest_windows(self) -> tuple[_BacktestWindow, ...]:
        if isinstance(self.config.validation.backtest, CalendarMonthBacktestSpec):
            return ()
        return _rolling_backtest_windows(
            self.builder, self.supervised_origins,
            schedule_origin=(self.origin if self.config.validation.get("schedule_mode") == "intraday" else None),
        )

    def _fit_or_reuse_runtime_transforms(
        self,
        *,
        origin_indices: tuple[int, ...],
        X_values: tuple[np.ndarray, ...],
        Y_values: np.ndarray,
        training_origins: tuple[pd.Timestamp, ...],
        training_series_ids: tuple[Any, ...],
        history_cutoff: pd.Timestamp,
        target_history: PointForecastTensor | None = None,
    ) -> tuple[
        CanonicalFeatureScaler,
        CanonicalTargetTransform,
        tuple[np.ndarray, ...],
        np.ndarray,
    ]:
        def factory():
            return _fit_runtime_transforms(
                self.config,
                self.builder,
                X_values,
                Y_values,
                training_origins,
                training_series_ids,
                history_cutoff,
                target_history=target_history,
            )

        if self.fold_transform_cache is None or target_history is not None:
            return factory()
        if self.compiled_cache_fingerprint is None:
            raise ValueError(
                "fold transform cache requires a raw design fingerprint"
            )
        key = fold_transform_fingerprint(
            self.config,
            raw_design_fingerprint=self.compiled_cache_fingerprint,
            origin_indices=origin_indices,
        )
        return self.fold_transform_cache.get_or_create(key, factory)

    @runtime_checkpoint_errors
    def fit(
        self,
        train_indices: tuple[int, ...],
        *,
        target_history: PointForecastTensor | None = None,
        force_serial: bool = False,
    ) -> tuple[
        CanonicalFeatureScaler,
        CanonicalTargetTransform,
        tuple[np.ndarray, ...],
        np.ndarray,
        Any,
    ]:
        """Fit this member's own transforms and model on given train origins.

        Returns ``(feature_scaler, target_transform, X_train_transformed,
        Y_train_transformed, artifact)``; the artifact is a point
        `CanonicalStrategyArtifact` or a quantile
        `CanonicalMarginalQuantileArtifact` depending on the config mode.
        """
        mode = self._mode()
        train_sample_indices = _sample_indices(
            train_indices,
            self.builder.n_series,
        )
        train_selector = _sample_selector(
            train_indices,
            n_series=self.builder.n_series,
        )
        X_train = tuple(design[train_selector] for design in self.X_all)
        Y_train = self.Y_all[train_selector]
        training_origins = tuple(
            self.supervised_sample_origins[index]
            for index in train_sample_indices
        )
        training_series_ids = tuple(
            self.supervised_sample_series_ids[index]
            for index in train_sample_indices
        )
        training_history_cutoff = max(
            _label_end(self.builder, self.supervised_origins[index])
            for index in train_indices
        )
        (
            feature_scaler,
            target_transform,
            X_train_transformed,
            Y_train_transformed,
        ) = self._fit_or_reuse_runtime_transforms(
            origin_indices=train_indices,
            X_values=X_train,
            Y_values=Y_train,
            training_origins=training_origins,
            training_series_ids=training_series_ids,
            history_cutoff=training_history_cutoff,
            target_history=target_history,
        )
        # 监督特征选择（2026-08-30 专项）：有监督步骤挂在训练 fit 边界，
        # 每个回测窗口/最终训练各自重拟合，只消费当前训练窗 (X, Y)，无泄漏；
        # 选中集写入 artifact.feature_schema，预测端按同名子集对齐。
        X_train_transformed, feature_schema = self._apply_feature_selection(
            X_train_transformed, Y_train_transformed
        )
        if mode == "point":
            _, artifact = _fit_point(
                self.config,
                feature_schema,
                X_train_transformed,
                Y_train_transformed,
                n_series=self.builder.n_series,
                execution_plan=self.execution_plan,
                max_workers=1 if force_serial else None,
                checkpoint=(self.checkpoint.child(fold=list(train_indices))
                            if self.checkpoint is not None else None),
            )
        else:
            _, artifact, _ = _fit_quantile(
                self.config,
                feature_schema,
                X_train_transformed,
                Y_train_transformed,
                n_series=self.builder.n_series,
                execution_plan=self.execution_plan,
                worker_plan=(1, 1) if force_serial else None,
                checkpoint=(self.checkpoint.child(fold=list(train_indices))
                            if self.checkpoint is not None else None),
            )
        return (
            feature_scaler,
            target_transform,
            X_train_transformed,
            Y_train_transformed,
            artifact,
        )

    def forecast_designs(
        self,
        origin: pd.Timestamp,
        feature_scaler: CanonicalFeatureScaler,
        target_transform: CanonicalTargetTransform,
    ) -> tuple[tuple[np.ndarray, ...], Any]:
        return _forecast_designs_with_scaler(
            self.builder,
            origin,
            feature_scaler,
            target_transform,
        )

    def predict(
        self,
        artifact: Any,
        designs: tuple[np.ndarray, ...],
        provider: Any,
        forecast_times: pd.DatetimeIndex,
        target_transform: CanonicalTargetTransform,
    ) -> PointForecastTensor | MarginalForecastDistribution:
        """Predict at the given origin and restore to the original target space."""
        base_design = designs[0]
        # 特征选择对齐（2026-08-30 专项）：artifact.feature_schema 是训练期选中集，
        # 预测端把全 schema 设计矩阵按同名子集对齐（provider 输出同为全 schema 宽）。
        artifact_schema = (
            artifact.feature_schema
            if isinstance(artifact, CanonicalStrategyArtifact)
            else next(iter(artifact.artifacts_by_level.values())).feature_schema
        )
        indices = selected_indices_for_artifact(
            self.builder.feature_schema, tuple(artifact_schema)
        )
        if indices is not None:
            base_design = np.asarray(base_design, dtype=float)[:, indices]
            full_provider = provider

            def selected_provider(call_index, coordinates, dependencies, predicted):
                return np.asarray(
                    full_provider(call_index, coordinates, dependencies, predicted),
                    dtype=float,
                )[:, indices]

            provider = selected_provider

        raw = _predict(
            self.config,
            artifact,
            base_design,
            provider,
            forecast_times,
            self.builder.series_ids,
        )
        return _restore_prediction(raw, target_transform)

    def actual(
        self,
        origin_index: int,
        forecast_times: pd.DatetimeIndex,
    ) -> PointForecastTensor:
        return _actual_at_origin(
            self.config,
            self.Y_all,
            origin_index,
            forecast_times,
            self.builder.series_ids,
        )

    def seasonal_naive(
        self,
        origin: pd.Timestamp,
        forecast_times: pd.DatetimeIndex,
    ) -> PointForecastTensor:
        return seasonal_naive_tensor(
            self.builder,
            origin,
            forecast_times,
        )

    def forecast_times(
        self,
        origin: pd.Timestamp,
    ) -> pd.DatetimeIndex:
        return pd.date_range(
            origin,
            periods=self.config.problem.horizon + 1,
            freq=self.config.problem.freq,
        )[1:]

    def target_history(self, origin: pd.Timestamp) -> PointForecastTensor:
        """Full target history as-of origin (forecast-plot context; E5+ Protocol)."""
        return self.builder.target_history(origin)

    def final_bundle_inputs(self) -> tuple[
        CanonicalFeatureScaler,
        CanonicalTargetTransform,
        tuple[np.ndarray, ...],
        np.ndarray,
    ]:
        """Fit final transforms under the same explicit window as backtesting."""
        backtest = self.config.validation.backtest
        if isinstance(backtest, FixedStepBacktestSpec):
            first_origin_index = max(
                0,
                len(self.supervised_origins) - backtest.train_window_steps,
            )
            origin_indices = tuple(
                range(first_origin_index, len(self.supervised_origins))
            )
        elif isinstance(backtest, CalendarMonthBacktestSpec):
            raw_history_times = self.builder.target_history_times(self.origin)
            if len(raw_history_times) < backtest.train_window_days:
                raise ValueError(
                    "calendar-month final fit has fewer raw history days than "
                    "validation.train_window_days"
                )
            train_start_time = pd.Timestamp(
                raw_history_times.to_numpy()[-backtest.train_window_days]
            )
            forecast_start = self.geometry.label_start(self.origin)
            origin_indices = tuple(
                index
                for index, candidate in enumerate(self.supervised_origins)
                if candidate >= train_start_time
                and self.geometry.label_end(candidate) < forecast_start
            )
        else:
            raise TypeError("canonical final fit requires typed backtest geometry")
        if not origin_indices:
            raise ValueError("canonical final fit has no safe supervised samples")
        sample_indices = _sample_indices(origin_indices, self.builder.n_series)
        sample_selector = _sample_selector(
            origin_indices,
            n_series=self.builder.n_series,
        )
        X_window = tuple(design[sample_selector] for design in self.X_all)
        Y_window = self.Y_all[sample_selector]
        sample_origins = tuple(
            self.supervised_sample_origins[index] for index in sample_indices
        )
        sample_series_ids = tuple(
            self.supervised_sample_series_ids[index] for index in sample_indices
        )
        history_cutoff = max(
            _label_end(self.builder, self.supervised_origins[index])
            for index in origin_indices
        )
        (
            feature_scaler,
            target_transform,
            X_all_transformed,
            Y_all_transformed,
        ) = self._fit_or_reuse_runtime_transforms(
            origin_indices=origin_indices,
            X_values=X_window,
            Y_values=Y_window,
            training_origins=sample_origins,
            training_series_ids=sample_series_ids,
            history_cutoff=history_cutoff,
        )
        return feature_scaler, target_transform, X_all_transformed, Y_all_transformed

    @runtime_checkpoint_errors
    def fit_final(
        self,
        X_transformed: tuple[np.ndarray, ...],
        Y_transformed: np.ndarray,
    ) -> tuple[Any, Any, Any]:
        """Train the final artifact and return (trainer, artifact, capabilities)."""
        mode = self._mode()
        X_transformed, feature_schema = self._apply_feature_selection(
            X_transformed, Y_transformed
        )
        if mode == "point":
            trainer, artifact = _fit_point(
                self.config,
                feature_schema,
                X_transformed,
                Y_transformed,
                n_series=self.builder.n_series,
                execution_plan=self.execution_plan,
                checkpoint=(self.checkpoint.child(fold="final")
                            if self.checkpoint is not None else None),
            )
            capabilities = trainer.capabilities
        else:
            trainer, artifact, capabilities = _fit_quantile(
                self.config,
                feature_schema,
                X_transformed,
                Y_transformed,
                n_series=self.builder.n_series,
                execution_plan=self.execution_plan,
                checkpoint=(self.checkpoint.child(fold="final")
                            if self.checkpoint is not None else None),
            )
        return trainer, artifact, capabilities

    def build_final_bundle(
        self,
        feature_scaler: CanonicalFeatureScaler,
        target_transform: CanonicalTargetTransform,
        trainer: Any,
        artifact: Any,
        capabilities: Any,
        extras: Mapping[str, Any] | None = None,
    ) -> ForecastModelBundle:
        """Build a self-contained schema-2 bundle from a completed final fit.

        ``extras``（可选）：单模型生命周期传入的附加产物元数据——
        ``unknown_series_policy``（来自 builder 训练域校验，默认 "raise"）、
        ``availability_summary``、``visibility_proof``、``feature_lineage``、
        ``source_lineage``、``calibration_state``。ensemble 成员路径不传，
        行为与迁移前逐字一致。
        """
        extras = dict(extras or {})
        mode = self._mode()
        if mode == "point":
            bundle_builder = trainer
            bundle_artifact = artifact
        else:
            point_level = float(self.config.probabilistic.get("point_quantile", 0.5))
            bundle_artifact = artifact.artifacts_by_level[point_level]
            selected_schema = tuple(bundle_artifact.feature_schema)
            bundle_builder = CanonicalTrainer(
                self.config,
                estimator_factory=make_model_factory(
                    self.config.estimator.model_type,
                    self.config.estimator.params,
                    feature_names=selected_schema,
                    quantile=point_level,
                ),
                capabilities=capabilities,
                feature_schema=selected_schema,
            )
        input_schema = {
            "columns": list(self.feature_schema),
            "panel": {
                "series_id_cols": list(self.config.problem.series_id_cols),
                "known_series_ids": [
                    list(value) if isinstance(value, tuple) else value
                    for value in self.series_ids
                ],
                "unknown_series_policy": extras.get("unknown_series_policy", "raise"),
            },
        }
        if extras.get("availability_summary") is not None:
            input_schema["availability_summary"] = extras["availability_summary"]
        if extras.get("visibility_proof") is not None:
            input_schema["visibility_proof"] = extras["visibility_proof"]
        bundle_kwargs: dict[str, Any] = {"series_ids": self.series_ids}
        for key in ("feature_lineage", "source_lineage", "calibration_state"):
            if extras.get(key) is not None:
                bundle_kwargs[key] = extras[key]
        bundle = build_strategy_model_bundle(
            bundle_builder,
            bundle_artifact,
            feature_scaler=feature_scaler,
            target_transform=target_transform,
            input_schema=input_schema,
            **bundle_kwargs,
        )
        if mode == "quantile":
            bundle.model = artifact
        return bundle

    def _apply_feature_selection(
        self,
        X_by_call: tuple[np.ndarray, ...],
        Y: np.ndarray,
    ) -> tuple[tuple[np.ndarray, ...], tuple[str, ...]]:
        """监督特征选择（features.selection，2026-08-30 专项）。

        有监督步骤挂在训练 fit 边界：每个回测窗口与最终训练各自重拟合选择器，
        只消费当前训练窗的 (X, Y)，无泄漏；未配置/未启用时原样直通。
        选中集进入 artifact.feature_schema，预测端按同名子集对齐。
        """
        feature_schema = self.builder.feature_schema
        spec = normalize_feature_selection(self.config.features.selection)
        if spec is None or not spec.enabled:
            return X_by_call, feature_schema
        selector = CanonicalFeatureSelector(spec, feature_schema)
        y_signal = Y.reshape(Y.shape[0], -1).mean(axis=1)
        selector.fit(X_by_call[0], y_signal)
        assert selector.selected_names_ is not None  # fit 后必有选中集
        logger.info(
            "[FeatureSelection] %d -> %d features (method=%s)",
            len(feature_schema),
            len(selector.selected_names_),
            spec.method,
        )
        return (
            tuple(selector.transform(design) for design in X_by_call),
            selector.selected_names_,
        )

    @runtime_checkpoint_errors
    def run(
        self,
        output_root: str | Path | None = None,
    ) -> CanonicalRuntimeResult:
        """Execute with process-level BLAS/OpenMP limits set before any pool."""
        with threadpool_limits(limits=self.execution_plan.model_threads):
            return run_lifecycle(self, output_root)

    def run_prelimited(
        self,
        output_root: str | Path | None = None,
    ) -> CanonicalRuntimeResult:
        """Execute under a parent-owned process-level threadpool limit."""
        return run_lifecycle(self, output_root)

    @cached_property
    def _execution_evidence_context(self) -> dict[str, Any]:
        return {"config_fingerprint": self.config.fingerprint(),
                "implementation_fingerprint": implementation_fingerprint(),
                "dependency_versions": dependency_versions()}

    def execution_evidence(self, artifact: Any, target_transform: Any) -> dict[str, Any]:
        """Snapshot one existing fitted unit without fitting or predicting."""
        models = collect_model_evidence(artifact)
        return json_evidence({
            "status": "recorded" if models else "unavailable",
            "reason": None if models else "no_supported_model_wrapper_found",
            **self._execution_evidence_context,
            "target_transform_window": target_transform.fit_window_metadata,
            "fitted_models": models,
        })


    def backtest_target_histories(
        self, windows: tuple[BacktestWindow, ...],
    ) -> tuple[PointForecastTensor | None, ...]:
        if CanonicalTargetTransform.from_config(self.config).is_identity:
            target_histories = (None,) * len(windows)
        else:
            target_histories = tuple(
                self.builder.target_history(
                    max(
                        _label_end(
                            self.builder,
                            cast(pd.Timestamp, self.supervised_origins[index]),
                        )
                        for index in backtest_window.train_indices
                    )
                )
                for backtest_window in windows
            )
        return target_histories

    def _mode(self) -> str:
        mode = str(self.config.probabilistic.get("mode", "point"))
        if mode not in {"point", "quantile"}:
            raise ValueError(
                f"unsupported canonical probabilistic mode: {mode!r}"
            )
        return mode


@runtime_checkpoint_errors
def run_canonical_config(
    config: ForecastConfigSpec,
    output_root: str | Path | None = None,
    *,
    generators: Mapping[str, Any] | None = None,
    checkpoint_root: str | Path | None = None,
) -> CanonicalRuntimeResult:
    """Train, backtest, forecast, and persist one canonical config."""
    if not isinstance(config, ForecastConfigSpec):
        raise TypeError("config must be a ForecastConfigSpec")
    if config.strategy is None:
        raise ValueError("run_canonical_config requires a strategy or ensemble")
    # builtin generators（chinese_holiday）默认可用；调用方同名注入时覆盖。
    merged_generators: dict[str, Any] = {**BUILTIN_GENERATORS, **(generators or {})}
    registry = SourceRegistry(config.data, Path.cwd(), generators=merged_generators)
    origin = resolve_origin(registry, config.validation.get("forecast_origin"))
    compiled_cache_root = (
        Path(output_root)
        if output_root is not None
        else Path(str(config.output.get("results_root", "results")))
    )
    runner = CanonicalBaseModelRunner(
        config,
        registry,
        origin,
        compiled_cache_root=compiled_cache_root,
        checkpoint_root=checkpoint_root,
    )
    return runner.run(output_root)


__all__ = [
    "CanonicalBaseModelRunner",
    "CanonicalRuntimeResult",
    "persist_model_bundle",
    "run_canonical_config",
]
