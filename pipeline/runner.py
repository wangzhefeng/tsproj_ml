"""Executable canonical runtime without legacy configuration translation."""

from __future__ import annotations

from dataclasses import replace
from functools import cached_property
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from model_testing.contracts import geometry as backtest_geometry
from data_loading import (
    BUILTIN_GENERATORS,
    SourceRegistry,
)
from model_training.estimators import make_model_factory
from model_training.weights import (
    resolve_training_sample_weight,
    training_sample_weight_spec,
)
from model_building.adapters.native_registry import native_history_cls as _native_history_cls
from model_building.catalog import MODEL_CATALOG
from feature_engineering import design_identity
from feature_engineering.selection import (
    CanonicalFeatureSelector,
    normalize_feature_selection,
    selected_indices_for_artifact,
)
from utils.log_util import logger
from forecasting_core.specs import (
    CalendarMonthBacktestSpec,
    ExpandingWindowBacktestSpec,
    FixedStepBacktestSpec,
    ForecastConfigSpec,
    SlidingWindowBacktestSpec,
)
from forecasting_core.origin import resolve_origin
from model_testing.contracts.primitives import seasonal_naive_tensor
from model_predicting.artifacts.persistence import (
    build_strategy_model_bundle,
    persist_model_bundle,
)
from model_training.strategies import CanonicalStrategyArtifact
from forecasting_core.design import IndexedDesign, retained_array_bytes
from forecasting_core.specs.temporal import (
    forecast_times as _temporal_forecast_times,
    forecast_ends as _temporal_forecast_ends,
    has_bounded_history,
    history_start as _temporal_history_start,
    select_training_origins,
)
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
from model_predicting.artifacts.evidence_collect import collect_model_evidence, dependency_versions, json_evidence
from pipeline.lifecycle import BacktestRuntimeResult, CanonicalRuntimeResult, run_lifecycle
from model_testing.contracts.protocols import BacktestWindow
from model_testing.contracts.windows import (
    raw_history_backtest_windows,
    rolling_backtest_windows,
    temporal_backtest_windows,
)
from pipeline.supervised_design import (
    SupervisedDesignBuilder,
    _actual_at_origin,
    _label_end,
    _sample_indices,
    _supervised_arrays,
    minimum_history_rows,
    supervised_candidate_origins,
)
from pipeline.fold_fit import (
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

# 原生序列模型（ETS/naive/theta）注册表位于 model_building/adapters/
# native_registry.py：纯 wrapper 接线属模型层，非编排层职责；runner 经
# _native_history_cls 按 model_type 分发。

# 回测公开原语（seasonal-naive 基线、actual 张量）位于
# model_testing/contracts/primitives.py，origin 解析位于 forecasting_core/origin.py；
# 本文件经公开名消费。


def _sample_selector(
    origin_indices: tuple[int, ...],
    *,
    n_series: int,
) -> slice | np.ndarray:
    """Use a zero-copy slice when origin-major sample rows are contiguous."""
    indices = _sample_indices(origin_indices, n_series)
    if not indices:
        return np.array([], dtype=int)
    if indices == tuple(range(indices[0], indices[-1] + 1)):
        return slice(indices[0], indices[-1] + 1)
    return np.asarray(indices, dtype=int)


class CanonicalBaseModelRunner:
    """经验证单模型运行时路径的窄门面。

    `run_canonical_config` 委托 `run` 执行；ensemble 成员只依赖本门面的
    公开能力面，不触碰运行时私有助手。
    """

    @runtime_checkpoint_errors
    def __init__(
        self,
        config: ForecastConfigSpec,
        registry: SourceRegistry,
        origin: pd.Timestamp,
        *,
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
        self._native_cls = _native_history_cls(config.estimator.model_type)
        if self._native_cls is not None:
            # 构造期即校验参数（未知参数 RAISE），与 ETS 原行为一致
            if str(config.probabilistic.get("mode", "point")) != "point":
                # 能力前置校验：native 历史模型不支持 scalar quantile，
                # 在构造期与参数校验同点位 RAISE（batch preflight 亦在此隔离）。
                raise ValueError(
                    f"model_type {config.estimator.model_type!r} (native history) "
                    "does not support probabilistic.mode=quantile"
                )
            self._native_cls(dict(config.estimator.params))
        self.calendar_runner_factory: Any = CanonicalBaseModelRunner
        self.registry = registry
        self.origin = origin
        self.resource_budget = runtime_budget_for_config(config, parent_budget=resource_budget)
        offset = pd.tseries.frequencies.to_offset(config.problem.freq)
        self.builder = SupervisedDesignBuilder(
            config, registry,
            history_start=_temporal_history_start(config.validation, origin, offset),
        )
        self.checkpoint_root = checkpoint_root
        self.checkpoint: FileFitCheckpoint | None = None
        if checkpoint_root is not None:
            self.checkpoint = FileFitCheckpoint(checkpoint_root, {
                "config": config.fingerprint(),
                "raw_design": design_identity.compute_raw_design_fingerprint(
                    config, base_dir=registry.base_dir, origin=origin,
                    generators=registry.generators),
                "implementation": implementation_fingerprint(),
            })
        self.fold_transform_cache = fold_transform_cache
        self.design_shared = False
        self.raw_design_fingerprint: str | None = None
        self.supervised_origins: tuple[pd.Timestamp, ...]
        self.supervised_sample_origins: tuple[pd.Timestamp, ...]
        self.supervised_sample_series_ids: tuple[Any, ...]
        self._precompiled_payload = precompiled_payload
        self._precompiled_fingerprint = precompiled_fingerprint
        self._shared_design_source: CanonicalBaseModelRunner | None = None
        self._preparation_lock = RLock()
        self._prepared = False
        self._X_all: tuple[np.ndarray | IndexedDesign, ...] | None = None
        self._Y_all: np.ndarray | None = None
        self.lifecycle_started = perf_counter()
        self.stage_wall_seconds: dict[str, float] = {"raw_design": 0.0}
        self._final_training_workload: dict[str, Any] | None = None
        self.design_status = "not_prepared"
        self.design_stage_wall_seconds = {"compile": 0.0, "share": 0.0}
        # 规划只探测一个原点的真实 schema；不构造全训练设计，不做数组持久化。
        self.supervised_origins = supervised_candidate_origins(self.builder, self.origin)
        if precompiled_payload is not None:
            if not precompiled_fingerprint:
                raise ValueError(
                    "precompiled_fingerprint is required with precompiled_payload"
                )
            expected = design_identity.compute_raw_design_fingerprint(
                config, base_dir=registry.base_dir, origin=origin,
                generators=registry.generators)
            if precompiled_fingerprint != expected:
                raise ValueError(
                    "precompiled fingerprint mismatch: "
                    f"got {precompiled_fingerprint}, expected {expected}"
                )
        if self._native_cls is not None:
            row_bytes = config.problem.horizon * self.builder.n_series * len(config.problem.targets) * 8
        else:
            planning_builder = SupervisedDesignBuilder(
                config, registry, history_start=self.builder.history_start,
            )
            designs, labels = planning_builder.training_row(self.supervised_origins[0])
            self.builder.feature_schema = planning_builder.feature_schema
            self.builder.categorical_schema = planning_builder.categorical_schema
            row_bytes = sum(d.nbytes for d in designs) + labels.nbytes
        self.workload = build_runtime_workload(
            config,
            training_rows=len(self.supervised_origins) * self.builder.n_series,
            feature_count=len(self.builder.feature_schema),
            design_bytes=len(self.supervised_origins) * row_bytes,
            series_count=self.builder.n_series,
        )
        self.execution_plan = plan_runtime_execution(
            config,
            self.workload,
            budget=self.resource_budget,
            base_dir=registry.base_dir,
            feature_schema=self.builder.feature_schema,
        )
        self.training_compile = self.builder.training_compile_summary()
        self.training_compile["mode"] = "planned"
        if fold_transform_cache is not None:
            self.raw_design_fingerprint = design_identity.compute_raw_design_fingerprint(
                config, base_dir=registry.base_dir, origin=origin,
                generators=registry.generators)

    def prepare_training(self) -> None:
        """按需物化一份训练设计；并发折共享同一次准备。"""
        with self._preparation_lock:
            if not self._prepared:
                try:
                    self._prepare_training()
                except Exception:
                    self._X_all = self._Y_all = None
                    raise
                self._prepared = True
                self._precompiled_payload = None
                self._shared_design_source = None

    @property
    def X_all(self) -> tuple[np.ndarray | IndexedDesign, ...]:
        if self._X_all is None:
            self.prepare_training()
        assert self._X_all is not None
        return self._X_all

    @X_all.setter
    def X_all(self, value):
        self._X_all = value

    @property
    def Y_all(self) -> np.ndarray:
        if self._Y_all is None:
            self.prepare_training()
        assert self._Y_all is not None
        return self._Y_all

    @Y_all.setter
    def Y_all(self, value):
        self._Y_all = value

    def share_training_design(self, source: CanonicalBaseModelRunner) -> bool:
        """共享同指纹的不可变设计；绝不共享对方的拟合状态。"""
        if source is self or self._prepared:
            raise ValueError("design sharing must precede preparation")

        def identity(runner: CanonicalBaseModelRunner) -> str:
            return design_identity.compute_raw_design_fingerprint(
                runner.config, base_dir=runner.registry.base_dir, origin=runner.origin,
                generators=runner.registry.generators)

        fingerprint = identity(self)
        if fingerprint != identity(source):
            return False
        source.raw_design_fingerprint = fingerprint
        self._shared_design_source = source
        self._precompiled_fingerprint = fingerprint
        return True

    def training_candidate_count(self) -> int:
        if self.config.validation.get("training_window") is None:
            return len(self.supervised_origins)
        times = pd.date_range(self.builder.history_start, self.origin, freq=self.config.problem.freq)
        candidates = times[minimum_history_rows(self.config) - 1:]
        return int((_temporal_forecast_ends(self.config.problem, self.config.validation, candidates) <= self.origin).sum())

    def _prepare_training(self) -> None:
        config, registry, origin = self.config, self.registry, self.origin
        precompiled_payload = (
            self._shared_design_source.raw_design_payload()
            if self._shared_design_source is not None else self._precompiled_payload
        )
        precompiled_fingerprint = self._precompiled_fingerprint
        design_started = perf_counter()
        design_status = "compiled"
        self.design_stage_wall_seconds = {"compile": 0.0, "share": 0.0}
        if precompiled_payload is not None:
            if not precompiled_fingerprint:
                raise ValueError(
                    "precompiled_fingerprint is required with precompiled_payload"
                )
            self._restore_raw_design(precompiled_payload)
            self.design_stage_wall_seconds["share"] = perf_counter() - design_started
            self.design_shared = True
            self.raw_design_fingerprint = precompiled_fingerprint
            design_status = "memory_hit"
        else:
            compile_started = perf_counter()
            self._compile_supervised_arrays()
            self.design_stage_wall_seconds["compile"] = perf_counter() - compile_started
        self.workload = build_runtime_workload(
            config,
            training_rows=len(self._Y_all),
            feature_count=len(self.builder.feature_schema),
            design_bytes=(
                sum(design.retained_bytes if isinstance(design, IndexedDesign) else design.nbytes
                    for design in self._X_all)
                + self._Y_all.nbytes
            ),
            series_count=self.builder.n_series,
        )
        # 保留父调度器已授予的线程份额；实际设计仍接受同预算资源校验。
        plan_runtime_execution(
            config, self.workload, budget=self.resource_budget,
            base_dir=registry.base_dir, feature_schema=self.builder.feature_schema,
        )
        self.design_status = design_status
        self.training_compile = self.builder.training_compile_summary()
        if self._native_cls is not None:
            self.training_compile.update(mode="native_history", origin_count=len(self.supervised_origins))
        if self.design_shared:
            self.training_compile["mode"] = "memory_shared"
        logger.info(
            "[TrainingCompile] mode=%s origins=%s calls=%s reasons=%s wall=%s",
            self.training_compile["mode"],
            self.training_compile["origin_count"],
            self.training_compile["call_count"],
            self.training_compile["reason_codes"],
            self.training_compile["stage_wall_seconds"],
        )
        self.stage_wall_seconds["raw_design"] = perf_counter() - design_started

    def _compile_supervised_arrays(self) -> None:
        if self._native_cls is not None:
            # 原生历史模型不编译监督特征；标签窗直接由原始历史切出。
            origins = self.supervised_origins
            history = self.builder.target_history(self.origin)
            positions = history.forecast_times.get_indexer(origins)
            windows = np.lib.stride_tricks.sliding_window_view(
                history.values[0, :, 0], self.config.problem.horizon,
            )
            self.Y_all = windows[positions + 1, :, None]
            self.X_all = tuple(
                np.empty((len(origins), 0)) for _ in self.builder.plan.call_coordinates
            )
            self.supervised_sample_origins = origins
            self.supervised_sample_series_ids = (self.builder.series_ids[0],) * len(origins)
            self.builder.feature_schema = self.builder.categorical_schema = ()
            return
        # 构造阶段已完成 schema 探测和资源准入，此处仅执行实际设计。
        (
            self.X_all,
            self.Y_all,
            self.supervised_origins,
            self.supervised_sample_origins,
            self.supervised_sample_series_ids,
        ) = _supervised_arrays(self.builder, self.origin)

    def _raw_design_payload(self) -> dict[str, Any]:
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
        self.prepare_training()
        return self._raw_design_payload()

    def _restore_raw_design(self, payload: Mapping[str, Any]) -> None:
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
                f"shared raw design payload is missing fields: {missing}"
            )
        X_all = tuple(
            design if isinstance(design, IndexedDesign) else np.asarray(design)
            for design in payload["X_all"]
        )
        Y_all = np.asarray(payload["Y_all"])
        sample_origins = tuple(
            cast(pd.Timestamp, pd.Timestamp(value))
            for value in payload["supervised_sample_origins"]
        )
        if any(len(design) != len(Y_all) for design in X_all):
            raise ValueError("shared raw design X/Y sample counts differ")
        if len(sample_origins) != len(Y_all):
            raise ValueError("shared raw design origin/Y sample counts differ")
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
    def geometry(self) -> backtest_geometry.TimeGeometry:
        return backtest_geometry.TimeGeometry(
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
        arrays = [self._Y_all] if self._Y_all is not None else []
        for design in self._X_all or ():
            if isinstance(design, IndexedDesign):
                arrays.extend(values for values, _ in design.columns)
                if design.rows is not None:
                    arrays.append(design.rows)
            else:
                arrays.append(design)
        model_indices = self.builder.plan.model_indices
        return {
            "design_storage": {
                "logical_design_bytes": self.workload.design_bytes,
                "retained_array_bytes": retained_array_bytes(arrays),
                "largest_model_rows": self.workload.training_rows * max(
                    model_indices.count(i) for i in set(model_indices)
                ),
            },
            "performance_profile": resolve_performance_profile(
                self.config, self.workload, budget=self.resource_budget,
                base_dir=self.registry.base_dir, feature_schema=self.feature_schema,
                execution_plan=self.execution_plan,
            ),
            "workload": self.workload.payload(),
            "budget": self.resource_budget.payload(),
            "execution_plan": self.execution_plan.payload(),
            "design_preparation": {
                "status": self.design_status,
                "fingerprint": self.raw_design_fingerprint,
                "stage_wall_seconds": dict(self.design_stage_wall_seconds),
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

    def backtest_windows(self) -> tuple[backtest_geometry.RollingOriginFold, ...]:
        if self.config.validation.get("training_window") is not None:
            return temporal_backtest_windows(
                config=self.config,
                registry=self.builder.registry,
                offset=self.builder.offset,
                origin=self.origin,
                minimum_history=minimum_history_rows(self.config),
            )
        if self.config.validation.get("train_history_steps") is not None:
            return raw_history_backtest_windows(
                config=self.config,
                registry=self.builder.registry,
                offset=self.builder.offset,
                origin=self.origin,
                minimum_history=minimum_history_rows(self.config),
            )
        if isinstance(self.config.validation.backtest, CalendarMonthBacktestSpec):
            return ()
        return rolling_backtest_windows(
            self.supervised_origins,
            offset=self.builder.offset,
            horizon=self.config.problem.horizon,
            backtest=self.config.validation.backtest,
            schedule_origin=(self.origin if self.config.validation.get("schedule_mode") == "intraday" else None),
        )

    def for_backtest_window(self, window: BacktestWindow) -> CanonicalBaseModelRunner:
        """每折独立设计与审计状态；原点和 W 同时进入设计/checkpoint 身份。"""
        if not has_bounded_history(self.config.validation):
            return self
        runner = CanonicalBaseModelRunner(
            self.config, self.registry, window.origin,
            resource_budget=self.resource_budget,
            checkpoint_root=self.checkpoint_root,
        )
        # 窗口外层负责并行预算，不在各折重新扩大资源份额。
        runner.execution_plan = self.execution_plan
        return runner

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
        if self.raw_design_fingerprint is None:
            raise ValueError(
                "fold transform cache requires a raw design fingerprint"
            )
        key = fold_transform_fingerprint(
            self.config,
            raw_design_fingerprint=self.raw_design_fingerprint,
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
        fit_started = perf_counter()
        self.prepare_training()
        candidate_count = len(train_indices)
        if self.config.validation.get("training_window") is not None:
            candidate_count = self.training_candidate_count()
        if self.config.validation.get("training_window") is None:
            train_indices = select_training_origins(
                self.supervised_origins, train_indices,
                self.config.validation.get("training", {}).get("origin_sampling"),
                freq=self.config.problem.freq,
            )
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
        if self.config.features.transformations.get("seasonal_baseline") is not None:
            baselines = np.concatenate([self.builder.seasonal_baseline(self.supervised_origins[index])
                                        for index in train_indices], axis=0)
            Y_train = Y_train - baselines
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
        train_sample_weight = resolve_training_sample_weight(
            self.config,
            training_origins,
            history_cutoff=training_history_cutoff,
            sample_weight_capable=MODEL_CATALOG[
                self.config.estimator.model_type
            ].sample_weight,
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
        # 原生模型只消费本折完整原始历史；监督数组仅用于共用调度，不用于原生序列拟合。
        native_cls = _native_history_cls(self.config.estimator.model_type)
        if native_cls is not None:
            if train_indices != tuple(range(len(self.supervised_origins))):
                raise ValueError(
                    f"{self.config.estimator.model_type} requires the complete configured "
                    "raw-history fold"
                )
            history = self.builder.target_history(self.origin)
            artifact = native_cls(dict(self.config.estimator.params)).fit_history(
                pd.Series(history.values[0, :, 0], index=history.forecast_times),
                as_of=self.origin, freq=self.config.problem.freq,
            )
            return feature_scaler, target_transform, X_train_transformed, Y_train_transformed, artifact
        # 监督特征选择：挂在训练 fit 边界——每个回测窗口与最终训练
        # 各自重拟合，只消费当前训练窗 (X, Y)，无泄漏；选中集写入
        # artifact.feature_schema，预测端按同名子集对齐。
        X_train_transformed, feature_schema = self._apply_feature_selection(
            X_train_transformed, Y_train_transformed
        )
        preparation_seconds = perf_counter() - fit_started
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
                sample_weight=train_sample_weight,
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
                sample_weight=train_sample_weight,
            )
        if isinstance(artifact, CanonicalStrategyArtifact):
            artifact = replace(artifact, training_workload={
                **artifact.training_workload,
                "candidate_origins": candidate_count,
                "selected_origins": len(train_indices),
                "first_training_origin": self.supervised_origins[train_indices[0]].isoformat(),
                "last_training_origin": self.supervised_origins[train_indices[-1]].isoformat(),
                "stage_wall_seconds": {
                    **artifact.training_workload["stage_wall_seconds"],
                    "training_preparation": preparation_seconds,
                    "fit_total": perf_counter() - fit_started,
                },
            })
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
        *,
        data_phase: str = "historical",
    ) -> tuple[tuple[np.ndarray, ...], Any]:
        return _forecast_designs_with_scaler(
            self.builder,
            origin,
            feature_scaler,
            target_transform,
            data_phase=data_phase,
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
        native_cls = _native_history_cls(self.config.estimator.model_type)
        if native_cls is not None:
            evidence = artifact.execution_evidence()
            expected = pd.date_range(pd.Timestamp(evidence["history_end"]),
                                     periods=self.config.problem.horizon + 1, freq=self.config.problem.freq)[1:]
            if not forecast_times.equals(expected):
                raise ValueError(
                    f"{self.config.estimator.model_type} prediction must immediately follow "
                    "the fitted as-of origin"
                )
            return PointForecastTensor(artifact.forecast(len(forecast_times))[None, :, None],
                                       self.series_ids, forecast_times, self.config.problem.targets)
        base_design = designs[0]
        # 特征选择对齐：artifact.feature_schema 是训练期选中集，预测端把
        # 全 schema 设计矩阵按同名子集对齐（provider 输出同为全 schema 宽）。
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
        restored = _restore_prediction(raw, target_transform)
        if self.config.features.transformations.get("seasonal_baseline") is not None:
            baseline = self.builder.seasonal_baseline(forecast_times[0] - self.builder.offset)
            return PointForecastTensor(restored.values + baseline, restored.series_ids,
                                       restored.forecast_times, restored.targets)
        return restored

    def actual(
        self,
        origin_index: int,
        forecast_times: pd.DatetimeIndex,
    ) -> PointForecastTensor:
        if has_bounded_history(self.config.validation):
            request = self.builder.request(self.origin, target_access="supervised_labels")
            information = self.registry.materialize(request, source_names=self.builder.target_source_names)
            values, _ = self.builder.labels_from_information_set(request, information)
            return PointForecastTensor(values, self.series_ids, forecast_times, self.config.problem.targets)
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
        return _temporal_forecast_times(self.config.problem, self.config.validation, origin)

    def target_history(self, origin: pd.Timestamp) -> PointForecastTensor:
        """origin 的 as-of 完整目标历史（预测图参照段；ensemble 协议面）。"""
        return self.builder.target_history(origin)

    def final_bundle_inputs(self) -> tuple[
        CanonicalFeatureScaler,
        CanonicalTargetTransform,
        tuple[np.ndarray, ...],
        np.ndarray,
    ]:
        """Fit final transforms under the same explicit window as backtesting.

        声明了 ``validation.training.sample_weight`` 时，最终拟合权重在
        ``fit_final`` 内按同一窗口的 origins/cutoff 重算（与折路径同一
        接线函数），保证 final fit 与回测共享加权语义。
        """
        preparation_started = perf_counter()
        self.prepare_training()
        self._final_training_workload = None
        if self.config.validation.get("train_history_steps") is not None:
            raise ValueError("train_history_steps currently requires backtest-only; final fit/bundle unsupported")
        backtest = self.config.validation.backtest
        if self.config.validation.get("training_window") is not None:
            origin_indices = tuple(range(len(self.supervised_origins)))
        elif isinstance(backtest, (FixedStepBacktestSpec, SlidingWindowBacktestSpec)):
            # sliding 与 fixed 同为固定长度训练窗口，final fit 语义一致
            first_origin_index = max(
                0,
                len(self.supervised_origins) - backtest.train_window_steps,
            )
            origin_indices = tuple(
                range(first_origin_index, len(self.supervised_origins))
            )
        elif isinstance(backtest, ExpandingWindowBacktestSpec):
            raise ValueError(
                "expanding_window backtest is backtest-only; final fit/bundle unsupported"
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
        candidate_count = len(origin_indices)
        if self.config.validation.get("training_window") is not None:
            candidate_count = self.training_candidate_count()
        if self.config.validation.get("training_window") is None:
            origin_indices = select_training_origins(
                self.supervised_origins, origin_indices,
                self.config.validation.get("training", {}).get("origin_sampling"),
                freq=self.config.problem.freq,
            )
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
        self._final_sample_weight = resolve_training_sample_weight(
            self.config,
            sample_origins,
            history_cutoff=history_cutoff,
            sample_weight_capable=MODEL_CATALOG[
                self.config.estimator.model_type
            ].sample_weight,
        )
        self._final_training_workload = {
            "candidate_origins": candidate_count,
            "selected_origins": len(origin_indices),
            "first_training_origin": self.supervised_origins[origin_indices[0]].isoformat(),
            "last_training_origin": self.supervised_origins[origin_indices[-1]].isoformat(),
            "stage_wall_seconds": {"training_preparation": perf_counter() - preparation_started},
        }
        return feature_scaler, target_transform, X_all_transformed, Y_all_transformed

    @runtime_checkpoint_errors
    def fit_final(
        self,
        X_transformed: tuple[np.ndarray, ...],
        Y_transformed: np.ndarray,
    ) -> tuple[Any, Any, Any]:
        """Train the final artifact and return (trainer, artifact, capabilities).

        声明了训练加权时，权重来自 ``final_bundle_inputs`` 在同一显式
        窗口上的计算（先于本方法调用）；未经 ``final_bundle_inputs``
        直接调用且配置声明了加权时报错，防止静默丢失加权语义。
        """
        if self.config.validation.get("train_history_steps") is not None:
            raise ValueError("train_history_steps currently requires backtest-only; final fit unsupported")
        mode = self._mode()
        fit_started = perf_counter()
        final_sample_weight = getattr(self, "_final_sample_weight", None)
        if (final_sample_weight is None
                and training_sample_weight_spec(self.config) is not None):
            raise ValueError(
                "validation.training.sample_weight requires "
                "final_bundle_inputs() before fit_final() so weights are "
                "computed on the same explicit window"
            )
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
                sample_weight=final_sample_weight,
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
                sample_weight=final_sample_weight,
            )
        if isinstance(artifact, CanonicalStrategyArtifact) and self._final_training_workload is not None:
            final_workload = self._final_training_workload
            artifact = replace(artifact, training_workload={
                **artifact.training_workload,
                **final_workload,
                "stage_wall_seconds": {
                    **artifact.training_workload["stage_wall_seconds"],
                    **final_workload["stage_wall_seconds"],
                    "fit_total": perf_counter() - fit_started + final_workload["stage_wall_seconds"]["training_preparation"],
                },
            })
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
        ``source_lineage``、``calibration_state``。ensemble 成员路径不传 extras，
        走默认产物元数据。
        """
        extras = dict(extras or {})
        if self.config.validation.get("train_history_steps") is not None:
            raise ValueError("train_history_steps currently requires backtest-only; bundle unsupported")
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
        """监督特征选择（features.selection）。

        挂在训练 fit 边界：每个回测窗口与最终训练各自重拟合选择器，
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
        *,
        backtest_only: bool = False,
    ) -> CanonicalRuntimeResult | BacktestRuntimeResult:
        """Execute with process-level BLAS/OpenMP limits set before any pool."""
        with threadpool_limits(limits=self.execution_plan.model_threads):
            if backtest_only:
                return run_lifecycle(self, output_root, backtest_only=True)
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
        native_cls = _native_history_cls(self.config.estimator.model_type)
        if native_cls is not None:
            models = [{
                "wrapper": native_cls.__name__,
                **artifact.execution_evidence(),
            }]
        return json_evidence({
            "status": "recorded" if models else "unavailable",
            "reason": None if models else "no_supported_model_wrapper_found",
            **self._execution_evidence_context,
            "target_transform_window": target_transform.fit_window_metadata,
            "fitted_models": models,
            "training_workload": getattr(artifact, "training_workload", {}),
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
    backtest_only: bool = False,
) -> CanonicalRuntimeResult | BacktestRuntimeResult:
    """Execute a canonical config, optionally stopping after rolling backtest."""
    if not isinstance(config, ForecastConfigSpec):
        raise TypeError("config must be a ForecastConfigSpec")
    if config.strategy is None:
        raise ValueError("run_canonical_config requires a strategy or ensemble")
    if config.validation.get("train_history_steps") is not None and not backtest_only:
        raise ValueError("train_history_steps currently requires backtest-only")
    # builtin generators（chinese_holiday）默认可用；调用方同名注入时覆盖。
    merged_generators: dict[str, Any] = {**BUILTIN_GENERATORS, **(generators or {})}
    registry = SourceRegistry(config.data, Path.cwd(), generators=merged_generators)
    origin = resolve_origin(registry, config.validation.get("forecast_origin"))
    runner = CanonicalBaseModelRunner(
        config,
        registry,
        origin,
        checkpoint_root=checkpoint_root,
    )
    if backtest_only:
        return runner.run(output_root, backtest_only=True)
    return runner.run(output_root)


__all__ = [
    "BacktestRuntimeResult",
    "CanonicalBaseModelRunner",
    "CanonicalRuntimeResult",
    "persist_model_bundle",
    "run_canonical_config",
]
