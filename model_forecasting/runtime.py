"""Executable canonical runtime without legacy configuration translation."""

from __future__ import annotations

import json
from functools import cached_property
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from model_testing import validation
from data_loading import (
    BUILTIN_GENERATORS,
    SourceRegistry,
)
from model_training.estimators import make_model_factory
from feature_engineering import CompiledFeatures
from feature_engineering import cache as compiled_cache
from model_evaluation.marginal import evaluate_marginal_distribution
from model_evaluation.point import (
    evaluate_point_forecasts,
    resolve_aggregate_weighting,
)
from feature_engineering.selection import (
    CanonicalFeatureSelector,
    normalize_feature_selection,
    selected_indices_for_artifact,
)
from utils.log_util import logger
from model_forecasting.results import (
    backtest_tensors_to_long,
    write_backtest_results,
    write_forecast_results,
)
from pandas.tseries.frequencies import to_offset
from probabilistic.calibration import ConformalCalibrationTracker
from forecasting_core.probabilistic_spec import probabilistic_spec_from_mapping
from forecasting_core.specs import (
    CalendarMonthBacktestSpec,
    FixedStepBacktestSpec,
    ForecastConfigSpec,
    TargetAdapter,
)
from model_testing.backtest import (
    actual_tensor,
    positive_validation_int,
    resolve_origin,
    seasonal_naive_tensor,
)
from model_forecasting.persistence import (
    build_strategy_model_bundle,
    persist_model_bundle,
)
from model_training.strategies import CanonicalStrategyArtifact
from forecasting_core.tensors import PointForecastTensor
from model_forecasting.transforms import CanonicalFeatureScaler, CanonicalTargetTransform
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
from model_forecasting.backtest_runtime import run_calendar_month_backtest
from model_forecasting.evidence import collect_model_evidence, dependency_versions, json_evidence
from model_forecasting.lifecycle import write_run_state
from model_forecasting.design import (
    _BacktestWindow,
    _RegistryDesignBuilder,
    _actual_at_origin,
    _label_end,
    _rolling_backtest_windows,
    _sample_indices,
    _supervised_arrays,
)
from model_forecasting.fit_service import (
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


@dataclass(frozen=True, slots=True)
class CanonicalRuntimeResult:
    run_dir: Path
    model_dir: Path
    test_dir: Path
    forecast_dir: Path
    fingerprint: str
    bundle: ForecastModelBundle


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _proof_payload(compiled_items: tuple[CompiledFeatures, ...]) -> list[dict[str, Any]]:
    payload = []
    seen = set()
    for compiled in compiled_items:
        for proof in compiled.visibility_proof:
            item = asdict(proof)
            key = tuple(item.items())
            if key in seen:
                continue
            seen.add(key)
            item["target_time"] = proof.target_time.isoformat()
            item["source_time"] = (
                proof.source_time.isoformat() if proof.source_time is not None else None
            )
            item["forecast_origin"] = proof.forecast_origin.isoformat()
            item["available_at"] = (
                proof.available_at.isoformat() if proof.available_at is not None else None
            )
            payload.append(item)
    return payload


def _holdout_proof_summary(
    compiled_items: tuple[CompiledFeatures, ...],
) -> dict[str, Any]:
    group_by = (
        "forecast_origin",
        "feature_name",
        "source_name",
        "role",
        "provider",
    )
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    total_lookups = 0
    for compiled in compiled_items:
        for proof in compiled.visibility_proof:
            total_lookups += 1
            key = (
                proof.forecast_origin,
                proof.feature_name,
                proof.source_name,
                proof.role,
                proof.provider,
            )
            item = grouped.get(key)
            if item is None:
                item = {
                    "forecast_origin": proof.forecast_origin,
                    "feature_name": proof.feature_name,
                    "source_name": proof.source_name,
                    "role": proof.role,
                    "provider": proof.provider,
                    "lookup_count": 0,
                    "horizon_step_min": proof.horizon_step,
                    "horizon_step_max": proof.horizon_step,
                    "target_time_min": proof.target_time,
                    "target_time_max": proof.target_time,
                    "source_time_count": 0,
                    "source_time_min": proof.source_time,
                    "source_time_max": proof.source_time,
                    "available_at_min": proof.available_at,
                    "available_at_max": proof.available_at,
                }
                grouped[key] = item
            item["lookup_count"] += 1
            item["horizon_step_min"] = min(
                item["horizon_step_min"], proof.horizon_step
            )
            item["horizon_step_max"] = max(
                item["horizon_step_max"], proof.horizon_step
            )
            item["target_time_min"] = min(item["target_time_min"], proof.target_time)
            item["target_time_max"] = max(item["target_time_max"], proof.target_time)
            item["available_at_min"] = min(
                item["available_at_min"], proof.available_at
            )
            item["available_at_max"] = max(
                item["available_at_max"], proof.available_at
            )
            if proof.source_time is not None:
                item["source_time_count"] += 1
                item["source_time_min"] = (
                    proof.source_time
                    if item["source_time_min"] is None
                    else min(item["source_time_min"], proof.source_time)
                )
                item["source_time_max"] = (
                    proof.source_time
                    if item["source_time_max"] is None
                    else max(item["source_time_max"], proof.source_time)
                )

    serialized = []
    timestamp_fields = (
        "forecast_origin",
        "target_time_min",
        "target_time_max",
        "source_time_min",
        "source_time_max",
        "available_at_min",
        "available_at_max",
    )
    for grouped_item in grouped.values():
        item = dict(grouped_item)
        for field in timestamp_fields:
            value = item[field]
            item[field] = value.isoformat() if value is not None else None
        serialized.append(item)
    return {
        "schema_version": 1,
        "group_by": list(group_by),
        "total_lookups": total_lookups,
        "group_count": len(serialized),
        "groups": serialized,
    }


def _source_lineage_payload(
    compiled_items: tuple[CompiledFeatures, ...],
) -> list[dict[str, Any]]:
    payload = []
    seen = set()
    for compiled in compiled_items:
        for lineage in compiled.source_lineage:
            item = {
                "source": lineage.source_name,
                "path_version": lineage.path_version,
                "path": lineage.path,
                "availability": lineage.availability_policy,
                "includes_target_labels": lineage.includes_target_labels,
            }
            key = tuple(item.items())
            if key not in seen:
                seen.add(key)
                payload.append(item)
    return payload


def _compiled_lineage(
    feature_schema: tuple[str, ...],
    proof_payload: list[dict[str, Any]],
    config: ForecastConfigSpec,
) -> tuple[tuple[dict[str, Any], ...], dict[str, list[str]]]:
    by_feature = {}
    for item in proof_payload:
        by_feature.setdefault(item["feature_name"], item)
    feature_lineage = []
    availability_summary: dict[str, list[str]] = {}
    source_availability = {
        source.name: (
            source.availability.value if source.availability is not None else "static"
        )
        for source in config.data.sources
    }
    for feature in feature_schema:
        if feature in config.problem.series_id_cols:
            feature_lineage.append(
                {
                    "feature": feature,
                    "source": "series_identity",
                    "role": "key",
                    "source_time": None,
                    "provider": None,
                    "availability": "static",
                }
            )
            availability_summary.setdefault("static", []).append(feature)
            continue
        proof = by_feature[feature]
        availability = cast(
            str,
            "known_future"
            if proof["source_name"] == "calendar"
            else source_availability.get(proof["source_name"], proof["role"]),
        )
        feature_lineage.append(
            {
                "feature": feature,
                "source": proof["source_name"],
                "role": proof["role"],
                "source_time": proof["source_time"],
                "provider": proof["provider"],
                "availability": availability,
            }
        )
        availability_summary.setdefault(availability, []).append(feature)
    return (
        tuple(feature_lineage),
        {key: availability_summary[key] for key in sorted(availability_summary)},
    )


def _log_provider_usage(audits: Any) -> None:
    """A3 可观测（2026-09-01）：聚合 VisibilityProof 的特征取值来源。

    只打日志、不进结果 schema。回答「某配置深 horizon 输入中合成值
    （provider，如 persistence 冻结）占比多少」——persistence 依赖度
    消融与配置审计的直接证据。
    """
    if not audits:
        return
    total = 0
    provider_hits = 0
    by_provider: dict[str, int] = {}
    by_feature: dict[str, int] = {}
    for compiled in audits:
        for proof in compiled.visibility_proof:
            if proof.role not in {"target", "observed_past"}:
                continue
            total += 1
            if proof.provider is not None:
                provider_hits += 1
                by_provider[proof.provider] = by_provider.get(proof.provider, 0) + 1
                by_feature[proof.feature_name] = by_feature.get(proof.feature_name, 0) + 1
    if total == 0:
        return
    if provider_hits == 0:
        logger.info(
            "[ProviderUsage] lag lookups: history 100%% (%d lookups, no provider involved)",
            total,
        )
        return
    top_features = sorted(by_feature.items(), key=lambda kv: -kv[1])[:5]
    top_text = ", ".join(f"{name} x{n}" for name, n in top_features)
    logger.warning(
        "[ProviderUsage] lag lookups: history %d%% / provider %d%% (%d/%d); "
        "providers=%s; top=%s",
        round(100.0 * (total - provider_hits) / total),
        round(100.0 * provider_hits / total),
        provider_hits,
        total,
        by_provider or {},
        top_text,
    )


def _output_paths(
    config: ForecastConfigSpec,
    fingerprint: str,
    output_root: str | Path | None,
) -> tuple[Path, Path, Path, Path]:
    output = config.output
    identity = output.get("identity", {})
    scenario = str(
        identity.get("scenario_subpath", output.get("scenario_subpath", "canonical"))
        if isinstance(identity, Mapping)
        else output.get("scenario_subpath", "canonical")
    ).strip("/") or "canonical"
    result_identity = config.result_identity()
    if output_root is not None:
        run_dir = Path(output_root) / scenario / result_identity
        return (
            run_dir,
            run_dir / "pretrained_models",
            run_dir / "results_test",
            run_dir / "results_forecast",
        )
    directories = output.get("directories", {})
    if isinstance(directories, Mapping) and {
        "checkpoints",
        "tests",
        "forecast",
    }.issubset(directories):
        model_dir = Path(str(directories["checkpoints"])) / scenario / result_identity
        test_dir = Path(str(directories["tests"])) / scenario / result_identity
        forecast_dir = Path(str(directories["forecast"])) / scenario / result_identity
        return forecast_dir, model_dir, test_dir, forecast_dir
    legacy_directories = {
        "checkpoints_dir": "model",
        "test_results_dir": "test",
        "pred_results_dir": "forecast",
    }
    if set(legacy_directories).issubset(output):
        model_dir = Path(str(output["checkpoints_dir"])) / scenario / result_identity
        test_dir = Path(str(output["test_results_dir"])) / scenario / result_identity
        forecast_dir = Path(str(output["pred_results_dir"])) / scenario / result_identity
        return forecast_dir, model_dir, test_dir, forecast_dir
    legacy_scenario = str(output.get("scenario_subpath", scenario)).strip("/") or scenario
    legacy_root = Path(str(output.get("results_root", "results")))
    run_dir = legacy_root / legacy_scenario / result_identity
    return (
        run_dir,
        run_dir / "pretrained_models",
        run_dir / "results_test",
        run_dir / "results_forecast",
    )


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
        self.builder = _RegistryDesignBuilder(config, registry)
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
    ) -> ForecastModelBundle:
        """Build a self-contained schema-2 bundle from a completed final fit."""
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
        bundle = build_strategy_model_bundle(
            bundle_builder,
            bundle_artifact,
            feature_scaler=feature_scaler,
            target_transform=target_transform,
            input_schema={
                "columns": list(self.feature_schema),
                "panel": {
                    "series_id_cols": list(self.config.problem.series_id_cols),
                    "known_series_ids": [
                        list(value) if isinstance(value, tuple) else value
                        for value in self.series_ids
                    ],
                    "unknown_series_policy": "raise",
                },
            },
            series_ids=self.series_ids,
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
            return self._run(output_root)

    def run_prelimited(
        self,
        output_root: str | Path | None = None,
    ) -> CanonicalRuntimeResult:
        """Execute under a parent-owned process-level threadpool limit."""
        return self._run(output_root)

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

    def _run(
        self,
        output_root: str | Path | None = None,
    ) -> CanonicalRuntimeResult:
        fingerprint = self.config.fingerprint()
        _, model_dir, _, _ = _output_paths(self.config, fingerprint, output_root)
        write_run_state(model_dir, fingerprint, "running")
        try:
            result = self._execute_lifecycle(output_root)
            write_run_state(model_dir, fingerprint, "completed")
            return result
        except BaseException:
            write_run_state(model_dir, fingerprint, "failed")
            raise

    def _execute_lifecycle(
        self,
        output_root: str | Path | None = None,
    ) -> CanonicalRuntimeResult:
        """Full single-model lifecycle: rolling backtest, final fit, persist."""
        builder = self.builder
        config = self.config
        origin = self.origin
        mode = self._mode()
        backtest_started = perf_counter()

        fingerprint = config.fingerprint()
        run_dir, model_dir, test_dir, forecast_dir = _output_paths(
            config,
            fingerprint,
            output_root,
        )
        aggregate_weights = resolve_aggregate_weighting(
            config.problem.targets,
            config.validation.get("aggregate_weighting"),
        )
        cv_frames = []
        score_frames = []
        prob_score_frames = []
        eval_mask_config = (
            config.validation.get("eval_mask")
            if isinstance(config.validation.get("eval_mask"), Mapping)
            else None
        )
        holdout_audits = []
        holdout_execution_evidence = []
        # CQR（2026-09-01 激活）：quantile 且声明 calibration 时启用 as-of
        # 校准追踪器；回测逐折 apply-before-collect，final 用全部合格历史折。
        calibration_tracker = None
        if mode == "quantile":
            prob_spec = probabilistic_spec_from_mapping(
                config.probabilistic.canonical_payload()
            )
            if prob_spec.calibration is not None:
                calibration_tracker = ConformalCalibrationTracker(
                    prob_spec,
                    freq_offset=to_offset(str(config.problem.freq)),
                )
        calibration_audits: list[dict[str, Any]] = []
        backtest_windows = self.backtest_windows()
        window_workers = min(
            self.execution_plan.window_workers,
            max(1, len(backtest_windows)),
        )
        parallel_fits = None
        if window_workers > 1 and backtest_windows:
            if CanonicalTargetTransform.from_config(config).is_identity:
                target_histories = (None,) * len(backtest_windows)
            else:
                target_histories = tuple(
                    builder.target_history(
                        max(
                            _label_end(
                                builder,
                                cast(pd.Timestamp, self.supervised_origins[index]),
                            )
                            for index in backtest_window.train_indices
                        )
                    )
                    for backtest_window in backtest_windows
                )

            def fit_window(item):
                backtest_window, target_history = item
                return self.fit(
                    backtest_window.train_indices,
                    target_history=target_history,
                    force_serial=True,
                )

            with ThreadPoolExecutor(max_workers=window_workers) as executor:
                parallel_fits = tuple(
                    executor.map(
                        fit_window,
                        zip(backtest_windows, target_histories),
                    )
                )

        for window_index, backtest_window in enumerate(backtest_windows):
            (
                holdout_feature_scaler,
                holdout_target_transform,
                _X_train_transformed,
                _Y_train_transformed,
                holdout_artifact,
            ) = (
                parallel_fits[window_index]
                if parallel_fits is not None
                else self.fit(backtest_window.train_indices)
            )

            builder.reset_audit()
            holdout_designs, holdout_provider = self.forecast_designs(
                backtest_window.origin,
                holdout_feature_scaler,
                holdout_target_transform,
            )
            holdout_times = self.forecast_times(backtest_window.origin)
            holdout_prediction = self.predict(
                holdout_artifact,
                holdout_designs,
                holdout_provider,
                holdout_times,
                holdout_target_transform,
            )
            actual = self.actual(
                backtest_window.origin_index,
                holdout_times,
            )
            seasonal_naive = self.seasonal_naive(
                backtest_window.origin,
                holdout_times,
            )
            holdout_point = (
                holdout_prediction
                if isinstance(holdout_prediction, PointForecastTensor)
                else holdout_prediction.point
            )
            fold_frame = backtest_tensors_to_long(
                actual,
                holdout_prediction,
                window=backtest_window.window,
            )
            if calibration_tracker is not None:
                # apply-before-collect：当前折只消费严格更早折的校准池
                fold_frame, calibration_audit = (
                    calibration_tracker.apply_to_frame(
                        fold_frame,
                        forecast_origin=backtest_window.origin,
                    )
                )
                calibration_audits.append(
                    {"window": backtest_window.window, **calibration_audit}
                )
            cv_frames.append(fold_frame)
            if calibration_tracker is not None:
                calibration_tracker.collect_from_frame(
                    fold_frame,
                    forecast_origin=backtest_window.origin,
                    window=backtest_window.window,
                )
            score_frames.append(
                evaluate_point_forecasts(
                    actual,
                    holdout_point,
                    aggregate_weighting=aggregate_weights,
                    seasonal_naive=seasonal_naive,
                    window=backtest_window.window,
                    eval_mask=eval_mask_config,
                )
            )
            # 概率评估接线（2026-08-30）：quantile 模式逐窗产出 pinball/central 区间
            # 指标；eval_mask 配置直传，掩码口径与点评估同一 payload（同一业务口径）。
            if not isinstance(holdout_prediction, PointForecastTensor):
                prob_score_frames.append(
                    evaluate_marginal_distribution(
                        actual,
                        holdout_prediction,
                        eval_mask=eval_mask_config,
                        window=backtest_window.window,
                    )
                )
            holdout_audits.extend(builder.audit)
            holdout_execution_evidence.append({
                "window": backtest_window.window,
                "origin": backtest_window.origin.isoformat(),
                **self.execution_evidence(holdout_artifact, holdout_target_transform),
            })
        holdout_audit = tuple(holdout_audits)
        _log_provider_usage(holdout_audit)
        if cv_frames:
            backtest = config.validation.backtest
            if not isinstance(backtest, FixedStepBacktestSpec):
                raise TypeError("fixed backtest results require FixedStepBacktestSpec")
            windows = self.backtest_windows()
            holdout_metadata = {
                **windows[-1].metadata,
                "mode": "fixed_steps",
                "history_steps": backtest.history_steps,
                "train_window_steps": backtest.train_window_steps,
                "fold_count": backtest.fold_count,
                "stride_steps": backtest.stride_steps,
                "windows": [window.metadata for window in windows],
                "execution_evidence": holdout_execution_evidence,
            }
            if calibration_audits:
                holdout_metadata["calibration"] = calibration_audits
            write_backtest_results(
                test_dir,
                pd.concat(cv_frames, ignore_index=True),
                pd.concat(score_frames, ignore_index=True),
                aggregate_weighting=aggregate_weights,
                metadata={
                    "backtest": holdout_metadata,
                    "runtime_resources": self.runtime_resources_payload(),
                },
                probabilistic_scores_df=(
                    pd.concat(prob_score_frames, ignore_index=True)
                    if prob_score_frames
                    else None
                ),
            )
        else:
            holdout_metadata, calibration_tracker = run_calendar_month_backtest(
                config, self.registry, self, test_dir,
                runner_factory=self.calendar_runner_factory,
            )

        self.stage_wall_seconds["backtest"] = perf_counter() - backtest_started
        final_fit_started = perf_counter()
        (
            final_feature_scaler,
            final_target_transform,
            X_all_transformed,
            Y_all_transformed,
        ) = self.final_bundle_inputs()
        final_trainer, final_artifact, final_capabilities = self.fit_final(
            X_all_transformed,
            Y_all_transformed,
        )
        self.stage_wall_seconds["final_fit"] = perf_counter() - final_fit_started
        forecast_started = perf_counter()

        builder.reset_audit()
        final_designs, final_provider = self.forecast_designs(
            origin,
            final_feature_scaler,
            final_target_transform,
        )
        forecast_times = self.forecast_times(origin)
        forecast = self.predict(
            final_artifact,
            final_designs,
            final_provider,
            forecast_times,
            final_target_transform,
        )
        final_audit = builder.audit
        # CQR final（2026-09-01 激活）：修正量入 bundle（部署自包含），
        # predict_pi<coverage>_* 列写入 prediction.csv；修正量由全部满足
        # as-of 的历史折池化计算。
        calibration_state = None
        forecast_extra_columns = None
        if calibration_tracker is not None:
            final_correction, final_calibration_audit = (
                calibration_tracker.final_correction(origin)
            )
            calibration_state = {
                "method": "cqr",
                **final_calibration_audit,
            }
            if final_correction.status == "applied":
                if not isinstance(forecast, MarginalForecastDistribution):
                    raise TypeError("CQR requires a marginal forecast distribution")
                if final_correction.correction is None:
                    raise ValueError("applied CQR correction must be finite")
                lower_col, upper_col = calibration_tracker.pi_columns
                lower, upper = calibration_tracker.correction_bounds(
                    forecast.quantiles.values,
                    forecast.quantiles.levels,
                    final_correction.correction,
                )
                forecast_extra_columns = {
                    lower_col: lower.reshape(-1),
                    upper_col: upper.reshape(-1),
                }
            logger.info(
                "[CQR] final calibration: status=%s correction=%s "
                "windows=%s scores=%s",
                final_calibration_audit["status"],
                final_calibration_audit["correction"],
                final_calibration_audit["selected_windows"],
                final_calibration_audit["selected_scores"],
            )
        visibility_proof = _proof_payload(final_audit)
        holdout_visibility_proof = _holdout_proof_summary(holdout_audit)
        source_lineage = _source_lineage_payload(final_audit)
        feature_lineage, availability_summary = _compiled_lineage(
            builder.feature_schema,
            visibility_proof,
            config,
        )
        if mode == "point":
            bundle_builder = final_trainer
            bundle_artifact = final_artifact
        else:
            point_level = float(config.probabilistic.get("point_quantile", 0.5))
            bundle_builder = CanonicalTrainer(
                config,
                estimator_factory=make_model_factory(
                    config.estimator.model_type,
                    config.estimator.params,
                    feature_names=builder.feature_schema,
                    quantile=point_level,
                ),
                capabilities=final_capabilities,
                feature_schema=builder.feature_schema,
            )
            bundle_artifact = final_artifact.artifacts_by_level[point_level]
        bundle = build_strategy_model_bundle(
            bundle_builder,
            bundle_artifact,
            feature_scaler=final_feature_scaler,
            target_transform=final_target_transform,
            input_schema={
                "columns": list(builder.feature_schema),
                "panel": {
                    "series_id_cols": list(config.problem.series_id_cols),
                    "known_series_ids": [
                        list(value) if isinstance(value, tuple) else value
                        for value in builder.series_ids
                    ],
                    "unknown_series_policy": str(
                        builder._training_scope_validation().get(
                            "unknown_series_policy",
                            "raise",
                        )
                    ).lower(),
                },
                "availability_summary": availability_summary,
                "visibility_proof": visibility_proof,
            },
            feature_lineage=feature_lineage,
            source_lineage=source_lineage,
            series_ids=builder.series_ids,
            calibration_state=calibration_state,
        )
        if mode == "quantile":
            bundle.model = final_artifact

        persist_model_bundle(bundle, model_dir)

        # 预测图历史参照段（2026-09-02）：as-of origin 的 target_history 末段
        # （5×horizon 步）随预测图绘制；历史段时间轴 <= origin，与预测段不重叠。
        try:
            forecast_history = builder.target_history(origin)
            history_steps = max(1, int(config.problem.horizon) * 5)
            forecast_history = PointForecastTensor(
                values=forecast_history.values[:, -history_steps:, :],
                series_ids=forecast_history.series_ids,
                forecast_times=forecast_history.forecast_times[-history_steps:],
                targets=forecast_history.targets,
            )
        except (ValueError, KeyError) as exc:
            logger.warning(
                "[forecast plot] target history unavailable, plot without "
                "history reference: %s",
                exc,
            )
            forecast_history = None
        write_forecast_results(
            forecast_dir,
            forecast,
            extra_columns=forecast_extra_columns,
            history=forecast_history,
        )
        self.stage_wall_seconds["forecast_persist"] = perf_counter() - forecast_started
        self.stage_wall_seconds["total"] = perf_counter() - self.lifecycle_started
        run_evidence = {
            **self.execution_evidence(final_artifact, final_target_transform),
            "raw_design_provenance": compiled_cache.raw_design_provenance(
                config, base_dir=self.registry._base_dir, origin=origin,
                generators=self.registry._generators,
            ),

        }
        _write_json(
            forecast_dir / "resolved_config.json",
            {
                **config.canonical_payload(),
                "config_fingerprint": fingerprint,
                "runtime": {
                    "lifecycle_schema_version": 1,
                    "forecast_origin": origin.isoformat(),
                    "run_evidence": run_evidence,
                    "resources": self.runtime_resources_payload(),
                    "series_order": [
                        list(value) if isinstance(value, tuple) else value
                        for value in builder.series_ids
                    ],
                    "feature_schema": list(builder.feature_schema),
                    "availability_summary": availability_summary,
                    "source_lineage": source_lineage,
                    "visibility_proof": visibility_proof,
                    "holdout_visibility_proof": holdout_visibility_proof,
                    "calibration": calibration_state,
                    "capability_probe": {
                        "native_multioutput_probed": (
                            config.estimator.target_adapter is TargetAdapter.NATIVE
                        ),
                        "resolved": final_capabilities.canonical_payload(),
                    },
                    "strategy": {
                        "model_count": builder.plan.model_count,
                        "dependencies": [
                            [
                                {
                                    "target": coordinate.target,
                                    "horizon_step": coordinate.horizon_step,
                                }
                                for coordinate in dependencies
                            ]
                            for dependencies in builder.plan.dependencies
                        ],
                    },
                    "holdout": holdout_metadata,
                },
            },
        )
        metadata_path = test_dir / "result_metadata.json"
        if metadata_path.exists():
            result_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            result_metadata["runtime_resources"] = self.runtime_resources_payload()
            result_metadata["run_evidence"] = run_evidence
            _write_json(metadata_path, result_metadata)
        return CanonicalRuntimeResult(
            run_dir=run_dir,
            model_dir=model_dir,
            test_dir=test_dir,
            forecast_dir=forecast_dir,
            fingerprint=fingerprint,
            bundle=bundle,
        )

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
