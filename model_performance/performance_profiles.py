"""One adopted CatBoost profile, not a registry or an automatic tuner.

The constants below were recovered from OPT-017's six archived runs. Runtime
never reads /tmp evidence or unpickles a benchmark model. A reference is a claim
of applicability, not an override: missing context or mismatches must RAISE.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from typing import Any

from feature_engineering.cache import file_sha256
from forecasting_core.runtime_resources import (
    RuntimeExecutionPlan, RuntimeResourceBudget, RuntimeWorkload,
)
from forecasting_core.specs import ForecastConfigSpec
from models.wrappers.catboost import CatBoostModel

CATBOOST_PROFILE_REF = "opt017-catboost-short-a-pointwise-v1"

# 5072 是 raw supervised origins；1424 是每折/final 的训练窗，非 5072 行 fit。
# single_model_horizon 在 fit 时展开 H=16 倍；此处与 runtime workload 同口径。
_SIGNATURE = {
    "semantic_sha256_at_5fold": "6f00fab66ebee786d074f83ffdcafc65e2de949a9a11072fb076bce1118a6243",
    "workload": {
        "strategy": "direct", "direct_layout": "single_model_horizon",
        "horizon": 16, "chunk_length": None, "target_count": 1,
        "series_count": 1, "quantile_count": 1, "member_count": 1,
        "model_group_count": 1, "logical_output_count": 1,
        "scalar_estimator_count": 1, "physical_fit_task_count": 1,
        "parallel_output_task_count": 1, "training_rows": 5072,
        "feature_count": 29, "design_bytes": 19476480,
        "adapter": "independent", "estimator": "catboost", "dynamic_horizons": [],
    },
    "feature_schema": [
        "value__lag_16", "value__lag_96", "value__lag_192", "value__lag_672",
        "dt_hour", "dt_minute", "dt_day", "dt_day_of_week", "dt_week_of_year",
        "dt_month", "dt_days_in_month", "dt_quarter", "dt_day_of_year", "dt_year",
        "forecast_horizon_idx",
        "value_rolling_mean_16", "value_rolling_std_16", "value_rolling_min_16",
        "value_rolling_max_16", "value_rolling_mean_96", "value_rolling_std_96",
        "value_rolling_min_96", "value_rolling_max_96", "value_rolling_mean_672",
        "value_rolling_std_672", "value_rolling_min_672", "value_rolling_max_672",
        "value_expanding_mean", "value_expanding_std",
    ],
    "source_sha256": {
        "target_history:history_path": "33a7ebf593684ddfe9bdf9865cb93596f4bda93073371e9ecdaa544c7d0ccaa4",
    },
    "estimator_versions": {"catboost": "1.2.10"},
    "estimator_defaults": {
        "loss_function": "MAE", "eval_metric": "MAE", "iterations": 300,
        "learning_rate": 0.05, "depth": 6, "verbose": False,
        "random_seed": 42, "allow_writing_files": False,
    },
    "training_scope": "local",
    "train_window_steps": 1424,
}

_PROVENANCE = {
    "evidence_root": "/tmp/tsproj-opt017-benchmark",
    "protocol": "three interleaved A/B rounds; isolated single-config 5-fold",
    "benchmark_fold_count": 5,
    "adopted_physical_fold_count": 31,
    "formal_benchmark_verified": False,
    "benchmark_raw_design_fingerprint": "94d60fc7b0af68397e3a74b5413b2566552298bd0ced47259debe859f6932fd7",
    "data_binding_evidence": "current source+environment rehash reproduced archived raw-design fingerprint",
    "estimator_version_evidence": "all six model.pkl CatBoost metadata: tags/v1.2.10, commit b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084",
    "metric_sha256": {
        "cat-a-1": "4d434dfdbf7f8cee3d4e97a694ca595dfbea4d49960c0e647aae5b67dc8f95a8",
        "cat-a-2": "aa0f49391c2bc5c583327d1dde3a03e33e663e4e0516ba568b02ac86ccfc0bd6",
        "cat-a-3": "c293ea5a208f21712b5761973639fe191564b55de810c2e6a92a186390afab17",
        "cat-b-1": "a206c5cb6a94482fbcfd866d1bc99ef2c7e077551e35ef49a23297b7a4543e20",
        "cat-b-2": "d7b2026259450116380a233743baa69a807b0e53e4de2c6c12c70c4c82304a9c",
        "cat-b-3": "d506a5fe1048de91e59d679a56d9cf217dae456aeb60d3cbce018b420fd4f14d",
    },
    "candidate_resolved_sha256": "d6a12bd7655c49f37d0daef14dfffd71fb11fce7e16ed1d981ae70b1ef0df600",
    "candidate_bundle_sha256": "7b1bd732585bf0412e71a8aa71aeb6719f5f02bc6990566f53cb3a31465a64e8",
    "artifact_identity": "direct-catboost-local-k1-6f00fab66ebe",
    "physical_cores": 8,
    "logical_cores": 8,
    "parent_concurrency": 1,
    "peak_rss_bytes": 451936256,
    "measurements": {
        "baseline_a": {
            "wall_median_seconds": 44.19090516702272,
            "wall_p95_seconds": 44.196782916691156,
            "cpu_median_seconds": 40.453993000000004,
            "mean_cores_median": 0.9190065912423371,
            "peak_rss_bytes": 417988608,
            "observed_available_memory_bytes": [4143464448, 4259840000, 4203872256],
        },
        "candidate_b": {
            "wall_median_seconds": 35.546239541028626,
            "wall_p95_seconds": 35.6665548422141,
            "cpu_median_seconds": 54.738060999999995,
            "mean_cores_median": 1.5467995126892942,
            "peak_rss_bytes": 451936256,
            "observed_available_memory_bytes": [4193861632, 4075388928, 4417060864],
        },
    },
    "memory_boundary": "measured 5-fold RSS lower admission bound, not a hard cap or a 31-fold memory guarantee",
    "swap_increase_bytes": 0,
    "csv_sha256": {
        "backtest": "bb11b9bd3e628e889f41fdaeb35fc3942270cf3272d83d92ebe7198862e91f27",
        "scores": "dc0746006b1419ad5ca109b018ca1bd07d5c4d3841138703d1a8c1caa6938d50",
        "forecast": "0f7491a47853d765817157c6bdd857f988ed7ccdf50e1ae088f64eecb9157037",
    },
    # Archived metrics contain this nonzero reload delta; do not call it exact.
    "reload_max_abs_diff": 1.8189894035458565e-12,
}


def adopted_catboost_profile() -> dict[str, Any]:
    """Return an isolated audit copy of the sole adopted evidence contract."""
    return deepcopy({
        "profile_ref": CATBOOST_PROFILE_REF,
        "signature": _SIGNATURE,
        "provenance": _PROVENANCE,
        "controls": {"model_thread_count": 8},
    })


def build_performance_signature(
    config: ForecastConfigSpec,
    workload: RuntimeWorkload,
    *,
    base_dir: str | Path,
    feature_schema: Sequence[str],
) -> dict[str, Any]:
    """Bind actual design shape/schema, config semantics, source bytes and version.

    Fold count is checked separately: only the original 5 and adopted physical
    31 are allowed. No other config semantics are dropped for applicability.
    """
    semantic = config.semantic_payload()
    validation = config.validation.semantic_payload()
    validation["fold_count"] = 5
    semantic["validation"] = validation
    encoded = json.dumps(semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    topology = workload.payload()
    topology["dynamic_horizons"] = list(workload.dynamic_horizons)
    topology.pop("fold_count")
    topology.pop("total_fit_task_count")
    hashes = {}
    for source in config.data.sources:
        for role in ("history_path", "backtest_path", "future_path"):
            path = getattr(source, role)
            if path is not None:
                hashes[f"{source.name}:{role}"] = file_sha256(Path(base_dir) / path)
    defaults = dict(CatBoostModel.DEFAULT_PARAMS)
    defaults.pop("thread_count", None)
    return {
        "semantic_sha256_at_5fold": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "workload": topology,
        "feature_schema": list(feature_schema),
        "source_sha256": hashes,
        "estimator_versions": {"catboost": version("catboost")},
        "estimator_defaults": defaults,
        "training_scope": config.problem.training_scope,
        "train_window_steps": config.validation.get("train_window_steps"),
    }


def resolve_performance_profile(
    config: ForecastConfigSpec,
    workload: RuntimeWorkload,
    *,
    budget: RuntimeResourceBudget,
    base_dir: str | Path | None = None,
    feature_schema: Sequence[str] | None = None,
    execution_plan: RuntimeExecutionPlan | None = None,
) -> dict[str, Any]:
    """Resolve provenance before planning; a reference fails closed on mismatch.

    No ref means an explicit override (or automatic default), never a validated
    benchmark claim. Call with the actual raw-design feature schema and project
    base_dir, not an estimated workload or the expected evidence schema.
    Metadata callers also pass the final execution_plan, so a parent scheduler
    cannot replace it and leave behind a stale benchmark applicability claim.
    """
    performance = dict(config.validation.performance or {})
    ref = performance.pop("profile_ref", None)
    if ref is None:
        return {
            "kind": "explicit_performance_override" if performance else "automatic_default",
            "profile_ref": None, "controls": performance,
            "formal_benchmark_verified": False,
        }
    if ref != CATBOOST_PROFILE_REF:
        raise ValueError(f"unknown performance profile_ref: {ref!r}")
    adopted_controls = {
        "model_thread_count": 8, "window_parallel_workers": 1,
        "multi_output_n_jobs": 1, "quantile_parallel_workers": 1,
    }
    for field, value in performance.items():
        if field in {"total_thread_limit", "memory_limit_bytes"}:
            continue
        if field not in adopted_controls or value != adopted_controls[field]:
            raise ValueError(f"performance profile control conflict: {field}; remove profile_ref for an explicit override")
    thread_limit = min(budget.total_thread_limit, performance.get("total_thread_limit", budget.total_thread_limit))
    memory_limit = min(budget.memory_limit_bytes, performance.get("memory_limit_bytes", budget.memory_limit_bytes))
    if (
        budget.physical_cores != 8 or budget.logical_cores != 8
        or budget.parent_concurrency != 1 or thread_limit < 8
        or memory_limit < _PROVENANCE["peak_rss_bytes"]
        or not budget.require_determinism
    ):
        raise ValueError("performance profile budget mismatch: requires exclusive 8-core budget and at least measured peak RSS")
    if base_dir is None or feature_schema is None:
        raise ValueError("performance profile requires actual base_dir and feature_schema")
    signature = build_performance_signature(
        config, workload, base_dir=base_dir, feature_schema=feature_schema,
    )
    mismatches = [key for key, expected in _SIGNATURE.items() if signature[key] != expected]
    configured_folds = config.validation.backtest.fold_count if config.validation.backtest else None
    if workload.fold_count not in (5, 31) or workload.fold_count != configured_folds:
        mismatches.append("fold_count")
    if workload.total_fit_task_count != workload.physical_fit_task_count * (workload.fold_count + 1):
        mismatches.append("total_fit_task_count")
    if mismatches:
        raise ValueError(f"performance profile {ref!r} signature mismatch: {', '.join(mismatches)}")
    if execution_plan is not None:
        expected_plan = {
            "selected_axis": "serial", "config_workers": 1, "member_workers": 1,
            "window_workers": 1, "quantile_workers": 1, "output_workers": 1,
            "model_threads": 8, "available_threads": 8, "budget_product": 8,
        }
        if any(getattr(execution_plan, key) != value for key, value in expected_plan.items()):
            raise ValueError("performance profile execution plan mismatch after scheduling")
    return {
        "kind": "adopted_benchmark_profile",
        "profile_ref": ref,
        "controls": {"model_thread_count": 8, **performance},
        "signature": signature,
        "provenance": deepcopy(_PROVENANCE),
        "benchmark_fold_count": 5,
        "actual_fold_count": workload.fold_count,
        "benchmark_geometry_match": workload.fold_count == 5,
        "formal_benchmark_verified": False,
        "resolved_execution_plan": execution_plan.payload() if execution_plan is not None else None,
    }
