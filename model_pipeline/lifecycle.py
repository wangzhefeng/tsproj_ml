"""单模型阶段编排、完成状态与产物元数据；不构造具体 runner。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, cast

from feature_engineering import CompiledFeatures
from feature_engineering import cache as compiled_cache
from forecasting_core.artifacts import ForecastModelBundle, MarginalForecastDistribution
from forecasting_core.specs import ForecastConfigSpec, TargetAdapter
from forecasting_core.tensors import PointForecastTensor
from model_forecasting.persistence import persist_model_bundle
from model_forecasting.results import write_forecast_results
from model_pipeline.run_state import write_run_state
from model_testing.calendar_month import run_calendar_month_backtest
from model_testing.fixed_step import run_fixed_step_backtest
from utils.log_util import logger

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


def run_lifecycle(runner: Any, output_root: str | Path | None = None) -> CanonicalRuntimeResult:
    fingerprint = runner.config.fingerprint()
    _, model_dir, _, _ = _output_paths(runner.config, fingerprint, output_root)
    write_run_state(model_dir, fingerprint, "running")
    try:
        result = execute_lifecycle(runner, output_root)
        write_run_state(model_dir, fingerprint, "completed")
        return result
    except BaseException:
        write_run_state(model_dir, fingerprint, "failed")
        raise


def execute_lifecycle(runner: Any, output_root: str | Path | None = None) -> CanonicalRuntimeResult:
    """Full single-model lifecycle: rolling backtest, final fit, persist."""
    builder = runner.builder
    config = runner.config
    origin = runner.origin
    mode = runner._mode()
    backtest_started = perf_counter()

    fingerprint = config.fingerprint()
    run_dir, model_dir, test_dir, forecast_dir = _output_paths(
        config,
        fingerprint,
        output_root,
    )
    holdout_metadata, calibration_tracker, holdout_audit = run_fixed_step_backtest(
        runner, test_dir, mode=mode,
    )
    if holdout_metadata is None:
        holdout_metadata, calibration_tracker = run_calendar_month_backtest(
            config, runner.registry, runner, test_dir,
            runner_factory=runner.calendar_runner_factory,
        )

    runner.stage_wall_seconds["backtest"] = perf_counter() - backtest_started
    final_fit_started = perf_counter()
    (
        final_feature_scaler,
        final_target_transform,
        X_all_transformed,
        Y_all_transformed,
    ) = runner.final_bundle_inputs()
    final_trainer, final_artifact, final_capabilities = runner.fit_final(
        X_all_transformed,
        Y_all_transformed,
    )
    runner.stage_wall_seconds["final_fit"] = perf_counter() - final_fit_started
    forecast_started = perf_counter()

    builder.reset_audit()
    final_designs, final_provider = runner.forecast_designs(
        origin,
        final_feature_scaler,
        final_target_transform,
    )
    forecast_times = runner.forecast_times(origin)
    forecast = runner.predict(
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
    # bundle 构建唯一入口（R6b）：消除与方法版 build_final_bundle 的漂移双份；
    # 单模型路径经 extras 传入完整产物元数据，ensemble 成员路径不传走默认。
    bundle = runner.build_final_bundle(
        final_feature_scaler,
        final_target_transform,
        final_trainer,
        final_artifact,
        final_capabilities,
        extras={
            "unknown_series_policy": str(
                builder._training_scope_validation().get(
                    "unknown_series_policy",
                    "raise",
                )
            ).lower(),
            "availability_summary": availability_summary,
            "visibility_proof": visibility_proof,
            "feature_lineage": feature_lineage,
            "source_lineage": source_lineage,
            "calibration_state": calibration_state,
        },
    )

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
    runner.stage_wall_seconds["forecast_persist"] = perf_counter() - forecast_started
    runner.stage_wall_seconds["total"] = perf_counter() - runner.lifecycle_started
    run_evidence = {
        **runner.execution_evidence(final_artifact, final_target_transform),
        "raw_design_provenance": compiled_cache.raw_design_provenance(
            config, base_dir=runner.registry._base_dir, origin=origin,
            generators=runner.registry._generators,
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
                "resources": runner.runtime_resources_payload(),
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
        result_metadata["runtime_resources"] = runner.runtime_resources_payload()
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
