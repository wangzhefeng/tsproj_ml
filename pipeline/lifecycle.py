"""单模型阶段编排、完成状态与产物元数据；不构造具体 runner。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from feature_engineering import design_identity
from forecasting_core.bundle import ForecastModelBundle
from forecasting_core.probability.distribution import MarginalForecastDistribution
from forecasting_core.specs import (
    CalendarMonthBacktestSpec,
    ExpandingWindowBacktestSpec,
    ForecastConfigSpec,
    SlidingWindowBacktestSpec,
    TargetAdapter,
)
from forecasting_core.tensors.point import PointForecastTensor
from probabilistic.residual import ResidualCalibrationTracker, apply_residual_state
from model_predicting.artifacts.evidence_assembly import (
    compiled_lineage,
    holdout_proof_summary,
    proof_payload,
    source_lineage_payload,
)
from model_predicting.artifacts.persistence import persist_model_bundle
from model_predicting.artifacts.results import write_forecast_results
from pipeline.run_state import write_run_state
from model_testing.loops.calendar_month import run_calendar_month_backtest
from model_testing.loops.expanding_window import run_expanding_window_backtest
from model_testing.loops.fixed_step import run_fixed_step_backtest
from model_testing.loops.sliding_window import run_sliding_window_backtest
from utils.log_util import logger

@dataclass(frozen=True, slots=True)
class CanonicalRuntimeResult:
    run_dir: Path
    model_dir: Path
    test_dir: Path
    forecast_dir: Path
    fingerprint: str
    bundle: ForecastModelBundle


@dataclass(frozen=True, slots=True)
class BacktestRuntimeResult:
    """仅回测产物；不包含可部署模型或最终预测。"""

    test_dir: Path
    fingerprint: str


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
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
    legacy_scenario = str(output.get("scenario_subpath", scenario)).strip("/") or scenario
    legacy_root = Path(str(output.get("results_root", "results")))
    run_dir = legacy_root / legacy_scenario / result_identity
    return (
        run_dir,
        run_dir / "pretrained_models",
        run_dir / "results_test",
        run_dir / "results_forecast",
    )


def run_lifecycle(
    runner: Any, output_root: str | Path | None = None, *, backtest_only: bool = False,
) -> CanonicalRuntimeResult | BacktestRuntimeResult:
    fingerprint = runner.config.fingerprint()
    if not backtest_only and runner.config.validation.get("train_history_steps") is not None:
        raise ValueError("train_history_steps currently requires backtest-only")
    _, model_dir, test_dir, _ = _output_paths(runner.config, fingerprint, output_root)
    # 回测状态与完整模型状态隔离，不能用回测 completed 宣称 bundle 可部署。
    state_dir = test_dir / "backtest_only" if backtest_only else model_dir
    write_run_state(state_dir, fingerprint, "running")
    try:
        result = (
            execute_lifecycle(runner, output_root, backtest_only=True)
            if backtest_only else execute_lifecycle(runner, output_root)
        )
        write_run_state(state_dir, fingerprint, "completed")
        return result
    except BaseException:
        write_run_state(state_dir, fingerprint, "failed")
        raise


def execute_lifecycle(
    runner: Any, output_root: str | Path | None = None, *, backtest_only: bool = False,
) -> CanonicalRuntimeResult | BacktestRuntimeResult:
    """复用回测阶段；仅回测模式在 final fit 之前退出。"""
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
    if backtest_only:
        test_dir = test_dir / "backtest_only"
    # 回测几何显式分派：按 validation.backtest 的 spec 类型选执行器，
    # 不依赖 fixed_step 返回 None 的隐式回退协议；各执行器产物合同一致。
    backtest_spec = config.validation.backtest
    if backtest_spec is None:
        raise ValueError("canonical lifecycle requires a configured backtest geometry")
    if isinstance(backtest_spec, CalendarMonthBacktestSpec):
        holdout_metadata, calibration_tracker = run_calendar_month_backtest(
            config, runner.registry, runner, test_dir,
            runner_factory=runner.calendar_runner_factory,
        )
        holdout_audit = ()
    elif isinstance(backtest_spec, SlidingWindowBacktestSpec):
        holdout_metadata, calibration_tracker, holdout_audit = run_sliding_window_backtest(
            runner, test_dir, mode=mode,
        )
    elif isinstance(backtest_spec, ExpandingWindowBacktestSpec):
        holdout_metadata, calibration_tracker, holdout_audit = run_expanding_window_backtest(
            runner, test_dir, mode=mode,
        )
    else:
        holdout_metadata, calibration_tracker, holdout_audit = run_fixed_step_backtest(
            runner, test_dir, mode=mode,
        )

    runner.stage_wall_seconds["backtest"] = perf_counter() - backtest_started
    if backtest_only:
        _write_json(test_dir / "backtest_metadata.json", {
            "execution_mode": "backtest_only",
            "result_method": config.result_method(),
            "config_fingerprint": fingerprint,
            "holdout": holdout_metadata,
            "backtest_wall_seconds": runner.stage_wall_seconds["backtest"],
        })
        return BacktestRuntimeResult(test_dir=test_dir, fingerprint=fingerprint)
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
    # CQR final：修正量写入 bundle（部署自包含），predict_pi<coverage>_*
    # 列写入 prediction.csv；修正量由全部满足 as-of 的历史折池化计算。
    calibration_state = None
    forecast_extra_columns = None
    if isinstance(calibration_tracker, ResidualCalibrationTracker):
        if not isinstance(forecast, PointForecastTensor):
            raise TypeError("absolute_residual requires point forecast")
        calibration_state = calibration_tracker.state(forecast, forecast_origin=origin)
        forecast = apply_residual_state(forecast, calibration_state)
    elif calibration_tracker is not None:
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
    visibility_proof = proof_payload(final_audit)
    holdout_visibility_proof = holdout_proof_summary(holdout_audit)
    source_lineage = source_lineage_payload(final_audit)
    feature_lineage, availability_summary = compiled_lineage(
        builder.feature_schema,
        visibility_proof,
        config,
    )
    # bundle 构建唯一入口：lifecycle 经 runner.build_final_bundle 组装，
    # 不持有平行实现；单模型路径经 extras 传入完整产物元数据，
    # ensemble 成员路径不传走默认。
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

    # 预测图历史参照段：as-of origin 的 target_history 末段（5×horizon 步）
    # 随预测图绘制；历史段时间轴 <= origin，与预测段不重叠。
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
        "raw_design_provenance": design_identity.raw_design_provenance(
            config, base_dir=runner.registry.base_dir, origin=origin,
            generators=runner.registry.generators,
        ),

    }
    _write_json(
        forecast_dir / "resolved_config.json",
        {
            **config.canonical_payload(),
            "config_fingerprint": fingerprint,
            "runtime": {
                "result_method": config.result_method(),
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
