"""融合器独立外层时间回测；内层 OOF 仅消费外层原点前的完整标签。"""
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import pandas as pd

from forecasting_core.artifacts import MarginalForecastDistribution
from forecasting_core.probabilistic_spec import probabilistic_spec_from_mapping
from model_ensemble.artifacts import OOFPredictionArtifact, method_artifact_audit_payload
from model_ensemble.contracts import BaseModelRunner, member_execution_evidence
from model_ensemble.inference.forecast import build_ensemble_forecast
from model_ensemble.training.oof import actual_for_folds, shared_origin_timeline, member_training_indices
from model_ensemble.inference.predictor import combine_members
from model_ensemble.configuration.specs import EnsembleConfigSpec
from model_ensemble.training.trainer import fit_fusion_artifact
from model_evaluation.point import evaluate_point_forecasts
from model_evaluation.marginal import evaluate_marginal_distribution
from model_testing.contracts.geometry import rolling_origin_folds
from model_testing.artifacts.tensor_frames import backtest_tensors_to_long


@dataclass
class EnsembleBacktestResult:
    frame: pd.DataFrame
    point_scores: pd.DataFrame
    probabilistic_scores: pd.DataFrame | None
    metadata: dict[str, Any]
    calibration_tracker: Any = None


def run_outer_backtest(
    config: EnsembleConfigSpec,
    runners: Mapping[str, BaseModelRunner],
    *,
    get_oof: Callable[[pd.Timestamp], OOFPredictionArtifact],
    calibration_tracker: Any = None,
) -> EnsembleBacktestResult:
    """每折独立拟合融合器和成员，不把 meta-train 分数当成测试分数。"""
    first = next(iter(runners.values()))
    timeline = shared_origin_timeline(runners)
    validation = config.validation
    train_steps = int(validation.get("train_window_steps", config.oof.train_window_steps))
    stride = int(validation.get("stride_steps", config.problem.horizon))
    folds = rolling_origin_folds(
        timeline.supervised_origins, timeline.geometry,
        history_steps=validation.get("history_steps"), train_window_steps=train_steps,
        fold_count=int(validation.get("fold_count", 1)), stride_steps=stride,
        schedule_origin=first.origin if validation.get("schedule_mode") == "intraday" else None,
    )
    probability = probabilistic_spec_from_mapping(config.probabilistic.canonical_payload())
    frames, point_scores, probabilistic_scores, summaries = [], [], [], []
    for fold in folds:
        inner = get_oof(fold.origin)
        positions = {origin.isoformat(): index for index, origin in enumerate(first.supervised_origins)}
        actual_inner = actual_for_folds(first, [
            {"origin_index": positions[item["origin"]], "origin": pd.Timestamp(item["origin"])}
            for item in inner.folds
        ])
        artifact = fit_fusion_artifact(config, inner, actual_inner, origin=fold.origin)
        times = first.forecast_times(fold.origin)
        values, evidence = {}, {}
        for name, runner in runners.items():
            history_steps = validation.get("history_steps", len(timeline.supervised_origins))
            earliest = timeline.supervised_origins[max(0, len(timeline.supervised_origins) - history_steps)]
            # 顶层窗口选择外层评估几何；成员拟合窗口与独立 final fit 同源。
            member_steps = int(runner.config.validation["train_window_steps"])
            indices = member_training_indices(runner, fold.origin, member_steps, earliest_origin=earliest)
            try:
                scaler, transform, _X, _Y, model = runner.fit(indices)
                designs, provider = runner.forecast_designs(fold.origin, scaler, transform)
                prediction = runner.predict(model, designs, provider, times, transform)
                values[name] = prediction.quantiles.values if isinstance(prediction, MarginalForecastDistribution) else prediction.values
                evidence[name] = member_execution_evidence(runner, model, transform)
            except Exception as exc:
                raise RuntimeError(f"outer fold={fold.window} member={name!r} failed") from exc
        prediction = build_ensemble_forecast(
            combine_members(artifact, values), probability=probability, series_ids=first.series_ids,
            forecast_times=times, targets=config.problem.targets, method_name=config.method.name,
        )
        actual = first.actual(first.supervised_origins.index(fold.origin), times)
        point = prediction.point if isinstance(prediction, MarginalForecastDistribution) else prediction
        frame = backtest_tensors_to_long(actual, prediction, window=fold.window)
        calibration = None
        if calibration_tracker is not None:
            frame, calibration = calibration_tracker.apply_to_frame(frame, forecast_origin=fold.origin)
            calibration_tracker.collect_from_frame(frame, forecast_origin=fold.origin, window=fold.window)
        frames.append(frame)
        score_context = {}
        if validation.get("seasonal_naive_lag") is not None:
            history = first.target_history(fold.origin)
            score_context = {
                "seasonal_naive": first.seasonal_naive(fold.origin, times, history=history),
                "insample_history": history, "naive_lag": validation["seasonal_naive_lag"],
            }
        point_scores.append(evaluate_point_forecasts(
            actual, point, window=fold.window, eval_mask=validation.get("eval_mask"),
            aggregate_weighting=validation.get("aggregate_weighting"),
            **score_context,
        ))
        if isinstance(prediction, MarginalForecastDistribution):
            probabilistic_scores.append(evaluate_marginal_distribution(
                actual, prediction, window=fold.window, eval_mask=validation.get("eval_mask"),
            ))
        summaries.append({
            **fold.metadata, "inner_folds": list(inner.folds), "inner_oof_fingerprint": inner.oof_fingerprint,
            "method_artifact": method_artifact_audit_payload(artifact.method_artifact),
            "member_execution_evidence": evidence, "calibration": calibration,
        })
    return EnsembleBacktestResult(
        pd.concat(frames, ignore_index=True), pd.concat(point_scores, ignore_index=True),
        pd.concat(probabilistic_scores, ignore_index=True) if probabilistic_scores else None,
        {"evaluation_role": "outer_holdout", "folds": summaries, "allow_overlapping_windows": stride < config.problem.horizon},
        calibration_tracker,
    )
