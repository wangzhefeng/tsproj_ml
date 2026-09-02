"""Calendar-month backtest orchestration for the canonical runtime."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from data_loading import SourceRegistry
from forecasting_core.artifacts import MarginalForecastDistribution
from forecasting_core.specs import (
    CalendarMonthBacktestSpec,
    ForecastConfigSpec,
)
from forecasting_core.tensors import PointForecastTensor
from model_evaluation.marginal import evaluate_marginal_distribution
from model_evaluation.point import (
    evaluate_point_forecasts,
    resolve_aggregate_weighting,
)
from model_forecasting.results import backtest_tensors_to_long, write_backtest_results
from model_forecasting.persistence import persist_model_bundle
from model_testing import validation
from pandas.tseries.frequencies import to_offset
from probabilistic.calibration import (
    ConformalCalibrationTracker,
    attach_cqr_interval_columns,
)
from forecasting_core.probabilistic_spec import probabilistic_spec_from_mapping


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def overwrite_calendar_month_backtest(
    config: ForecastConfigSpec,
    registry: SourceRegistry,
    final_runner: Any,
    result: Any,
    *,
    runner_factory: Any,
) -> None:
    backtest = config.validation.backtest
    if not isinstance(backtest, CalendarMonthBacktestSpec):
        raise TypeError(
            "calendar-month backtest requires CalendarMonthBacktestSpec"
        )
    folds = validation.calendar_month_folds(
        final_runner.builder.target_history_times(final_runner.origin),
        train_window_days=backtest.train_window_days,
        fold_count=backtest.fold_count,
        stride_months=backtest.stride_months,
    )
    if not folds:
        raise ValueError("calendar-month runtime requires at least one complete fold")

    aggregate_weights = resolve_aggregate_weighting(
        config.problem.targets,
        config.validation.get("aggregate_weighting"),
    )
    eval_mask_config = (
        config.validation.get("eval_mask")
        if isinstance(config.validation.get("eval_mask"), Mapping)
        else None
    )
    cv_frames = []
    score_frames = []
    probabilistic_frames = []
    fold_metadata = []
    # CQR（2026-09-01 激活）：calendar-month 与 fixed-step 同一追踪器语义，
    # 逐折 apply-before-collect；final 修正量消费全部合格历史折。
    calibration_tracker = None
    if str(config.probabilistic.get("mode", "point")) == "quantile":
        prob_spec = probabilistic_spec_from_mapping(
            config.probabilistic.canonical_payload()
        )
        if prob_spec.calibration is not None:
            calibration_tracker = ConformalCalibrationTracker(
                prob_spec,
                freq_offset=to_offset(str(config.problem.freq)),
            )
    calibration_audits: list[dict[str, Any]] = []
    runners_by_horizon: dict[int, Any] = {}
    fold_contexts = []
    raw_history_times = final_runner.builder.target_history_times(final_runner.origin)
    dynamic_history_steps = len(final_runner.supervised_origins)
    for fold in folds:
        dynamic_problem = replace(config.problem, horizon=fold.horizon)
        dynamic_validation = {
            key: value
            for key, value in dict(config.validation).items()
            if key not in {"train_window_days", "stride_months"}
        }
        dynamic_validation.update(
            {
                "horizon_mode": "fixed_steps",
                "history_steps": dynamic_history_steps,
                "train_window_steps": min(
                    backtest.train_window_days,
                    dynamic_history_steps - 1,
                ),
                "fold_count": 1,
                "stride_steps": fold.horizon,
                "seasonal_naive_lag": max(fold.horizon, 1),
            }
        )
        dynamic_config = replace(
            config,
            problem=dynamic_problem,
            validation=dynamic_validation,
        )
        runner = runners_by_horizon.get(fold.horizon)
        if runner is None:
            runner = runner_factory(
                dynamic_config,
                registry,
                final_runner.origin,
            )
            runners_by_horizon[fold.horizon] = runner
        try:
            origin_index = runner.supervised_origins.index(fold.origin)
        except ValueError as exc:
            raise ValueError(
                f"calendar-month origin {fold.origin} is not a supervised origin"
            ) from exc
        holdout_label_start = runner.geometry.label_start(fold.origin)
        train_start_time = pd.Timestamp(raw_history_times[fold.train_indices[0]])
        train_indices = tuple(
            index
            for index in range(origin_index)
            if runner.supervised_origins[index] >= train_start_time
            and runner.geometry.label_end(runner.supervised_origins[index])
            < holdout_label_start
        )
        if not train_indices:
            raise ValueError(
                f"calendar-month fold {fold.window} has no safe supervised samples"
            )
        training_label_end_max = max(
            runner.geometry.label_end(runner.supervised_origins[index])
            for index in train_indices
        )
        fold_contexts.append(
            (
                fold,
                runner,
                origin_index,
                train_indices,
                training_label_end_max,
                runner.builder.target_history(training_label_end_max),
            )
        )

    performance = config.validation.get("performance", {})
    if not isinstance(performance, Mapping):
        raise TypeError("validation.performance must be a mapping")
    configured_workers = performance.get("window_parallel_workers", 1)
    if (
        isinstance(configured_workers, bool)
        or not isinstance(configured_workers, int)
        or configured_workers <= 0
    ):
        raise ValueError(
            "validation.performance.window_parallel_workers must be positive"
        )
    window_workers = min(configured_workers, len(fold_contexts))

    def fit_context(context):
        return context[1].fit(
            context[3],
            target_history=context[5],
            force_serial=window_workers > 1,
        )

    if window_workers > 1:
        with ThreadPoolExecutor(max_workers=window_workers) as executor:
            fold_fits = tuple(executor.map(fit_context, fold_contexts))
    else:
        fold_fits = tuple(fit_context(context) for context in fold_contexts)

    for context, fit_result in zip(fold_contexts, fold_fits):
        fold, runner, origin_index, train_indices, training_label_end_max, _ = context
        scaler, transform, _X, _Y, artifact = fit_result
        designs, provider = runner.forecast_designs(
            fold.origin,
            scaler,
            transform,
        )
        forecast_times = runner.forecast_times(fold.origin)
        prediction = runner.predict(
            artifact,
            designs,
            provider,
            forecast_times,
            transform,
        )
        actual = runner.actual(origin_index, forecast_times)
        naive = runner.seasonal_naive(fold.origin, forecast_times)
        point = (
            prediction
            if isinstance(prediction, PointForecastTensor)
            else prediction.point
        )
        fold_frame = backtest_tensors_to_long(actual, prediction, window=fold.window)
        if calibration_tracker is not None:
            fold_frame, calibration_audit = calibration_tracker.apply_to_frame(
                fold_frame,
                forecast_origin=fold.origin,
            )
            calibration_audits.append({"window": fold.window, **calibration_audit})
        cv_frames.append(fold_frame)
        if calibration_tracker is not None:
            calibration_tracker.collect_from_frame(
                fold_frame,
                forecast_origin=fold.origin,
                window=fold.window,
            )
        score_frames.append(
            evaluate_point_forecasts(
                actual,
                point,
                aggregate_weighting=aggregate_weights,
                seasonal_naive=naive,
                window=fold.window,
                eval_mask=eval_mask_config,
            )
        )
        if isinstance(prediction, MarginalForecastDistribution):
            probabilistic_frames.append(
                evaluate_marginal_distribution(
                    actual,
                    prediction,
                    eval_mask=eval_mask_config,
                    window=fold.window,
                )
            )
        fold_metadata.append(
            {
                **fold.metadata,
                "training_label_end_max": training_label_end_max.isoformat(),
            }
        )

    metadata = {
        "mode": "calendar_month",
        "train_window_days": backtest.train_window_days,
        "fold_count": backtest.fold_count,
        "stride_months": backtest.stride_months,
        "windows": fold_metadata,
    }
    if calibration_audits:
        metadata["calibration"] = calibration_audits
    write_backtest_results(
        result.test_dir,
        pd.concat(cv_frames, ignore_index=True),
        pd.concat(score_frames, ignore_index=True),
        aggregate_weighting=aggregate_weights,
        metadata={"backtest": metadata},
        probabilistic_scores_df=(
            pd.concat(probabilistic_frames, ignore_index=True)
            if probabilistic_frames
            else None
        ),
    )
    resolved_path = result.forecast_dir / "resolved_config.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved.setdefault("runtime", {})["holdout"] = metadata
    if calibration_tracker is not None:
        # CQR final：calendar_month 的 fixed-step 折为空，最终修正量在这里
        # 由 calendar 折池计算，冻结进 bundle 并重写 prediction.csv。
        final_correction, final_calibration_audit = (
            calibration_tracker.final_correction(final_runner.origin)
        )
        calibration_state = {"method": "cqr", **final_calibration_audit}
        result.bundle.calibration_state = calibration_state
        persist_model_bundle(result.bundle, result.model_dir)
        resolved["runtime"]["calibration"] = calibration_state
        if final_correction.status == "applied":
            prediction_path = result.forecast_dir / "prediction.csv"
            frame = pd.read_csv(prediction_path)
            correction = float(final_correction.correction)
            frame = attach_cqr_interval_columns(
                frame,
                lower=frame[calibration_tracker.lower_column].to_numpy(dtype=float)
                - correction,
                upper=frame[calibration_tracker.upper_column].to_numpy(dtype=float)
                + correction,
                target_coverage=calibration_tracker.target_coverage,
            )
            frame.to_csv(prediction_path, index=False, encoding="utf_8_sig")
    _write_json(resolved_path, resolved)


__all__ = ["overwrite_calendar_month_backtest"]
