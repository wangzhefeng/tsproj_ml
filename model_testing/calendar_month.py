"""Calendar-month backtest orchestration for the canonical runtime."""

from __future__ import annotations


from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import pandas as pd

from data_loading import SourceRegistry
from forecasting_core.specs import (
    CalendarMonthBacktestSpec,
    ForecastConfigSpec,
)
from model_evaluation.point import resolve_aggregate_weighting
from model_testing.reporting import write_backtest_results

from model_testing import geometry as validation
from model_testing.scoring import score_holdout_fold
from model_testing.contracts import BacktestRunner, BacktestRunnerFactory
from pandas.tseries.frequencies import to_offset
from probabilistic.calibration import ConformalCalibrationTracker
from forecasting_core.probabilistic_spec import probabilistic_spec_from_mapping


def run_calendar_month_backtest(
    config: ForecastConfigSpec,
    registry: SourceRegistry,
    final_runner: BacktestRunner,
    test_dir: Path,
    *,
    runner_factory: BacktestRunnerFactory,
) -> tuple[dict[str, Any], ConformalCalibrationTracker | None]:
    backtest_started = perf_counter()
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
    runners_by_horizon: dict[int, BacktestRunner] = {}
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
            checkpoint_root = getattr(final_runner, "checkpoint_root", None)
            runner = runner_factory(
                dynamic_config,
                registry,
                final_runner.origin,
                **({"checkpoint_root": checkpoint_root} if checkpoint_root is not None else {}),
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

    window_workers = min(
        final_runner.execution_plan.window_workers,
        len(fold_contexts),
    )

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
        fold_ctx, runner, origin_index, train_indices, training_label_end_max, _ = context
        fold = score_holdout_fold(
            runner=runner,
            fit_result=fit_result,
            origin=fold_ctx.origin,
            origin_index=origin_index,
            window=fold_ctx.window,
            calibration_tracker=calibration_tracker,
            aggregate_weights=aggregate_weights,
            eval_mask_config=eval_mask_config,
        )
        cv_frames.append(fold.frame)
        if fold.calibration_audit is not None:
            calibration_audits.append({"window": fold.window, **fold.calibration_audit})
        score_frames.append(fold.point_scores)
        if fold.probabilistic_scores is not None:
            probabilistic_frames.append(fold.probabilistic_scores)
        fold_metadata.append(
            {
                **fold_ctx.metadata,
                "training_label_end_max": training_label_end_max.isoformat(),
                "execution_evidence": fold.execution_evidence,
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
        test_dir,
        pd.concat(cv_frames, ignore_index=True),
        pd.concat(score_frames, ignore_index=True),
        aggregate_weighting=aggregate_weights,
        metadata={
            "backtest": metadata,
            "runtime_resources": final_runner.runtime_resources_payload(),
        },
        probabilistic_scores_df=(
            pd.concat(probabilistic_frames, ignore_index=True)
            if probabilistic_frames
            else None
        ),
    )
    final_runner.stage_wall_seconds["backtest"] = perf_counter() - backtest_started
    return metadata, calibration_tracker


__all__ = ["run_calendar_month_backtest"]
