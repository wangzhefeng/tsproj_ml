"""Calendar-month backtest orchestration for the canonical runtime."""

from __future__ import annotations

import json
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
    build_eval_mask_payload,
    evaluate_point_forecasts,
    resolve_aggregate_weighting,
)
from model_forecasting.results import backtest_tensors_to_long, write_backtest_results
from model_testing import validation


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
    for fold in folds:
        dynamic_problem = replace(config.problem, horizon=fold.horizon)
        dynamic_validation = {
            key: value
            for key, value in dict(config.validation).items()
            if key not in {"train_window_days", "stride_months"}
        }
        dynamic_history_steps = len(final_runner.supervised_origins)
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
        runner = runner_factory(
            dynamic_config,
            registry,
            final_runner.origin,
        )
        try:
            origin_index = runner.supervised_origins.index(fold.origin)
        except ValueError as exc:
            raise ValueError(
                f"calendar-month origin {fold.origin} is not a supervised origin"
            ) from exc
        holdout_label_start = runner.geometry.label_start(fold.origin)
        raw_history_times = final_runner.builder.target_history_times(
            final_runner.origin
        )
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

        scaler, transform, _X, _Y, artifact = runner.fit(train_indices)
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
        cv_frames.append(
            backtest_tensors_to_long(actual, prediction, window=fold.window)
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
            mask_payload = build_eval_mask_payload(eval_mask_config, actual)
            probabilistic_frames.append(
                evaluate_marginal_distribution(
                    actual,
                    prediction,
                    valid_masks=(
                        {
                            target: payload["valid_mask"]
                            for target, payload in mask_payload.items()
                        }
                        if mask_payload is not None
                        else None
                    ),
                    window=fold.window,
                )
            )
        fold_metadata.append(
            {
                **fold.metadata,
                "training_label_end_max": max(
                    runner.geometry.label_end(runner.supervised_origins[index])
                    for index in train_indices
                ).isoformat(),
            }
        )

    metadata = {
        "mode": "calendar_month",
        "train_window_days": backtest.train_window_days,
        "fold_count": backtest.fold_count,
        "stride_months": backtest.stride_months,
        "windows": fold_metadata,
    }
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
    _write_json(resolved_path, resolved)


__all__ = ["overwrite_calendar_month_backtest"]
