"""固定步长回测：并行拟合、按窗口顺序评分、聚合与回测产物。"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Mapping

import pandas as pd
from pandas.tseries.frequencies import to_offset
from forecasting_core.probabilistic_spec import probabilistic_spec_from_mapping
from forecasting_core.specs import FixedStepBacktestSpec
from forecasting_core.specs.temporal import has_bounded_history
from model_evaluation.point import resolve_aggregate_weighting
from model_testing.contracts import BacktestRunner, BacktestWindow, FitResult
from model_testing.reporting import write_backtest_results
from model_testing.scoring import score_holdout_fold
from model_testing.execution import ordered_bounded_map
from probabilistic.calibration import ConformalCalibrationTracker
from utils.log_util import logger

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


def _fit_raw_history_windows(
    runner: BacktestRunner, windows: tuple[BacktestWindow, ...], workers: int,
) -> Iterator[tuple[BacktestRunner, FitResult]]:
    """编译与拟合同属有界任务，消费方按序评分后释放。"""
    def fit(window):
        context = runner.for_backtest_window(window)
        return context, context.fit(window.train_indices, force_serial=workers > 1)
    yield from ordered_bounded_map(fit, windows, workers=workers)


def run_fixed_step_backtest(
    runner: BacktestRunner, test_dir: Path, *, mode: str,
) -> tuple[dict[str, Any] | None, ConformalCalibrationTracker | None, tuple[Any, ...]]:
    config = runner.config
    builder = runner.builder
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
    backtest_windows = runner.backtest_windows()
    strict_history = has_bounded_history(config.validation)

    window_workers = min(
        runner.execution_plan.window_workers,
        max(1, len(backtest_windows)),
    )
    refit_every = config.validation.get("refit_every", 1)
    parallel_fits = None
    strict_fits = (_fit_raw_history_windows(runner, backtest_windows, window_workers)
                   if strict_history and refit_every == 1 else None)
    if refit_every == 1 and not strict_history and window_workers > 1 and backtest_windows:
        target_histories = runner.backtest_target_histories(backtest_windows)

        def fit_window(item):
            backtest_window, target_history = item
            return runner.fit(
                backtest_window.train_indices,
                target_history=target_history,
                force_serial=True,
            )

        parallel_fits = ordered_bounded_map(
            fit_window, zip(backtest_windows, target_histories), workers=window_workers,
        )

    retained_fit = None
    fitted_origin = None
    for window_index, backtest_window in enumerate(backtest_windows):
        did_refit = window_index % refit_every == 0
        if refit_every > 1:
            # 每个预测原点更新只读信息集，不沿用旧的历史下界。
            fold_runner = runner.for_backtest_window(backtest_window)
            if did_refit:
                retained_fit = fold_runner.fit(backtest_window.train_indices)
                fitted_origin = backtest_window.origin
            if retained_fit is None:
                raise RuntimeError("refit schedule requires an initial fitted model")
            fit_result = retained_fit
        elif strict_fits is not None:
            fold_runner, fit_result = next(strict_fits)
        else:
            fold_runner = runner
            fit_result = (
                next(parallel_fits)
                if parallel_fits is not None
                else runner.fit(backtest_window.train_indices)
            )
        if refit_every == 1:
            fitted_origin = backtest_window.origin
        if fitted_origin is None:
            raise RuntimeError("missing fitted origin")
        builder = fold_runner.builder
        builder.reset_audit()
        fold = score_holdout_fold(
            runner=fold_runner,
            fit_result=fit_result,
            origin=backtest_window.origin,
            origin_index=backtest_window.origin_index,
            window=backtest_window.window,
            calibration_tracker=calibration_tracker,
            aggregate_weights=aggregate_weights,
            eval_mask_config=eval_mask_config,
        )
        if config.validation.get("training_window") is not None:
            fold.frame["forecast_origin"] = backtest_window.origin.isoformat()
            times = pd.DatetimeIndex(pd.to_datetime(fold.frame["time"]))
            grid = pd.date_range(backtest_window.origin, times.max(), freq=config.problem.freq)
            lead_steps = grid.get_indexer(times)
            if (lead_steps <= 0).any():
                raise ValueError("forecast times must follow origin on the configured frequency grid")
            fold.frame["lead_steps"] = lead_steps
            first_lead = int(fold.frame["lead_steps"].min())
            fold.point_scores["forecast_origin"] = backtest_window.origin.isoformat()
            if "horizon" in fold.point_scores:
                fold.point_scores["lead_steps"] = fold.point_scores["horizon"] + first_lead - 1
        cv_frames.append(fold.frame)
        if fold.calibration_audit is not None:
            calibration_audits.append(
                {"window": fold.window, **fold.calibration_audit}
            )
        score_frames.append(fold.point_scores)
        if fold.probabilistic_scores is not None:
            prob_score_frames.append(fold.probabilistic_scores)
        holdout_audits.extend(builder.audit)
        holdout_execution_evidence.append({
            "window": fold.window,
            "origin": fold.origin.isoformat(),
            **fold.execution_evidence,
            "did_refit": did_refit,
            "model_fit_origin": fitted_origin.isoformat(),
            "refit_every": refit_every,
            "effective_window_workers": 1 if refit_every > 1 else window_workers,
            "feature_design_seconds": fold_runner.stage_wall_seconds.get("raw_design", 0.0) if strict_history else None,
        })
        del fold_runner, fit_result, builder
    holdout_audit = tuple(holdout_audits)
    _log_provider_usage(holdout_audit)
    if cv_frames:
        backtest = config.validation.backtest
        if not isinstance(backtest, FixedStepBacktestSpec):
            raise TypeError("fixed backtest results require FixedStepBacktestSpec")
        windows = runner.backtest_windows()
        holdout_metadata = {
            **windows[-1].metadata,
            "mode": "fixed_steps",
            "history_steps": backtest.history_steps,
            "training_window": dict(config.validation["training_window"]),
            "fold_count": backtest.fold_count,
            "stride_steps": backtest.stride_steps,
            "windows": [window.metadata for window in windows],
            "execution_evidence": holdout_execution_evidence,
        }
        if calibration_audits:
            holdout_metadata["calibration"] = calibration_audits
        write_started = perf_counter()
        write_backtest_results(
            test_dir,
            pd.concat(cv_frames, ignore_index=True),
            pd.concat(score_frames, ignore_index=True),
            aggregate_weighting=aggregate_weights,
            metadata={
                "result_method": config.result_method(),
                "allow_overlapping_windows": config.validation.get("training_window") is not None,
                "backtest": holdout_metadata,
                "runtime_resources": runner.runtime_resources_payload(),
            },
            probabilistic_scores_df=(
                pd.concat(prob_score_frames, ignore_index=True)
                if prob_score_frames
                else None
            ),
        )
        runner.stage_wall_seconds["backtest_result_write"] = perf_counter() - write_started
    else:
        holdout_metadata = None
    return holdout_metadata, calibration_tracker, holdout_audit
