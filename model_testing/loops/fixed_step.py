"""固定步长回测：并行拟合、按窗口顺序评分、聚合与回测产物。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator, Mapping

import pandas as pd
from pandas.tseries.frequencies import to_offset
from forecasting_core.probabilistic_spec import probabilistic_spec_from_mapping
from forecasting_core.specs import (
    ExpandingWindowBacktestSpec,
    FixedStepBacktestSpec,
    SlidingWindowBacktestSpec,
)
from model_evaluation.point import resolve_aggregate_weighting
from model_testing.contracts.protocols import BacktestRunner, BacktestWindow, FitResult
from model_testing.artifacts.reporting import write_backtest_results
from model_testing.loops.scoring import score_holdout_fold
from probabilistic.calibration import ConformalCalibrationTracker
from probabilistic.residual import ResidualCalibrationTracker
from forecasting_core.point_intervals import ResidualCalibrationSpec
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
    """有界批次：不一次保留所有折的独立设计矩阵。"""
    for start in range(0, len(windows), workers):
        batch = windows[start:start + workers]
        contexts = tuple(runner.for_backtest_window(window) for window in batch)

        def fit(item):
            context, window = item
            return context.fit(window.train_indices, force_serial=workers > 1)

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = tuple(executor.map(fit, zip(contexts, batch)))
        else:
            results = (fit((contexts[0], batch[0])),)
        yield from zip(contexts, results)
        # 先释放上一批，再创建下一批；避免同时驻留两批大矩阵。
        del contexts, results


def run_rolling_backtest(
    runner: BacktestRunner, test_dir: Path, *, mode: str,
    stitch_overview: bool = True, mode_label: str = "fixed_steps",
) -> tuple[dict[str, Any] | None, ConformalCalibrationTracker | ResidualCalibrationTracker | None, tuple[Any, ...]]:
    """rolling 系回测共用引擎（fixed/sliding/expanding）。

    折构造由 runner 按 spec 分派（contracts/windows.py），本引擎负责并行拟合、
    按窗口顺序评分、聚合与产物写盘；``stitch_overview=False``（sliding 重叠折）
    时跳过拼接总图，``mode_label`` 写入回测 metadata。
    """
    config = runner.config
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
    prob_spec = probabilistic_spec_from_mapping(config.probabilistic.canonical_payload())
    if isinstance(prob_spec.calibration, ResidualCalibrationSpec):
        calibration_tracker = ResidualCalibrationTracker(
            prob_spec.calibration, freq_offset=to_offset(str(config.problem.freq)),
        )
    elif mode == "quantile":
        if prob_spec.calibration is not None:
            calibration_tracker = ConformalCalibrationTracker(
                prob_spec,
                freq_offset=to_offset(str(config.problem.freq)),
            )
    calibration_audits: list[dict[str, Any]] = []
    backtest_windows = runner.backtest_windows()
    strict_history = config.validation.get("train_history_steps") is not None
    refit_every = config.validation.get("refit_every", 1)
    fit_indices = tuple(index for index in range(len(backtest_windows))
                        if index == 0 or (refit_every > 0 and index % refit_every == 0))
    fit_windows = tuple(backtest_windows[index] for index in fit_indices)

    window_workers = min(
        runner.execution_plan.window_workers,
        max(1, len(fit_windows)),
    )
    parallel_fits = None
    strict_fits = _fit_raw_history_windows(runner, backtest_windows, window_workers) if strict_history else None
    if not strict_history and window_workers > 1 and backtest_windows:
        target_histories = runner.backtest_target_histories(fit_windows)

        def fit_window(item):
            backtest_window, target_history = item
            return runner.fit(
                backtest_window.train_indices,
                target_history=target_history,
                force_serial=True,
            )

        with ThreadPoolExecutor(max_workers=window_workers) as executor:
            parallel_fits = dict(zip(fit_indices,
                executor.map(
                    fit_window,
                    zip(fit_windows, target_histories),
                ),
            ))

    fit_result = None
    fitted_window = None
    for window_index, backtest_window in enumerate(backtest_windows):
        refitted = window_index in fit_indices
        if strict_fits is not None:
            fold_runner, fit_result = next(strict_fits)
        else:
            fold_runner = runner
            if refitted:
                fit_result = (
                    parallel_fits[window_index]
                    if parallel_fits is not None
                    else runner.fit(backtest_window.train_indices)
                )
        if refitted:
            fitted_window = backtest_window
        if fit_result is None or fitted_window is None:
            raise RuntimeError("backtest must fit before reusing model state")
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
            **({"refitted": refitted, "fit_origin": fitted_window.origin.isoformat(),
                "fit_window": fitted_window.window, "fit_metadata": dict(fitted_window.metadata)}
               if "refit_every" in config.validation else {}),
        })
        if strict_history:
            del fold_runner, builder
            fit_result = None
    holdout_audit = tuple(holdout_audits)
    _log_provider_usage(holdout_audit)
    if cv_frames:
        backtest = config.validation.backtest
        if not isinstance(
            backtest,
            (FixedStepBacktestSpec, SlidingWindowBacktestSpec, ExpandingWindowBacktestSpec),
        ):
            raise TypeError("rolling backtest results require a rolling-mode backtest spec")
        windows = runner.backtest_windows()
        holdout_metadata = {
            **windows[-1].metadata,
            "mode": mode_label,
            "history_steps": backtest.history_steps,
            "fold_count": backtest.fold_count,
            "stride_steps": backtest.stride_steps,
            "windows": [window.metadata for window in windows],
            "execution_evidence": holdout_execution_evidence,
        }
        if isinstance(backtest, FixedStepBacktestSpec):
            holdout_metadata["train_window_steps"] = backtest.train_window_steps
            holdout_metadata["train_history_steps"] = backtest.train_history_steps
        elif isinstance(backtest, SlidingWindowBacktestSpec):
            holdout_metadata["train_window_steps"] = backtest.train_window_steps
        if calibration_audits:
            holdout_metadata["calibration"] = calibration_audits
        write_backtest_results(
            test_dir,
            pd.concat(cv_frames, ignore_index=True),
            pd.concat(score_frames, ignore_index=True),
            aggregate_weighting=aggregate_weights,
            metadata={
                "result_method": config.result_method(),
                "backtest": holdout_metadata,
                "runtime_resources": runner.runtime_resources_payload(),
            },
            probabilistic_scores_df=(
                pd.concat(prob_score_frames, ignore_index=True)
                if prob_score_frames
                else None
            ),
            stitch_overview=stitch_overview,
        )
    else:
        holdout_metadata = None
    return holdout_metadata, calibration_tracker, holdout_audit


def run_fixed_step_backtest(
    runner: BacktestRunner, test_dir: Path, *, mode: str,
) -> tuple[dict[str, Any] | None, ConformalCalibrationTracker | ResidualCalibrationTracker | None, tuple[Any, ...]]:
    """固定步长回测入口：rolling 引擎的 fixed_steps 形态（不重叠、拼接总图）。"""
    return run_rolling_backtest(runner, test_dir, mode=mode)
