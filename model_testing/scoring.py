# -*- coding: utf-8 -*-
"""回测逐折评分共用体（R5b，2026-09-06 自两份重复循环体合一，方案 v3）。

fixed-step（model_testing.fixed_step）与 calendar-month（model_testing.calendar_month）
的逐折「predict 后处理 → 评分 → CQR apply/collect → 证据」体是同一条业务规则，
此前两处各写一遍（2026-09-01 CQR 接线时就双份维护）。本模块抽出唯一实现：
调用方负责折构造与并行调度，本模块只消费每折已完成的 (fit 结果, origin, 窗口号)。

输入输出均为 forecasting_core 合同类型 + model_evaluation 评分帧；不 import
model_forecasting（runner 以显式 FoldScoringRunner 协议传入），供两侧共用且不构成包环。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from forecasting_core.artifacts import MarginalForecastDistribution
from forecasting_core.tensors import PointForecastTensor
from model_evaluation.marginal import evaluate_marginal_distribution
from model_evaluation.point import evaluate_point_forecasts
from probabilistic.calibration import ConformalCalibrationTracker

from model_testing.tensor_frames import backtest_tensors_to_long
from model_testing.contracts import FitResult, FoldScoringRunner


@dataclass
class FoldScoreResult:
    """一折的评分/落盘载荷；调用方聚合后统一写盘。"""

    window: int
    origin: pd.Timestamp
    frame: pd.DataFrame
    point_scores: pd.DataFrame
    probabilistic_scores: pd.DataFrame | None = None
    calibration_audit: Mapping[str, Any] | None = None
    execution_evidence: Mapping[str, Any] = field(default_factory=dict)


def score_holdout_fold(
    *,
    runner: FoldScoringRunner,
    fit_result: FitResult,
    origin: pd.Timestamp,
    origin_index: int,
    window: int,
    calibration_tracker: ConformalCalibrationTracker | None,
    aggregate_weights: Any,
    eval_mask_config: Mapping[str, Any] | None,
) -> FoldScoreResult:
    """对单个已完成拟合的回测折执行统一后处理。

    ``fit_result`` 为 runner.fit 返回的五元组
    ``(feature_scaler, target_transform, X, Y, artifact)``；
    ``runner`` 需提供 forecast_designs/predict/actual/seasonal_naive/
    forecast_times/execution_evidence 公开能力（与 ensemble 成员同一协议面）。
    """
    feature_scaler, target_transform, _X, _Y, artifact = fit_result
    designs, provider = runner.forecast_designs(origin, feature_scaler, target_transform)
    forecast_times = runner.forecast_times(origin)
    prediction = runner.predict(artifact, designs, provider, forecast_times, target_transform)
    actual = runner.actual(origin_index, forecast_times)
    seasonal_naive = runner.seasonal_naive(origin, forecast_times)
    point = (
        prediction
        if isinstance(prediction, PointForecastTensor)
        else prediction.point
    )
    frame = backtest_tensors_to_long(actual, prediction, window=window)
    calibration_audit = None
    if calibration_tracker is not None:
        # apply-before-collect：当前折只消费严格更早折的校准池
        frame, calibration_audit = calibration_tracker.apply_to_frame(
            frame,
            forecast_origin=origin,
        )
        calibration_tracker.collect_from_frame(
            frame,
            forecast_origin=origin,
            window=window,
        )
    point_scores = evaluate_point_forecasts(
        actual,
        point,
        aggregate_weighting=aggregate_weights,
        seasonal_naive=seasonal_naive,
        window=window,
        eval_mask=eval_mask_config,
    )
    probabilistic_scores = None
    if isinstance(prediction, MarginalForecastDistribution):
        probabilistic_scores = evaluate_marginal_distribution(
            actual,
            prediction,
            eval_mask=eval_mask_config,
            window=window,
        )
    return FoldScoreResult(
        window=window,
        origin=origin,
        frame=frame,
        point_scores=point_scores,
        probabilistic_scores=probabilistic_scores,
        calibration_audit=calibration_audit,
        execution_evidence=runner.execution_evidence(artifact, target_transform),
    )
