# -*- coding: utf-8 -*-
"""融合器 meta-train 诊断，不是独立泛化评估。

本函数在学习融合参数所用的 OOF 样本上评分；真正的外层留出评估在
training.backtesting 中。复用 canonical 指标，不复制指标公式。

样本轴按 fold-major/series-minor 展平后映射到 N，fold_* 仅为诊断行标签；
forecast_times 是指标张量的占位，不表示真实预测时间。真实坐标保留在 OOF
folds 与 outputs.reporting 生成的诊断 long 表中。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model_ensemble.artifacts import EnsembleArtifact, OOFPredictionArtifact
from model_ensemble.inference.predictor import combine_members
from model_evaluation.point import evaluate_point_forecasts
from forecasting_core.tensors import MarginalQuantileForecastTensor, PointForecastTensor
from model_evaluation.marginal import evaluate_marginal_distribution
from forecasting_core.artifacts import MarginalForecastDistribution


def evaluate_fused_oof(
    ens_artifact: EnsembleArtifact,
    oof: OOFPredictionArtifact,
    actual: np.ndarray,
    *,
    point_level: float = 0.5,
) -> dict[str, pd.DataFrame | None]:
    """对融合后的 OOF 预测评分。

    Returns:
        ``{"point": DataFrame, "probabilistic": DataFrame | None}``；
        quantile 模式两个都产出（point 行用分布的 point_quantile 切片，与
        pinball 同一点预测口径），point 模式 probabilistic 为 None。

    此处不应用 eval_mask，诊断完整拟合样本；顶层掩码与聚合权重在独立
    外层回测消费，不能把本诊断分数作为标准测试成绩。
    """
    combined = np.asarray(
        combine_members(ens_artifact, oof.values_by_member), dtype=float
    )
    actual = np.asarray(actual, dtype=float)
    n_folds, horizon = combined.shape[0], combined.shape[1]
    if actual.shape[:2] != (n_folds, horizon):
        raise ValueError(
            f"actual shape {actual.shape} does not match fused OOF "
            f"{combined.shape} on (folds, horizon)"
        )
    series_ids = tuple(f"fold_{index + 1}" for index in range(n_folds))
    forecast_times = pd.date_range("2000-01-01", periods=horizon, freq="1h")
    actual_tensor = PointForecastTensor(
        values=actual,
        series_ids=series_ids,
        forecast_times=forecast_times,
        targets=oof.targets,
    )

    if oof.quantile_levels is None:
        prediction = PointForecastTensor(
            values=combined,
            series_ids=series_ids,
            forecast_times=forecast_times,
            targets=oof.targets,
        )
        return {
            "point": evaluate_point_forecasts(actual_tensor, prediction, window=0),
            "probabilistic": None,
        }

    quantiles = MarginalQuantileForecastTensor(
        values=combined,
        levels=tuple(float(level) for level in oof.quantile_levels),
        point_level=float(point_level),
        series_ids=series_ids,
        forecast_times=forecast_times,
        targets=oof.targets,
    )
    distribution = MarginalForecastDistribution(
        point=quantiles.point(),
        quantiles=quantiles,
        dependence_model=None,
    )
    return {
        "point": evaluate_point_forecasts(actual_tensor, distribution.point, window=0),
        "probabilistic": evaluate_marginal_distribution(actual_tensor, distribution),
    }


__all__ = ["evaluate_fused_oof"]
