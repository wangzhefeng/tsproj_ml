# -*- coding: utf-8 -*-
"""融合 OOF 评分：ensemble 的无泄漏质量证据（2026-08-30 遗留收口）。

ensemble 无独立回测循环，其无泄漏评估语义等价物是 OOF：用融合器把各成员的
OOF 预测组合后，在与 OOF 对齐的 actual 上评分。复用 canonical 评估实现
（`evaluate_point_forecasts` / `evaluate_marginal_distribution`），不复制指标公式。

张量映射约定：fold 维映射到 N（series）轴（series_ids 为 fold_1..fold_n）；
``forecast_times`` 是满足张量轴合同的占位（指标只消费 values/targets，
不消费时间戳）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model_ensemble.artifacts import EnsembleArtifact, OOFPredictionArtifact
from model_ensemble.predictor import combine_members
from model_evaluation.point import evaluate_point_forecasts
from model_forecasting.tensors import MarginalQuantileForecastTensor, PointForecastTensor
from model_evaluation.marginal import evaluate_marginal_distribution
from probabilistic.types import MarginalForecastDistribution


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

    掩码说明：eval_mask 是成员级 validation 概念，ensemble 层未定义融合口径，
    本函数不接掩码（与单模型掩码口径的差异见 AGENTS.md 流水线数据契约）。
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
