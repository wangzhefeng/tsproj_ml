# -*- coding: utf-8 -*-
"""点模型预测区间（绝对残差校准）评分。

消费 `forecasting_core.probability.intervals.PointIntervalForecast`（残差校准产出的
lower/upper/available），按 target 与 horizon 输出 coverage/width/winkler/
coverage_gap；只计算 ``available`` 且通过评估掩码的样本，无可用区间时
``n_points=0``、指标 NaN。不生成 pinball 或伪 quantile 指标——点模型的区间
不是分位数预测。

生产通路：canonical 回测 point + absolute_residual 校准模式 →
`model_testing/loops/scoring.py` → `results_test/test_scores_probabilistic_df.csv`。
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from forecasting_core.probability.intervals import PointIntervalForecast
from forecasting_core.tensors.point import PointForecastTensor
from forecasting_core.tensors.layout import require_matching_point_axes
from model_evaluation.metrics import interval_metrics
from model_evaluation.point import build_eval_mask_payload

# 输出 metric 名 → interval_metrics 返回键
_METRIC_KEYS = {
    "interval_coverage": "coverage",
    "interval_width": "width",
    "interval_winkler": "winkler",
    "coverage_gap": "coverage_gap",
}


def evaluate_point_intervals(
    actual: PointForecastTensor,
    forecast: PointIntervalForecast,
    *,
    eval_mask: Mapping[str, Any] | None = None,
    window: int | None = None,
) -> pd.DataFrame:
    """对点模型的校准区间评分（scope 与 marginal 评分帧同集，2026-10-05 补齐）。

    scope ∈ {target, horizon, aggregate, aggregate_horizon}；aggregate 系行按
    掩码后有效点跨 target 池化（proper score 语义，与 marginal 同口径），
    无可用区间的 target 不参与池化。

    Args:
        actual: 真值点张量。
        forecast: 残差校准区间预测（含 ``available`` 可用性掩码与
            ``target_coverage``）。
        eval_mask: 可选，validation.eval_mask 配置（mode/percentile/min_value/
            max_value），与点/概率评估同一入口构造逐 target 掩码。
        window: 可选窗口编号，写入 ``window`` 列（回测逐窗调用时传入）。

    Returns:
        DataFrame，列：``window, scope, target, horizon, metric,
        interval_name, target_coverage, value, n_points``；``horizon`` 仅
        per-horizon 行非空（1-based）。
    """
    require_matching_point_axes(actual, forecast.point)
    masks = build_eval_mask_payload(eval_mask, actual) if eval_mask is not None else None
    rows = []
    n_series, n_horizons = actual.shape[0], actual.shape[1]
    # 池化收集：aggregate（整体）与 aggregate_horizon（逐 h）跨 target 有效点。
    pooled = {"y": [], "lower": [], "upper": []}
    horizon_pools = {h: {"y": [], "lower": [], "upper": []} for h in range(n_horizons)}
    for k, target in enumerate(actual.targets):
        target_mask = (np.asarray(masks[target]["valid_mask"], dtype=bool).reshape(n_series, n_horizons)
                       if masks else np.ones((n_series, n_horizons), dtype=bool))
        # 完整有效掩码（2D）只算一遍：target 行用全量，horizon 行按 h 切片。
        valid_2d = (
            forecast.available[:, :, k]
            & np.isfinite(actual.values[:, :, k])
            & target_mask
        )
        for h in (None, *range(n_horizons)):
            if h is None:
                values = actual.values[:, :, k]
                lower = forecast.lower[:, :, k]
                upper = forecast.upper[:, :, k]
                valid = valid_2d
            else:
                values = actual.values[:, h, k]
                lower = forecast.lower[:, h, k]
                upper = forecast.upper[:, h, k]
                valid = valid_2d[:, h]
            n = int(valid.sum())
            scores = interval_metrics(values[valid], lower[valid], upper[valid], forecast.target_coverage) if n else {}
            for metric, key in _METRIC_KEYS.items():
                rows.append({"window": window, "scope": "target" if h is None else "horizon", "target": target,
                             "horizon": pd.NA if h is None else h + 1, "metric": metric,
                             "interval_name": "absolute_residual", "target_coverage": forecast.target_coverage,
                             "value": scores.get(key, float("nan")), "n_points": n})
            if n:
                sink = pooled if h is None else horizon_pools[h]
                sink["y"].append(values[valid])
                sink["lower"].append(lower[valid])
                sink["upper"].append(upper[valid])

    def _emit_pool(scope: str, h: int | None, pool: dict) -> None:
        y = np.concatenate(pool["y"])
        lower = np.concatenate(pool["lower"])
        upper = np.concatenate(pool["upper"])
        scores = interval_metrics(y, lower, upper, forecast.target_coverage)
        for metric, key in _METRIC_KEYS.items():
            rows.append({"window": window, "scope": scope, "target": "__aggregate__",
                         "horizon": pd.NA if h is None else h + 1, "metric": metric,
                         "interval_name": "absolute_residual", "target_coverage": forecast.target_coverage,
                         "value": scores[key], "n_points": int(len(y))})

    if pooled["y"]:
        _emit_pool("aggregate", None, pooled)
        for h in range(n_horizons):
            if horizon_pools[h]["y"]:
                _emit_pool("aggregate_horizon", h, horizon_pools[h])
    return pd.DataFrame(rows)


__all__ = ["evaluate_point_intervals"]
