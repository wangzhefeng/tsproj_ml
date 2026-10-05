# -*- coding: utf-8 -*-
"""点模型预测区间（绝对残差校准）评分。

消费 `forecasting_core.point_intervals.PointIntervalForecast`（残差校准产出的
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

from forecasting_core.point_intervals import PointIntervalForecast
from forecasting_core.tensors import PointForecastTensor, require_matching_point_axes
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
    """对点模型的校准区间评分（scope ∈ {target, horizon}，无 aggregate 行）。

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
    for k, target in enumerate(actual.targets):
        target_mask = (np.asarray(masks[target]["valid_mask"], dtype=bool).reshape(actual.shape[0], actual.shape[1])
                       if masks else np.ones(actual.shape[:2], dtype=bool))
        for h in (None, *range(actual.shape[1])):
            values = actual.values[:, :, k] if h is None else actual.values[:, h, k]
            lower = forecast.lower[:, :, k] if h is None else forecast.lower[:, h, k]
            upper = forecast.upper[:, :, k] if h is None else forecast.upper[:, h, k]
            available = forecast.available[:, :, k] if h is None else forecast.available[:, h, k]
            valid = available & np.isfinite(values) & (target_mask if h is None else target_mask[:, h])
            n = int(valid.sum())
            scores = interval_metrics(values[valid], lower[valid], upper[valid], forecast.target_coverage) if n else {}
            for metric, key in _METRIC_KEYS.items():
                rows.append({"window": window, "scope": "target" if h is None else "horizon", "target": target,
                             "horizon": pd.NA if h is None else h + 1, "metric": metric,
                             "interval_name": "absolute_residual", "target_coverage": forecast.target_coverage,
                             "value": scores.get(key, float("nan")), "n_points": n})
    return pd.DataFrame(rows)


__all__ = ["evaluate_point_intervals"]
