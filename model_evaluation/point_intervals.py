"""点模型预测区间评分；不生成 pinball 或伪 quantile 指标。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting_core.point_intervals import PointIntervalForecast
from forecasting_core.tensors import PointForecastTensor, require_matching_point_axes
from model_evaluation.metrics import interval_metrics
from model_evaluation.point import build_eval_mask_payload


def evaluate_point_intervals(actual: PointForecastTensor, forecast: PointIntervalForecast, *, eval_mask=None, window=None) -> pd.DataFrame:
    require_matching_point_axes(actual, forecast.point)
    masks = build_eval_mask_payload(eval_mask, actual) if eval_mask is not None else None
    rows = []
    names = {"interval_coverage": "coverage", "interval_width": "width", "interval_winkler": "winkler", "coverage_gap": "coverage_gap"}
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
            for metric, key in names.items():
                rows.append({"window": window, "scope": "target" if h is None else "horizon", "target": target,
                             "horizon": pd.NA if h is None else h + 1, "metric": metric,
                             "interval_name": "absolute_residual", "target_coverage": forecast.target_coverage,
                             "value": scores.get(key, float("nan")), "n_points": n})
    return pd.DataFrame(rows)
