# -*- coding: utf-8 -*-
"""边际 quantile 分布评估：per-target/aggregate 的 pinball + central 区间指标。

自 `probabilistic/model_evaluation.py` 迁入（2026-08-30 evaluation 模块化），实现逐字保真。
生产通路：canonical 回测 quantile 模式 → `results_test/test_scores_probabilistic_df.csv`；
ensemble 融合 OOF 评分（`model_ensemble/model_evaluation.py`）复用本模块。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

from model_evaluation.metrics import interval_metrics, pinball_loss
from forecasting_core.tensors import PointForecastTensor
from forecasting_core.artifacts import MarginalForecastDistribution


def _central_interval_pairs(levels) -> List[Tuple[float, float]]:
    """对称 central 区间对：level q 与 1-q 同时存在时构成 (q, 1-q)。"""
    ordered = sorted(float(level) for level in levels)
    pairs: List[Tuple[float, float]] = []
    for low in ordered:
        if low >= 0.5:
            continue
        matches = [
            high
            for high in ordered
            if np.isclose(high, 1.0 - low, rtol=0.0, atol=1e-12)
        ]
        if len(matches) == 1:
            pairs.append((low, matches[0]))
    return pairs


def evaluate_marginal_distribution(
    actual: PointForecastTensor,
    distribution: MarginalForecastDistribution,
    *,
    valid_masks: Mapping[str, Any] | None = None,
    window: int | None = None,
) -> pd.DataFrame:
    """Return explicit per-target point, pinball and central-interval metrics.

    - ``valid_masks``: 可选，target -> 逐点 bool 掩码（与 ``actual.values[:, :, k]
      .reshape(-1)`` 同序），把与点评估相同的业务口径（D13 eval_mask）同步作用于
      概率指标；未提供时掩码逻辑不参与，行为与历史逐值一致。掩码与 isfinite
      正交组合。
    - ``window``: 可选窗口编号，写入 ``window`` 列（回测逐窗调用时传入）。
    - 输出 schema（tidy long）：``window, scope, target, metric, quantile,
      interval_name, lower_quantile, upper_quantile, target_coverage,
      value, n_points, ci_lower, ci_upper``。metric ∈ {mae, pinball,
      interval_coverage, interval_width, interval_winkler, coverage_gap,
      calibration_error}；interval_* / coverage_* 行的 quantile 为 NaN，
      mae/pinball 行的 interval_* 列为 NaN。
    - 区间对自动从 quantile levels 推导：q 与 1-q 同时存在即构成 central
      区间（如 [0.1, 0.5, 0.9] → central80，target_coverage=0.8）。
    - aggregate 行（``scope="aggregate"``, ``target="__aggregate__"``）按掩码后
      有效点跨 target 池化（proper score 语义，所有有效点等权），与点评估的
      target 加权 aggregate 语义不同，勿混用。
    """
    if not isinstance(actual, PointForecastTensor):
        raise TypeError("actual must be a PointForecastTensor")
    if not isinstance(distribution, MarginalForecastDistribution):
        raise TypeError("distribution must be a MarginalForecastDistribution")
    point = distribution.point
    if (
        actual.series_ids != point.series_ids
        or actual.targets != point.targets
        or not actual.forecast_times.equals(point.forecast_times)
    ):
        raise ValueError("actual and marginal distribution axes must match")

    levels = tuple(float(level) for level in distribution.quantiles.levels)
    level_index = {level: i for i, level in enumerate(levels)}
    interval_pairs = _central_interval_pairs(levels)

    rows = []
    actual_values = actual.values
    point_values = point.values
    quantile_values = distribution.quantiles.values
    pooled: Dict[str, Any] = {"y_true": [], "y_point": [], "quantiles": {level: [] for level in levels}}

    def _emit(scope, target, y_true, y_point, quantile_flat, mask):
        valid = np.isfinite(y_true) & np.isfinite(y_point)
        for level in levels:
            valid = valid & np.isfinite(quantile_flat[level])
        if mask is not None:
            valid = valid & mask
        n_points = int(valid.sum())

        def _row(metric, value, quantile=np.nan, interval_name=None,
                 lower_quantile=np.nan, upper_quantile=np.nan,
                 target_coverage=np.nan, ci_lower=np.nan, ci_upper=np.nan):
            rows.append(
                {
                    "window": window if window is not None else pd.NA,
                    "scope": scope,
                    "target": target,
                    "metric": metric,
                    "quantile": quantile,
                    "interval_name": interval_name,
                    "lower_quantile": lower_quantile,
                    "upper_quantile": upper_quantile,
                    "target_coverage": target_coverage,
                    "value": value,
                    "n_points": n_points,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            )

        if n_points == 0:
            _row("mae", float("nan"))
            for level in levels:
                _row("pinball", float("nan"), quantile=level)
            return None

        y_valid = y_true[valid]
        point_valid = y_point[valid]
        _row("mae", float(np.mean(np.abs(y_valid - point_valid))))
        for level in levels:
            loss = pinball_loss(y_valid, quantile_flat[level][valid], level)
            _row("pinball", float(np.mean(loss)), quantile=level)
        for lower, upper in interval_pairs:
            metrics = interval_metrics(
                y_valid,
                quantile_flat[lower][valid],
                quantile_flat[upper][valid],
                upper - lower,
            )
            name = f"central{int(round((upper - lower) * 100))}"
            _row(
                "interval_coverage",
                metrics["coverage"],
                interval_name=name,
                lower_quantile=lower,
                upper_quantile=upper,
                target_coverage=upper - lower,
                ci_lower=metrics["coverage_ci_lower"],
                ci_upper=metrics["coverage_ci_upper"],
            )
            _row("interval_width", metrics["width"], interval_name=name,
                 lower_quantile=lower, upper_quantile=upper,
                 target_coverage=upper - lower)
            _row("interval_winkler", metrics["winkler"], interval_name=name,
                 lower_quantile=lower, upper_quantile=upper,
                 target_coverage=upper - lower)
            _row("coverage_gap", metrics["coverage_gap"], interval_name=name,
                 lower_quantile=lower, upper_quantile=upper,
                 target_coverage=upper - lower)
            _row("calibration_error", metrics["calibration_error"],
                 interval_name=name, lower_quantile=lower, upper_quantile=upper,
                 target_coverage=upper - lower)
        return valid

    for target_index, target in enumerate(actual.targets):
        y_true = actual_values[:, :, target_index].reshape(-1)
        y_point = point_values[:, :, target_index].reshape(-1)
        quantile_flat = {
            level: quantile_values[:, :, target_index, level_index[level]].reshape(-1)
            for level in levels
        }
        mask = None
        if valid_masks is not None and target in valid_masks:
            mask = np.asarray(valid_masks[target], dtype=bool).reshape(-1)
            if mask.shape != y_true.shape:
                raise ValueError(
                    f"valid mask for target {target!r} has shape {mask.shape}, "
                    f"expected {y_true.shape}"
                )
        valid = _emit("target", target, y_true, y_point, quantile_flat, mask)
        if valid is not None:
            pooled["y_true"].append(y_true[valid])
            pooled["y_point"].append(y_point[valid])
            for level in levels:
                pooled["quantiles"][level].append(quantile_flat[level][valid])

    if pooled["y_true"]:
        _emit(
            "aggregate",
            "__aggregate__",
            np.concatenate(pooled["y_true"]),
            np.concatenate(pooled["y_point"]),
            {level: np.concatenate(pooled["quantiles"][level]) for level in levels},
            None,
        )
    return pd.DataFrame(rows)


__all__ = ["evaluate_marginal_distribution"]
