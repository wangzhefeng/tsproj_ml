# -*- coding: utf-8 -*-
"""评估模块（2026-08-30 流水线阶段重排新建，回应「独立评估模块」架构决策）。

职责边界：只做**指标计算**（点评估、评估掩码、边际 quantile 评估、指标内核），
不写结果文件（结果读写 schema 留在 `model_forecasting/results.py`）、不做预测期
后处理（crossing 修复在 `probabilistic/postprocessing.py`）。

- `mask.py`：评估掩码（percentile/absolute/combined）——点/概率评估共用口径；
- `point.py`：点预测评估（MAE/RMSE/MAPE/Accuracy + naive 对照 + 聚合加权）；
- `marginal.py`：边际 quantile 评估（pinball + central 区间覆盖率/宽度/winkler）；
- `metrics.py`：指标内核（pinball_loss、interval_metrics、wilson_interval 等纯函数）。
"""

from model_evaluation.mask import build_eval_mask
from model_evaluation.marginal import evaluate_marginal_distribution
from model_evaluation.point import (
    build_eval_mask_payload,
    evaluate_point_forecasts,
    resolve_aggregate_weighting,
)

__all__ = [
    "build_eval_mask",
    "build_eval_mask_payload",
    "evaluate_marginal_distribution",
    "evaluate_point_forecasts",
    "resolve_aggregate_weighting",
]
