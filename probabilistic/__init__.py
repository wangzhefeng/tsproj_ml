# -*- coding: utf-8 -*-
"""概率预测校准能力包（CQR）。

2026-09-06 R4 行为不变重构（方案 v3，.hermes/plans/2026-09-06_162605）：
quantile 逐 level 训练编排与原生 objective 能力映射已迁出——
`model_training/quantile.py` 与 `model_training/objectives.py`；
本包仅保留 `calibration`（Conformalized Quantile Regression 内核，
经 runtime 的 as-of 校准追踪器接线）。Crossing 修复由
`model_forecasting.predictor` 的张量级实现负责，不维护 DataFrame 版副本。
"""

from forecasting_core.probabilistic_spec import validate_quantile_grid

__all__ = ["validate_quantile_grid"]
