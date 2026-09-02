# -*- coding: utf-8 -*-
"""概率预测后处理流水线。

【休眠状态（2026-09-01 标注）】本模块不在 canonical 主链上：生产链的
crossing 修复在 ``model_forecasting/forecaster.py``（张量级，消费
``probabilistic.crossing.method``），CQR 校准由 ``probabilistic/calibration.py``
经 ``model_forecasting/runtime.py`` 的 as-of 校准追踪器接线。本模块的
DataFrame 版组合入口仅被 ``tests/test_probabilistic_pipeline.py`` 引用，
属于早期设计的预留实现，保留供参考，不得当作现役通路。
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from probabilistic.calibration import (
    attach_cqr_interval_columns,
    calibrate_quantile_band,
)
from probabilistic.postprocessing import (
    _parse_quantile_columns,
    repair_quantile_crossing,
)


def finalize_quantile_forecast(
    frame: pd.DataFrame,
    monotone_enabled: bool,
    conformal_scores: Optional[np.ndarray] = None,
    alpha: float = 0.1,
    min_scores: int = 30,
    allow_interval_shrink: bool = False,
) -> Tuple[pd.DataFrame, Optional[float]]:
    """统一完成 q50 锚定修复，并可选追加独立 CQR interval。"""
    processed = repair_quantile_crossing(frame, enabled=monotone_enabled)
    if conformal_scores is None:
        return processed, None

    quantile_columns = _parse_quantile_columns(processed)
    if len(quantile_columns) < 2:
        raise ValueError("CQR calibration requires at least two quantile columns")
    lower_column = quantile_columns[0][1]
    upper_column = quantile_columns[-1][1]
    lower, upper, correction = calibrate_quantile_band(
        processed[lower_column].to_numpy(),
        processed[upper_column].to_numpy(),
        conformal_scores,
        alpha=alpha,
        min_scores=min_scores,
        allow_interval_shrink=allow_interval_shrink,
    )
    if lower is None or upper is None:
        return processed, None
    result = attach_cqr_interval_columns(
        processed,
        lower=lower,
        upper=upper,
        target_coverage=1.0 - float(alpha),
    )
    return result, correction


__all__ = ["finalize_quantile_forecast"]
