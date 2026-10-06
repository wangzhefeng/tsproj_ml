"""融合预测的张量组装；运行、回测与部署共用概率合同。"""
from typing import Any

import numpy as np
import pandas as pd

from forecasting_core.artifacts import MarginalForecastDistribution
from forecasting_core.probabilistic_spec import ProbabilisticSpec
from forecasting_core.tensors import MarginalQuantileForecastTensor, PointForecastTensor
from model_predicting.loops.predictor import build_crossing_report, repair_marginal_quantile_crossing


def build_ensemble_forecast(
    values: np.ndarray,
    *,
    probability: ProbabilisticSpec,
    series_ids: tuple[Any, ...],
    forecast_times: pd.DatetimeIndex,
    targets: tuple[str, ...],
    method_name: str,
) -> PointForecastTensor | MarginalForecastDistribution:
    """在原始目标空间执行融合后的 crossing，并绑定顶层 point_quantile。"""
    if probability.mode == "point":
        return PointForecastTensor(values, series_ids, forecast_times, targets)
    raw = MarginalQuantileForecastTensor(
        values=values, levels=probability.quantiles, point_level=probability.point_quantile,
        series_ids=series_ids, forecast_times=forecast_times, targets=targets,
    )
    quantiles = repair_marginal_quantile_crossing(raw, method=probability.crossing_method)
    metadata: dict[str, Any] = {"ensemble_method": method_name, "crossing_method": probability.crossing_method}
    if probability.crossing_report_raw:
        report = build_crossing_report(raw, quantiles)
        if report is not None:
            metadata["crossing_report"] = report
    return MarginalForecastDistribution(quantiles.point(), quantiles, metadata=metadata)
