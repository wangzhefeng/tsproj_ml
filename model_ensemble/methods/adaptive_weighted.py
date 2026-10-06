"""按原点时间衰减的逆误差融合；只学习一次，部署使用冻结权重。"""
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from model_ensemble.artifacts import TemporalWeightsArtifact
from model_ensemble.methods.horizon import fit_horizon_weights
from model_ensemble.methods.weighted import fit_weighted

METHOD_NAME = "adaptive_weighted"


def fit_adaptive_weighted(
    member_values: Mapping[str, np.ndarray], actual: np.ndarray, *,
    sample_label_ends: Sequence[str], forecast_origin: pd.Timestamp,
    halflife_days: float, metric: str = "rmse", weight_scope: str = "target",
) -> TemporalWeightsArtifact:
    if isinstance(halflife_days, bool) or not math.isfinite(halflife_days) or halflife_days <= 0:
        raise ValueError("halflife_days must be finite and positive")
    times = pd.DatetimeIndex(sample_label_ends)
    if len(times) != actual.shape[0] or times.hasnans or pd.isna(forecast_origin):
        raise ValueError("sample label times must be finite and match samples")
    if (times > forecast_origin).any():
        raise ValueError("sample labels must be available at forecast origin")
    # 以最新完整标签平移年龄，归一化结果不变，并避免远期 origin 导致全体下溢。
    age_days = np.asarray((times.max() - times).total_seconds(), dtype=float) / 86400.0
    weights = np.exp2(-age_days / float(halflife_days))
    weights /= weights.sum()
    kwargs = {"metric": metric, "sample_weights": weights}
    if weight_scope == "target_horizon":
        fitted = fit_horizon_weights(fit_weighted, member_values, actual, **kwargs)
    elif weight_scope == "target":
        fitted = fit_weighted(member_values, actual, **kwargs)
    else:
        raise ValueError("weight_scope must be target or target_horizon")
    return TemporalWeightsArtifact(
        fitted, forecast_origin.isoformat(), float(halflife_days),
        tuple(time.isoformat() for time in times), tuple(float(value) for value in weights),
    )
