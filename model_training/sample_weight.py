"""显式训练时间权重，不以评估掩码或数据删除替代加权。"""
from collections.abc import Mapping

import numpy as np
import pandas as pd


def temporal_sample_weight(origins, cutoff, spec: Mapping | None) -> np.ndarray | None:
    if spec is None:
        return None
    if not isinstance(spec, Mapping) or set(spec) != {'method', 'halflife_days'}:
        raise ValueError('sample_weight requires method and halflife_days only')
    half = spec['halflife_days']
    if spec['method'] != 'exponential' or isinstance(half, bool) or not isinstance(half, (int, float)):
        raise ValueError('sample_weight requires exponential method and numeric halflife_days')
    if not np.isfinite(half) or half <= 0:
        raise ValueError('sample_weight halflife_days must be finite and positive')
    times = pd.DatetimeIndex(origins)
    end = pd.Timestamp(cutoff)
    if times.empty or times.hasnans or pd.isna(end) or (times > end).any():
        raise ValueError('sample_weight origins must be nonempty, valid and not after cutoff')
    ages = np.asarray((end - times).total_seconds(), dtype=float) / 86400.0
    weights = np.exp2(-(ages - ages.min()) / float(half))
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('sample_weight underflow or invalid weights')
    return weights / weights.mean()
