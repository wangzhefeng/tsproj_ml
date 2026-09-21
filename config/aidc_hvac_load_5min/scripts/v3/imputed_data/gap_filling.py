"""v3 固定窗口补缺：仅过去原观测，长缺口不递归读取估计值。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.aidc_hvac_load_5min.scripts.imputed_data.impute_hvac_data import true_runs

DAY = 288
METHODS = ('locf', 'past_mean_1h', 'past_day_repeat')
LOOKBACKS = (1, 12, DAY)
BUCKETS = (1, 3, 12, 36, 72, DAY, 3 * DAY, 7 * DAY, 24 * DAY)
CALIBRATION = 30 * DAY


def predictions(values, starts, length):
    """遮蔽起点之后的值一律不可用；昨日曲线按模重复，不越过原点。"""
    starts = np.asarray(starts, dtype=int)
    result = np.full((len(starts), len(METHODS), length), np.nan)
    for method, lookback in enumerate(LOOKBACKS):
        valid = starts >= lookback
        origins = starts[valid]
        if method == 0:
            pred = np.repeat(values[origins - 1, None], length, axis=1)
        elif method == 1:
            history = values[origins[:, None] - np.arange(1, 13)]
            pred = np.repeat(history.mean(axis=1)[:, None], length, axis=1)
        else:
            # 短缺口取昨日对应片段；长缺口重复同一完整过去日。
            pred = values[origins[:, None] - DAY + np.arange(length) % DAY]
        result[valid, method] = pred
    return result


def validation_blocks(values, length):
    """仅原观测计分；长段缺失标签不补值，覆盖不足的整块剔除。"""
    starts = np.arange(DAY, len(values) - length + 1, 72)
    truth = values[starts[:, None] + np.arange(length)]
    observed = np.isfinite(truth)
    coverage = observed.mean(axis=1)
    pred = predictions(values, starts, length)
    errors = np.abs(pred - truth[:, None, :])
    sums = np.where(observed[:, None, :], errors, 0.).sum(axis=2)
    scores = sums / np.maximum(observed.sum(axis=1), 1)[:, None]
    threshold = 1. if length <= 72 else .8
    scores[coverage < threshold] = np.nan
    scores[~np.isfinite(pred).all(axis=2)] = np.nan
    return starts, scores, coverage


def fill_series(source, *, output_start):
    """只补 output_start 起的缺口；此前最多30天原观测仅用于校准。"""
    index = source.index
    if (not isinstance(index, pd.DatetimeIndex) or index.empty
            or not index.equals(pd.date_range(index[0], periods=len(index), freq='5min'))):
        raise ValueError('需要唯一、升序、连续5min时间轴')
    output_start = pd.Timestamp(output_start)
    if output_start not in index:
        raise ValueError('输出起点不在输入网格')
    values = source.to_numpy(dtype=float, copy=True)
    if np.isinf(values).any():
        raise ValueError('无限值不能作为缺失或正常观测')
    result = values.copy()
    audits = []
    validations = {}
    first = int(index.searchsorted(output_start))
    for start, stop in true_runs(np.isnan(values)):
        if stop <= first:
            continue
        length = int(stop - start)
        if start < first or start == 0 or stop == len(values):
            raise ValueError(f'{source.name}: 无双端原观测的边缘缺口 {index[start]}')
        if length > BUCKETS[-1]:
            raise ValueError(f'{source.name}: 缺口超过已验证长度上限 {length}')
        bucket = next(size for size in BUCKETS if size >= length)
        candidates = predictions(values, [start], length)[0]
        available = np.isfinite(candidates).all(axis=1)
        if bucket not in validations:
            validations[bucket] = validation_blocks(values, bucket)
        origins, errors, coverage = validations[bucket]
        valid = ((origins + bucket <= start) & (origins >= start - CALIBRATION)
                 & np.isfinite(errors[:, available]).all(axis=1))
        if not available.any() or valid.sum() < 3:
            raise ValueError(f'{source.name}: 缺口{index[start]} 长度{length} 校准不足: {int(valid.sum())}')
        scores = np.full(len(METHODS), np.inf)
        scores[available] = np.median(errors[valid][:, available], axis=0)
        winner = int(scores.argmin())
        result[start:stop] = candidates[winner]
        audits.append({
            'point': source.name, 'gap_start': str(index[start]), 'gap_end': str(index[stop - 1]),
            'gap_points': length, 'validation_bucket_points': bucket, 'method': METHODS[winner],
            'validation_count': int(valid.sum()),
            'validation_coverage_min': float(coverage[valid].min()),
            'validation_start': str(index[origins[valid][0]]),
            'validation_end': str(index[origins[valid][-1] + bucket - 1]),
            'raw_dependency_start': str(index[min(origins[valid][0] - max(
                np.asarray(LOOKBACKS)[available]), start - LOOKBACKS[winner])]),
            'raw_dependency_end': str(index[start - 1]),
            'eligibility_known_at': str(index[stop]),
            **{method + '_median_mae': float(scores[i]) if np.isfinite(scores[i]) else None
               for i, method in enumerate(METHODS)},
        })
    return pd.Series(result, index=index, name=source.name), audits
