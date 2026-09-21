"""原始点位异常判定；只返回候选/标记，不填补，不读取派生值。"""
import numpy as np
import pandas as pd


AUDIT_COLUMNS = ['time', 'point', 'old_value', 'local_baseline', 'total_value',
                 'total_baseline', 'total_residual', 'neighbor_spread', 'amplitude_floor',
                 'point_residual', 'explained_fraction', 'detection_known_at', 'status']


def isolated_evidence(series):
    """前后各3槽必须完整且稳定；只有单槽离群，不清除短平台/阶跃。"""
    neighbors = pd.concat([series.shift(k) for k in (-3, -2, -1, 1, 2, 3)], axis=1)
    baseline = neighbors.median(axis=1)
    residual = series - baseline
    spread = neighbors.max(axis=1) - neighbors.min(axis=1)
    mad = neighbors.sub(baseline, axis=0).abs().median(axis=1)
    floor = baseline.abs().mul(0.10).clip(lower=1.0)
    accepted = (neighbors.notna().all(axis=1) & series.notna()
                & residual.abs().ge(floor) & residual.abs().ge(6 * 1.4826 * mad)
                & spread.le(residual.abs() * 0.25))
    return accepted, baseline, residual, spread, floor


def detect_isolated(points, start, end):
    if points.empty or not points.index.equals(pd.date_range(points.index[0], periods=len(points), freq='5min')):
        raise ValueError('需要非空连续5min点位表')
    values = points.to_numpy(dtype=float)
    if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
        raise ValueError('负荷不允许负数或无限值')
    total = points.sum(axis=1, min_count=len(points.columns))
    candidate, baseline, residual, spread, floor = isolated_evidence(total)
    candidate &= (points.index >= pd.Timestamp(start)) & (points.index <= pd.Timestamp(end))
    # 点位选择基于同一原始快照；不让已修正结果反过来触发第二轮清洗。
    point_evidence = {point: isolated_evidence(points[point]) for point in points}
    rows = []
    for time in points.index[candidate]:
        significant = [p for p, e in point_evidence.items() if bool(e[0].loc[time])]
        point = significant[0] if len(significant) == 1 else ''
        delta = float(point_evidence[point][2].loc[time]) if point else np.nan
        fraction = delta / float(residual.loc[time])
        status = 'accepted' if point and 0.80 <= fraction <= 1.20 else 'review_multiple_or_unexplained'
        rows.append({'time': str(time), 'point': point,
                     'old_value': float(points.loc[time, point]) if point else np.nan,
                     'local_baseline': float(point_evidence[point][1].loc[time]) if point else np.nan,
                     'total_value': float(total.loc[time]), 'total_baseline': float(baseline.loc[time]),
                     'total_residual': float(residual.loc[time]), 'neighbor_spread': float(spread.loc[time]),
                     'amplitude_floor': float(floor.loc[time]), 'point_residual': delta,
                     'explained_fraction': fraction, 'detection_known_at': str(time + pd.Timedelta(minutes=15)),
                     'status': status})
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def detect_candidates(source, events, recipe):
    """事件前固定1h基线，隔离先前已识别异常；不读取后侧观测定阈值。"""
    selected = pd.Series(False, index=source.index)
    audit = []
    observed = source.copy()
    previous_end = None
    for event in sorted(events, key=lambda row: row['start']):
        start, end = pd.Timestamp(event['start']), pd.Timestamp(event['end'])
        if start > end or (previous_end is not None and start <= previous_end):
            raise ValueError('事件区间必须有序且不重叠')
        previous_end = end
        index = source.index[(source.index >= start) & (source.index <= end)]
        if index.empty:
            continue
        before = observed.loc[(observed.index < start) &
                              (observed.index >= start - pd.Timedelta(minutes=5 * recipe['baseline_points']))].dropna()
        entry = {'event_start': str(start), 'event_end': str(end), 'reference_count': len(before)}
        if len(before) < recipe['minimum_baseline_points']:
            audit.append({**entry, 'status': 'insufficient_reference', 'selected': 0})
            continue
        baseline = float(before.median())
        mad = float((before - baseline).abs().median())
        threshold = max(recipe['absolute_threshold_kw'], abs(baseline) * recipe['relative_threshold'],
                        recipe['mad_multiplier'] * 1.4826 * mad)
        bad = source.loc[index].notna() & (source.loc[index] - baseline).abs().ge(threshold)
        times = index[bad]
        selected.loc[times] = True
        observed.loc[times] = np.nan
        audit.append({**entry, 'baseline': baseline, 'threshold': threshold,
                      'reference_start': str(before.index[0]), 'reference_end': str(before.index[-1]),
                      'status': 'screened', 'selected': len(times)})
    return selected, audit
