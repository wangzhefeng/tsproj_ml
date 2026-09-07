"""离线数值特征编码；最后观测估计与缺失标记必须共同消费。"""
import numpy as np
import pandas as pd


def encode_features(frame: pd.DataFrame, *, time_col: str, targets: list[str], columns: list[str]) -> pd.DataFrame:
    """保留行序及观测值，不使用未来行；无历史时0仅为编码占位。"""
    if frame.empty or not frame.columns.is_unique or not targets or not columns:
        raise ValueError('nonempty frame, targets and columns with unique schema required')
    if len(set(columns)) != len(columns) or set(columns) & {time_col, *targets}:
        raise ValueError('encoding columns must be unique and exclude time/targets')
    if not {time_col, *targets, *columns}.issubset(frame.columns):
        raise ValueError('encoding input columns absent')
    times = pd.DatetimeIndex(pd.to_datetime(frame[time_col], errors='raise'))
    if times.hasnans or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError('encoding requires unique increasing timestamps')
    for target in targets:
        if not np.isfinite(np.asarray(pd.to_numeric(frame[target], errors='raise'), dtype=float)).all():
            raise ValueError('target must be finite; target imputation forbidden')
    markers = [column + suffix for column in columns for suffix in ('__missing', '__no_history')]
    if set(markers) & set(frame.columns):
        raise ValueError('encoding marker collision')
    result = frame.copy(deep=True)
    indicators = {}
    for column in columns:
        observed = pd.Series(pd.to_numeric(frame[column], errors='raise'), index=frame.index)
        if np.isinf(observed.to_numpy(dtype=float)).any():
            raise ValueError('infinite feature is invalid, not a missing value')
        missing = observed.isna()
        estimated = observed.ffill()
        result[column] = estimated.fillna(0.0)
        indicators[column + '__missing'] = missing.astype('int64')
        indicators[column + '__no_history'] = estimated.isna().astype('int64')
    return pd.concat([result, pd.DataFrame(indicators, index=frame.index)], axis=1)
