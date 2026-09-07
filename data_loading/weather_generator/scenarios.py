"""显式上一自然年映射；月末映射保持自然月，不推断丢失日期。"""
import pandas as pd
from data_loading.weather_generator.contracts import WeatherSnapshot


def combine_historical_snapshots(snapshots, data_kind, origin):
    """只组合原点可见的同物理合同历史分片；任何重叠都不选版本。"""
    selected = [s for s in snapshots if s.metadata['data_kind'] == data_kind]
    if len(selected) < 2:
        return selected
    first = selected[0]
    fields = ('source_id','location_id','latitude','longitude','coordinate_system','product','model','timezone','variables')
    ids = [s.metadata['snapshot_id'] for s in selected]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate historical snapshot identity')
    for snapshot in selected[1:]:
        if any(snapshot.metadata[k] != first.metadata[k] for k in fields):
            raise ValueError('historical partitions have incompatible physical contracts')
    visible = pd.concat([s.frame if origin is None else s.frame.loc[s.frame.available_at <= origin]
                         for s in selected], ignore_index=True)
    if visible.duplicated(['time','variable']).any():
        raise ValueError('historical partition overlap; no keep-last version selection')
    dependencies = tuple(dict.fromkeys(dep for s in selected for dep in s.dependency_hashes))
    return [WeatherSnapshot(first.metadata,visible,dependencies,tuple(ids))]


def proxy_time(time, timezone, leap_day, freq):
    local = time.tz_convert(timezone)
    if local.month == 2 and local.day == 29 and freq != '1ME' and leap_day == 'reject':
        raise ValueError('prior-year leap day has no exact counterpart')
    mapped = local - pd.DateOffset(years=1)
    if freq == '1ME':
        mapped = mapped + pd.offsets.MonthEnd(0)
    return mapped.tz_convert('UTC')
