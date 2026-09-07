"""天气时间对齐；不以插值掩盖原生采样缺口。"""
import pandas as pd
import numpy as np


def circular_mean(values):
    radians = np.deg2rad(np.asarray(values, dtype=float))
    if radians.size == 0 or not np.isfinite(radians).all():
        raise ValueError('circular mean needs finite directions')
    sine, cosine = np.sin(radians).mean(), np.cos(radians).mean()
    if np.hypot(sine, cosine) <= 1e-12:
        raise ValueError('wind direction mean has ambiguous zero resultant')
    angle = float(np.rad2deg(np.arctan2(sine, cosine)) % 360.)
    return 0. if abs(angle - 360.) <= 1e-12 else angle


def circular_difference(first, last):
    delta = (float(last) - float(first) + 180.) % 360. - 180.
    if abs(delta) == 180.:
        raise ValueError('opposite wind directions have ambiguous difference')
    return delta


def convert_unit(value, source, target):
    if source == target:
        return value
    conversions = {
        ('K', 'degC'): (1., -273.15), ('degC', 'K'): (1., 273.15),
        ('delta_K', 'delta_degC'): (1., 0.), ('delta_degC', 'delta_K'): (1., 0.),
        ('km/h', 'm/s'): (1 / 3.6, 0.), ('m/s', 'km/h'): (3.6, 0.),
        ('hPa', 'Pa'): (100., 0.), ('Pa', 'hPa'): (.01, 0.),
        ('J/m2', 'MJ/m2'): (1e-6, 0.), ('MJ/m2', 'J/m2'): (1e6, 0.),
    }
    if (source, target) not in conversions:
        raise ValueError(f'incompatible weather units: {source} -> {target}')
    scale, offset = conversions[source, target]
    return value * scale + offset


def point_value(part, time, metadata, temporal):
    indexed = part.set_index('time').sort_index()
    native = pd.Timedelta(metadata['native_freq'])
    if metadata['semantics'] != 'point':
        raise ValueError('point sampling requires point variable semantics')
    if time in indexed.index:
        row = indexed.loc[time]
        return float(row.value), row.available_at
    if temporal.upsampling == 'exact':
        raise ValueError(f'missing exact weather timestamp: {time}')
    previous = indexed.index[indexed.index < time]
    if previous.empty:
        raise ValueError('weather preceding endpoint missing')
    left = previous[-1]
    age = time - left
    if age >= native or age > pd.Timedelta(temporal.max_age):
        raise ValueError('weather hold/interpolation exceeds native validity or max_age')
    first = indexed.loc[left]
    if temporal.upsampling == 'hold':
        return float(first.value), first.available_at
    later = indexed.index[indexed.index > time]
    if later.empty or later[0] - left != native or later[0] - time > pd.Timedelta(temporal.max_age):
        raise ValueError('weather interpolation requires adjacent visible native endpoints')
    right = indexed.loc[later[0]]
    weight = age / native
    if metadata['unit'] == 'degree':
        delta = circular_difference(first.value, right.value)
        return (float(first.value) + weight * delta) % 360., max(first.available_at, right.available_at)
    return float(first.value * (1 - weight) + right.value * weight), max(first.available_at, right.available_at)


def request_interval(time, temporal):
    local = time.tz_convert(temporal.timezone)
    if temporal.freq in {'1ME', '1MS'}:
        start = local.normalize().replace(day=1)
        end = start + pd.offsets.MonthBegin(1)
        expected_label = end - pd.DateOffset(days=1) if temporal.freq == '1ME' else start
        if local != expected_label:
            raise ValueError('monthly weather label is not calendar boundary')
    elif temporal.freq == '1D':
        if local != local.normalize():
            raise ValueError('daily weather label must be midnight')
        start = local if temporal.label == 'left' else local - pd.DateOffset(days=1)
        end = start + pd.DateOffset(days=1)
    else:
        delta = pd.Timedelta(temporal.freq)
        start = local if temporal.label == 'left' else local - delta
        end = start + delta
    return start.tz_convert('UTC'), end.tz_convert('UTC')


def resampled_value(part, time, metadata, temporal, variable):
    if variable.aggregation == 'point':
        value, available = point_value(part, time, metadata, temporal)
        return convert_unit(value, metadata['unit'], variable.unit), available
    start, end = request_interval(time, temporal)
    step = pd.Timedelta(metadata['native_freq'])
    if metadata['semantics'] == 'point':
        if variable.aggregation not in {'mean', 'min', 'max'}:
            raise ValueError('point variables require explicit statistical aggregation')
        expected = pd.date_range(start, end, freq=step, inclusive='left')
        if temporal.closed == 'right':
            expected = expected + step
        indexed = part.set_index('time')
        if expected.empty or not expected.isin(indexed.index).all() or end - start != len(expected) * step:
            raise ValueError('incomplete native coverage for aggregate')
        rows = indexed.loc[expected]
        if metadata['unit'] == 'degree':
            if variable.aggregation != 'mean':
                raise ValueError('wind direction has no scalar extrema')
            value = circular_mean(rows.value)
        else:
            value = getattr(rows.value, variable.aggregation)()
        return convert_unit(float(value), metadata['unit'], variable.unit), rows.available_at.max()
    rows = part.copy()
    rows['start'] = rows.time - (step if metadata['label'] == 'right' else pd.Timedelta(0))
    rows['end'] = rows.start + step
    rows = rows.loc[(rows.start < end) & (rows.end > start)].sort_values('start')
    cursor = start
    durations = []
    for row in rows.itertuples():
        left, right = max(row.start, start), min(row.end, end)
        if left != cursor:
            raise ValueError('missing or overlapping native interval')
        if right - left < step and (temporal.upsampling != 'hold' or pd.Timedelta(temporal.max_age) < step):
            raise ValueError('interval splitting requires explicit hold rate and native max_age')
        durations.append((right - left).total_seconds())
        cursor = right
    if cursor != end or not durations:
        raise ValueError('incomplete interval weather coverage')
    weights = pd.Series(durations, index=rows.index)
    if metadata['semantics'] == 'interval_sum':
        if variable.aggregation != 'sum':
            raise ValueError('interval totals require sum, not mean')
        value = (rows.value * weights / step.total_seconds()).sum()
        source_unit = metadata['unit']
    elif variable.aggregation == 'integral' and metadata['unit'] == 'W/m2':
        value = (rows.value * weights).sum()
        source_unit = 'J/m2'
    elif variable.aggregation == 'mean':
        value = (rows.value * weights).sum() / weights.sum()
        source_unit = metadata['unit']
    elif variable.aggregation in {'min', 'max'}:
        value = getattr(rows.value, variable.aggregation)()
        source_unit = metadata['unit']
    else:
        raise ValueError('incompatible interval aggregation')
    return convert_unit(float(value), source_unit, variable.unit), rows.available_at.max()
