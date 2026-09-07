"""纯气象派生，完整原生预热与全部依赖可得时间传播。"""
import copy

import numpy as np
import pandas as pd

from data_loading.weather_generator.resampling import convert_unit, circular_mean, circular_difference


def derive_native(frame, descriptions, features):
    result = frame.copy(deep=True)
    descriptions = copy.deepcopy(descriptions)
    for feature in features:
        if feature.name in descriptions or any(name not in descriptions for name in feature.inputs):
            raise ValueError('native feature conflicts or references missing/forward input')
        metadata = descriptions[feature.inputs[0]]
        if any(descriptions[name]['native_freq'] != metadata['native_freq'] or descriptions[name]['semantics'] != metadata['semantics'] or descriptions[name]['label'] != metadata['label'] for name in feature.inputs):
            raise ValueError('native feature dependencies require same grid and interval semantics')
        step = pd.Timedelta(metadata['native_freq'])
        parts = [result.loc[result.variable.eq(name)].set_index('time').sort_index() for name in feature.inputs]
        records = []
        if feature.operation == 'relative_humidity':
            if metadata['semantics'] != 'point':
                raise ValueError('relative humidity requires native point temperatures')
            for time in parts[0].index.intersection(parts[1].index):
                rows = [part.loc[time] for part in parts]
                air = convert_unit(float(rows[0].value), metadata['unit'], 'degC')
                dew = convert_unit(float(rows[1].value), descriptions[feature.inputs[1]]['unit'], 'degC')
                if dew > air or air <= -237.29 or dew <= -237.29:
                    raise ValueError('physically invalid temperature/dewpoint for RH')
                value = 100. * np.exp(17.2693 * dew / (237.29 + dew) - 17.2693 * air / (237.29 + air))
                records.append({'time': time, 'variable': feature.name, 'value': value, 'available_at': max(r.available_at for r in rows)})
            unit = '%'
        else:
            duration = pd.Timedelta(feature.window)
            if duration % step or duration < step:
                raise ValueError('native window must be a positive multiple of native frequency')
            part = parts[0]
            for time in part.index:
                start = time - duration + (step if feature.operation == 'rolling_mean' else pd.Timedelta(0))
                expected = pd.date_range(start, time, freq=step)
                if not expected.isin(part.index).all():
                    continue  # 不合格预热点不生成；请求触及这些点由覆盖校验 RAISE。
                rows = part.loc[expected]
                if metadata['unit'] == 'degree':
                    value = circular_mean(rows.value) if feature.operation == 'rolling_mean' else circular_difference(rows.value.iloc[0], rows.value.iloc[-1])
                else:
                    value = rows.value.mean() if feature.operation == 'rolling_mean' else rows.value.iloc[-1] - rows.value.iloc[0]
                records.append({'time': time, 'variable': feature.name, 'value': value, 'available_at': rows.available_at.max()})
            unit = metadata['unit']
            if feature.operation == 'difference' and unit in {'K', 'degC', 'degree'}:
                unit = 'delta_' + unit
        extra = pd.DataFrame(records, columns=['time', 'variable', 'value', 'available_at'])
        result = pd.concat([result, extra], ignore_index=True)
        descriptions[feature.name] = {**metadata, 'name': feature.name, 'column': feature.name, 'unit': unit}
    return result, descriptions
