"""可审计的日历条件冷启动；所有水平仅由预测原点之前数据估计。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_loading import chinese_holiday_frame
from data_loading.calendar_generator.named_holidays import named_holiday_frame, spring_festival_bounds

DEFAULT_RECIPE = Path(__file__).with_name('annual_recipe.json')


def validate_recipe(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != {'recipe_version', 'cold_start'} or value['recipe_version'] != 1:
        raise ValueError('annual recipe requires recipe_version=1 and cold_start only')
    spec = value['cold_start']
    if not isinstance(spec, dict) or set(spec) != {'method', 'normal_floor_ratio', 'transition_days'}:
        raise ValueError('cold_start requires method, normal_floor_ratio, transition_days')
    if spec['method'] not in ('calendar_baseline', 'calendar_pointwise'):
        raise ValueError('unsupported non-recursive cold_start method')
    floor, transition = spec['normal_floor_ratio'], spec['transition_days']
    if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not np.isfinite(floor) or not 0 < floor <= 1:
        raise ValueError('normal_floor_ratio must be in (0,1]')
    if isinstance(transition, bool) or not isinstance(transition, int) or not 0 <= transition <= 14:
        raise ValueError('transition_days must be integer in [0,14]')
    return {'recipe_version': 1, 'cold_start': dict(spec)}


def load_recipe(path: Path = DEFAULT_RECIPE) -> dict:
    return validate_recipe(json.loads(Path(path).read_text()))


def calendar_baseline(history: pd.DataFrame, times: pd.DatetimeIndex, spec: dict) -> tuple[np.ndarray, dict]:
    if history.empty or not (history.time < times[0]).all() or not np.isfinite(history.value).all():
        raise ValueError('cold_start needs finite strictly historical observations')
    calendar = chinese_holiday_frame(history.time.min(), history.time.max() + pd.Timedelta(days=1))
    frame = history.merge(calendar[['time', 'holiday_name']], on='time', validate='one_to_one')
    ordinary = frame.loc[frame.holiday_name == '', 'value']
    festive = frame.loc[frame.holiday_name == 'Spring Festival', 'value']
    if ordinary.empty or festive.empty:
        raise ValueError('calendar baseline requires past ordinary and Spring Festival observations')
    reference = float(ordinary.median())
    normal = frame[(frame.holiday_name == '') & (frame.value >= spec['normal_floor_ratio'] * reference)]
    if normal.empty:
        raise ValueError('no historical normal-level reference')
    normal_level = float(normal.value.median())
    holiday_level = float(festive.median())
    weekday_levels = {}
    for weekday in range(7):
        group = normal[normal.time.dt.dayofweek == weekday].value
        weekday_levels[weekday] = float(group.median()) if len(group) >= 2 else normal_level
    features = named_holiday_frame(times)
    values = []
    for timestamp, is_spring in zip(times, features.is_spring_festival):
        normal_value = weekday_levels[timestamp.dayofweek]
        if is_spring:
            value = holiday_level
        else:
            _, end = spring_festival_bounds(timestamp.year)
            elapsed = (timestamp.date() - end).days
            transition = spec['transition_days']
            if transition and 0 < elapsed < transition:
                fraction = elapsed / transition
                value = holiday_level + fraction * (normal_value - holiday_level)
            else:
                value = normal_value
        values.append(value)
    return np.asarray(values, dtype=float), {
        'normal_reference': normal_level, 'spring_festival_reference': holiday_level,
        'normal_observations': len(normal), 'holiday_observations': len(festive),
        'weekday_levels': weekday_levels, 'parameters': spec,
        'assumption': 'official calendar transition is a forecasting prior, not enterprise restart facts',
    }
