"""按显式声明提供命名节日数值特征，不将日历当作企业运行计划。"""
from datetime import date
from functools import lru_cache

import chinese_calendar as cc
import pandas as pd

NAMED_HOLIDAY_FEATURES = ('is_spring_festival', 'is_named_holiday',
                          'days_to_spring_festival', 'days_after_spring_festival')


@lru_cache(maxsize=32)
def spring_festival_bounds(year: int) -> tuple[date, date]:
    days = [day for day in cc.get_holidays(date(year, 1, 1), date(year, 12, 31))
            if cc.get_holiday_detail(day) == (True, 'Spring Festival')]
    if not days:
        raise ValueError(f'no supported Spring Festival calendar for {year}')
    return min(days), max(days)


def named_holiday_frame(times: pd.DatetimeIndex) -> pd.DataFrame:
    """距当年春节开始/结束的非负距离截断到31天；期间两距离均为0。"""
    rows = []
    for timestamp in pd.DatetimeIndex(times):
        day = timestamp.date()
        holiday, name = cc.get_holiday_detail(day)
        start, end = spring_festival_bounds(day.year)
        rows.append({'is_spring_festival': int(holiday and name == 'Spring Festival'),
                     'is_named_holiday': int(holiday and bool(name)),
                     'days_to_spring_festival': min(31, max(0, (start - day).days)),
                     'days_after_spring_festival': min(31, max(0, (day - end).days))})
    return pd.DataFrame(rows, columns=NAMED_HOLIDAY_FEATURES)
