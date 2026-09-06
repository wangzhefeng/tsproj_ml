"""中国节假日/节气纯日历属性计算，区间 [start, end)。"""
from __future__ import annotations

import bisect
import datetime
from typing import Any

import pandas as pd
import chinese_calendar as cc

# next_holiday_days 兜底上限：按年取全集后理论可达，仅防御异常数据。
_MAX_HORIZON_DAYS = 400
# 删失哨兵：下一假日超出已知年历时的确定性标记值（有文档的截断标记）。
_CENSORED_NEXT_HOLIDAY_DAYS = 400

def _sorted_holiday_dates(dates: list[datetime.date]) -> list[datetime.date]:
    """取 dates 覆盖年份及其次年的休息日全集（含普通周末与调休连休）。

    次年年历可能尚未由国务院发布/库未收录（NotImplementedError），
    防御性跳过——next_holiday_days 的删失哨兵见 chinese_holiday_frame。
    """
    years = {*(date.year for date in dates), *(date.year + 1 for date in dates)}
    holiday_set: set[datetime.date] = set()
    for year in sorted(years):
        try:
            holiday_set.update(
                cc.get_holidays(datetime.date(year, 1, 1), datetime.date(year, 12, 31))
            )
        except NotImplementedError:
            continue
    return sorted(holiday_set)


def chinese_holiday_frame(
    start: Any,
    end: Any,
    *,
    freq: str = "1D",
) -> pd.DataFrame:
    """按日历网格逐点生成节假日特征帧。

    区间为 ``[start, end)``（end 排他）；``freq`` 支持 ``1D`` 与
    ``1h``/``15min``/``5min`` 等日内频率（同日全部点继承当日状态）。
    """
    grid = pd.date_range(start, end, freq=freq, inclusive="left")
    if grid.empty:
        raise ValueError("chinese_holiday generator requires a non-empty date range")
    # 逐点取 date() 后去重：同一自然日共享一行日历状态。
    dates = sorted({ts.date() for ts in grid})

    day_cache: dict[datetime.date, tuple[int, str]] = {}
    for date in dates:
        on_holiday, name = cc.get_holiday_detail(date)
        # 调休班日（周末上班）detail 返回 (False, 所属节日名)，清空名字以
        # 维持「holiday_name 非空 => is_holiday=1」单向不变式；反向不成立
        # ——补假的周末 detail 可为 (True, None)（日历未指名归属）。
        day_cache[date] = (
            int(bool(on_holiday)),
            str(name) if (on_holiday and name) else "",
        )
    # next_holiday_days / prev_holiday_days：距最近假日（含当日）的日历
    # 日数；假日为 0，非假日向前/向后找（跨周末，体现节前渐近与节后恢复）。
    # 按年取全集 + 双向 bisect。删失语义：请求年次年/上年的年历若尚未入库
    # （国务院通常年底发布次年安排），取哨兵值 _CENSORED_NEXT_HOLIDAY_DAYS
    # ——有文档的「已知截断」标记，不是编造的距离。
    holiday_days = _sorted_holiday_dates(dates)
    next_day_distance: dict[datetime.date, float] = {}
    prev_day_distance: dict[datetime.date, float] = {}
    for date in dates:
        if day_cache[date][0] == 1:
            next_day_distance[date] = 0.0
            prev_day_distance[date] = 0.0
            continue
        position = bisect.bisect_left(holiday_days, date)
        if position < len(holiday_days):
            distance = (holiday_days[position] - date).days
            if distance <= _MAX_HORIZON_DAYS:
                next_day_distance[date] = float(distance)
            else:
                next_day_distance[date] = float(_CENSORED_NEXT_HOLIDAY_DAYS)
        else:
            next_day_distance[date] = float(_CENSORED_NEXT_HOLIDAY_DAYS)
        prev_position = bisect.bisect_right(holiday_days, date) - 1
        if prev_position >= 0:
            back = (date - holiday_days[prev_position]).days
            if back <= _MAX_HORIZON_DAYS:
                prev_day_distance[date] = float(back)
                continue
        prev_day_distance[date] = float(_CENSORED_NEXT_HOLIDAY_DAYS)

    # is_adjusted_workday：周末但国务院安排上班（调休班日）。非周末恒 0。
    adjusted_workday: dict[datetime.date, int] = {}
    for date in dates:
        if date.weekday() < 5:
            adjusted_workday[date] = 0
        else:
            # 同一日期已由 get_holiday_detail 校验；不得吞掉异常伪造工作状态。
            adjusted_workday[date] = int(cc.is_workday(date))

    # 一次请求只构建一张节气表；用公开 API 处理闰年/世纪修正，避免复制公式。
    # 扩展到前一年，使一月初可继承上一年的冬至；无需计算请求之后的年份。
    terms = cc.get_solar_terms(
        datetime.date(dates[0].year - 1, 1, 1),
        datetime.date(dates[-1].year, 12, 31),
    )
    term_dates = [day for day, _ in terms]
    solar_term_cache = {
        day: terms[bisect.bisect_right(term_dates, day) - 1][1]
        for day in dates
    }

    return pd.DataFrame(
        {
            "time": grid,
            # 日内频率：grid 每点按其自然日映射回日值状态。
            "is_holiday": [day_cache[ts.date()][0] for ts in grid],
            "holiday_name": [day_cache[ts.date()][1] for ts in grid],
            "next_holiday_days": [next_day_distance[ts.date()] for ts in grid],
            "prev_holiday_days": [prev_day_distance[ts.date()] for ts in grid],
            "is_adjusted_workday": [adjusted_workday[ts.date()] for ts in grid],
            "solar_term": [solar_term_cache[ts.date()] for ts in grid],
        }
    )
