# -*- coding: utf-8 -*-
"""中国节假日 canonical generated source（chinese-calendar 后端）。

设计（2026-09-01 方案 A+B 组合，wangzf 裁决）：
- 方案 A（默认）：注册为 builtin generator，YAML 声明
  ``source_type: generated + generator: chinese_holiday`` 即可入模，
  走 canonical known_future 编译与 VisibilityProof 留痕通路。
- 方案 B（审计兜底）：``scripts/export_chinese_holiday_csv.py`` 用同一
  实现导出 CSV，供人工核对 / file source 引用。

合同（与 SourceRegistry 生通路校验严格对齐）：
- 帧必须恰好覆盖 ``request.forecast_times``（known_future 逐点精确匹配，
  缺行/多行都 RAISE）；月频等非逐日网格不适用（逐点匹配 RAISE）。
- ``chinese_holiday_frame`` 为 **end 排他**（``inclusive="left"``，与仓库
  时间边界约定一致）；CLI 导出脚本的 ``--end`` 为人类友好的包含语义。
- 帧携带 ``available_at`` 列（generated 校验强制）；值取 forecast_origin
  ——节假日日历为天然可知信息，逐日值不依赖获取时刻。
- 覆盖范围外直接 RAISE（chinese-calendar 自身行为，2004 起、按年扩展），
  不得静默降级。

YAML 用法（spec 层强约束：generated known_future 必须 generator_defined）::

    data:
      sources:
      - name: chinese_holiday
        source_type: generated
        generator: chinese_holiday
        columns:
        - name: is_holiday
          role: known_future
          categorical: false
        - name: holiday_name
          role: known_future
          categorical: true
        - name: next_holiday_days
          role: known_future
          categorical: false
        time_col: time
        availability: generator_defined
"""

from __future__ import annotations

import bisect
import datetime
from typing import Any

import pandas as pd

_GENERATOR_NAME = "chinese_holiday"
_AVAILABLE_AT = "available_at"
# next_holiday_days 兜底上限：按年取全集后理论可达，仅防御异常数据。
_MAX_HORIZON_DAYS = 400
# 删失哨兵：下一假日超出已知年历时的确定性标记值（有文档的截断标记）。
_CENSORED_NEXT_HOLIDAY_DAYS = 400


def generator_name() -> str:
    """builtin 注册名。"""
    return _GENERATOR_NAME


def _sorted_holiday_dates(dates: list[datetime.date]) -> list[datetime.date]:
    """取 dates 覆盖年份及其次年的法定假日全集（含调休连休）。

    次年年历可能尚未由国务院发布/库未收录（NotImplementedError），
    防御性跳过——next_holiday_days 的删失哨兵见 chinese_holiday_frame。
    """
    import chinese_calendar as cc

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


# 节气→公历月份映射（每个节气的月份固定：1 月小寒/大寒……12 月大雪/冬至）。
SOLAR_TERM_MONTH_ORDER = {
    "the Beginning of Spring": 2,
    "Rain Water": 2,
    "the Waking of Insects": 3,
    "the Spring Equinox": 3,
    "Pure Brightness": 4,
    "Grain Rain": 4,
    "the Beginning of Summer": 5,
    "Lesser Fullness of Grain": 5,
    "Grain in Beard": 6,
    "the Summer Solstice": 6,
    "Lesser Heat": 7,
    "Greater Heat": 7,
    "the Beginning of Autumn": 8,
    "the End of Heat": 8,
    "White Dew": 9,
    "the Autumn Equinox": 9,
    "Cold Dew": 10,
    "Frost's Descent": 10,
    "the Beginning of Winter": 11,
    "Lesser Snow": 11,
    "Greater Snow": 12,
    "the Winter Solstice": 12,
    "Lesser Cold": 1,
    "Greater Cold": 1,
}


def _solar_term_table(years: set[int]) -> list[tuple[datetime.date, str]]:
    """计算给定年份集合（±1 年扩展）的全年 24 节气有序表 [(date, 中文名)]。"""
    from chinese_calendar.solar_terms import (
        SOLAR_TERMS_C_NUMS,
        SOLAR_TERMS_DELTA,
    )

    table: list[tuple[datetime.date, str]] = []
    for year in years:
        year_mod = year % 100
        century_offset = 1 if year >= 2001 else 0
        for term, c_values in SOLAR_TERMS_C_NUMS.items():
            month = SOLAR_TERM_MONTH_ORDER[term.value[0]]
            day = (
                int(year_mod * 0.2422 + c_values[century_offset])
                - int(year_mod / 4)
                + SOLAR_TERMS_DELTA.get((year, term), 0)
            )
            table.append((datetime.date(year, month, day), term.value[1]))
    return sorted(table)


def _solar_term_lookup(date: datetime.date) -> str:
    """当日所处的节气：节气日返回自身名，其余日继承上一节气名。

    语义：列值是「当前节令」（季节内漂移的解释变量），逐日继承；
    例：2026-08-06 -> 小暑，08-07 -> 立秋，08-08 -> 立秋。
    """
    padded_years = {date.year - 1, date.year, date.year + 1}
    table = _solar_term_table(padded_years)
    position = bisect.bisect_right([d for d, _ in table], date) - 1
    if position < 0:
        return ""
    return table[position][1]


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
    import chinese_calendar as cc

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
            try:
                adjusted_workday[date] = int(cc.is_workday(date))
            except NotImplementedError:
                adjusted_workday[date] = 0

    # solar_term：当日所处的节气（节气日返回自身，其余日继承上一节气）。
    # 季节内温度-负荷关系的漂移解释变量，天然已知（known_future）。
    solar_term_cache: dict[datetime.date, str] = {}
    for date in dates:
        solar_term_cache[date] = _solar_term_lookup(date)

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


def chinese_holiday_generator(
    source: Any,
    request: Any,
) -> pd.DataFrame:
    """SourceGenerator 签名入口：``(DataSourceSpec, InformationSetRequest)``。

    只产 known_future 需要的预测网格行，附 ``available_at`` 列。
    """
    forecast_times = request.forecast_times
    first = pd.Timestamp(forecast_times[0])
    last = pd.Timestamp(forecast_times[-1])
    # end 排他：日频 +1D 即可；日内频率由下方逐点展开继承当日状态。
    frame = chinese_holiday_frame(first, last + pd.Timedelta(days=1), freq="1D")

    # 日内频率：同一天的所有点继承当日状态；日频直接一一命中。
    frame_days = [ts.date() for ts in pd.to_datetime(frame["time"])]
    day_to_row = {day: index for index, day in enumerate(frame_days)}
    expanded = frame.iloc[
        [day_to_row[ts.date()] for ts in forecast_times]
    ].reset_index(drop=True)
    expanded["time"] = forecast_times.to_numpy()
    expanded[_AVAILABLE_AT] = pd.Timestamp(request.forecast_origin)
    return expanded


__all__ = [
    "chinese_holiday_frame",
    "chinese_holiday_generator",
    "generator_name",
]
