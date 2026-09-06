"""中国节假日 canonical generated source（chinese-calendar 后端）。

设计（2026-09-01 方案 A+B 组合，wangzf 裁决）：
- 方案 A（默认）：注册为 builtin generator，YAML 声明
  ``source_type: generated + generator: chinese_holiday`` 即可入模，
  走 canonical known_future 编译与 VisibilityProof 留痕通路。
- 方案 B（审计兜底）：``scripts/export_chinese_holiday_csv.py`` 用同一
  实现导出 CSV，供人工核对 / file source 引用。

合同（与 SourceRegistry 生成通路校验严格对齐）：
- 帧必须恰好覆盖 ``request.forecast_times``（known_future 逐点精确匹配，
  缺行/多行都 RAISE）。稀疏/月末网格按对应日期取日状态，不代表月度聚合。
- ``chinese_holiday_frame`` 为 **end 排他**（``inclusive="left"``，与仓库
  时间边界约定一致）；CLI 导出脚本的 ``--end`` 为人类友好的包含语义。
- 帧携带 ``available_at`` 列（generated 校验强制）；值取 forecast_origin
  ——保留 calendar-known 假设；未核验历史公告版本，不是实际发布时间证明。
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

口径与边界：
- is_holiday 表示休息日，包含普通周末；holiday_name 仅命名节日非空。
- next_holiday_days / prev_holiday_days 是距休息日距离，休息日为 0；
  超出已知年历时保留删失哨兵 400，不改为距命名法定节日距离。
- 另输出 is_adjusted_workday 与 solar_term；节气使用依赖库公开算法，
  非节气日继承上一节气（例如 2026-08-06 为大暑，08-07 为立秋）。
- source 参数保留统一 SourceGenerator 合同；输出时间列固定为 time，
  声明 source 时应使用 time_col: time。具体列投影由 registry 完成。
- 计算与 CSV 导出复用 calendar_features.chinese_holiday_frame，不复制算法。
"""
from __future__ import annotations

import pandas as pd

from data_loading.calendar_generator.calendar_features import chinese_holiday_frame
from data_loading.information.information_set import InformationSetRequest
from forecasting_core.specs.data import DataSourceSpec

GENERATOR_NAME = "chinese_holiday"
_AVAILABLE_AT = "available_at"


def chinese_holiday_generator(
    source: DataSourceSpec,
    request: InformationSetRequest,
) -> pd.DataFrame:
    """只计算请求涉及的自然日，再逐点映射回原预测网格。"""
    forecast_times = request.forecast_times
    first_day = pd.Timestamp(forecast_times[0]).normalize()
    last_day = pd.Timestamp(forecast_times[-1]).normalize()
    # 使用自然日偏移；不让日内时分秒额外跨到下一日期/年历。
    frame = chinese_holiday_frame(first_day, last_day + pd.DateOffset(days=1), freq="1D")
    day_to_row = {timestamp.date(): index for index, timestamp in enumerate(frame["time"])}
    expanded = frame.iloc[
        [day_to_row[timestamp.date()] for timestamp in forecast_times]
    ].reset_index(drop=True)
    expanded["time"] = forecast_times.to_numpy()
    expanded[_AVAILABLE_AT] = pd.Timestamp(request.forecast_origin)
    return expanded
