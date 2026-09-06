#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""导出中国节假日特征 CSV（方案 B 审计兜底，2026-09-01）。

与 data_loading/calendar_generator/calendar_features.py 共用同一核心实现，产物用于：
1. 人工核对 generated source 的在线特征（审计兜底）；
2. 需要版本固定的场景直接作为 canonical file source 的三段路径数据。

用法（项目根）::

    env -u PYTHONPATH .venv/bin/python \\
        scripts/export_chinese_holiday_csv.py --start 2025-10-01 --end 2026-12-31 \\
        --output dataset/shared/holidays/chinese_holiday_20251001_20261231.csv
    # --freq 1D|1h|15min|5min（默认 1D；日内频率同日多点继承当日状态）

列：time, is_holiday, holiday_name, next_holiday_days, prev_holiday_days,
is_adjusted_workday, solar_term（与 generated source 一致，
file source 引用时另加 available_at 列由 config 声明或按 source_time 政策处理）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# 脚本目录不在包内，引导仓库根后复用 canonical 实现（单一事实来源）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loading import chinese_holiday_frame  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="闭区间起始日，如 2025-10-01")
    parser.add_argument("--end", required=True, help="闭区间结束日，如 2026-12-31")
    parser.add_argument("--output", required=True, help="输出 CSV 路径")
    parser.add_argument("--freq", default="1D", help="日历网格频率（默认 1D）")
    args = parser.parse_args()

    # CLI --end 为人类友好的包含语义；帧合同是 end 排他，+1 天换算。
    end_exclusive = str(pd.Timestamp(args.end) + pd.Timedelta(days=1))[:10]
    frame = chinese_holiday_frame(args.start, end_exclusive, freq=args.freq)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    holidays = int(frame["is_holiday"].sum())
    named = int((frame["holiday_name"] != "").sum())
    print(f"exported {len(frame)} rows -> {output}")
    print(f"holiday points: {holidays}, named holiday points: {named}")
    print(f"date range: {frame['time'].min()} .. {frame['time'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
