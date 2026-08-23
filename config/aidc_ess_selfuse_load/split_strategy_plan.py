# -*- coding: utf-8 -*-
"""exogenous_strategy_plan/ 下 history/future 数据分割。

将 up_sampled/ 下的 5min 全量计划数据按 now_time（2026-07-28 23:55）分割为：
  - history: {A,B}_PCS_plan_5min_20251001_20260728.csv（~07-28 23:55，训练+滑窗测试用）
  - future:  {A,B}_PCS_plan_future_5min_20260729_20260731.csv（07-29~07-31，forecast 用）

用法（仓库根目录）：
    uv run python config/aidc_ess_selfuse_load/split_strategy_plan.py
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "dataset/aidc_ess_selfuse_load/exogenous_strategy_plan"
HISTORY_END = pd.Timestamp("2026-07-28 23:55:00")
FUTURE_START = pd.Timestamp("2026-07-29 00:00:00")
FUTURE_END = pd.Timestamp("2026-07-31 23:55:00")


def split_route(gate: str) -> None:
    src = DATA_DIR / f"up_sampled/{gate}_PCS_plan_5min_20251001_20260731.csv"
    df = pd.read_csv(src)
    df.columns = df.columns.str.strip("﻿")
    df["time"] = pd.to_datetime(df["time"])
    # 当前业务契约：目标日完整计划在前一日23:55前已经发布。若上游后续提供
    # 真实发布时间/版本，应直接透传真实 available_at，不再用该规则派生。
    df["available_at"] = df["time"].dt.normalize() - pd.Timedelta(minutes=5)

    hist = df[df["time"] <= HISTORY_END]
    fut = df[(df["time"] >= FUTURE_START) & (df["time"] <= FUTURE_END)]

    hist_path = DATA_DIR / f"{gate}_PCS_plan_5min_20251001_20260728.csv"
    fut_path = DATA_DIR / f"{gate}_PCS_plan_future_5min_20260729_20260731.csv"
    hist.to_csv(hist_path, index=False)
    fut.to_csv(fut_path, index=False)

    print(f"  {gate}: 全量 {len(df)} 行 → history {len(hist)} 行 + future {len(fut)} 行")
    print(f"    history: {hist['time'].min()} ~ {hist['time'].max()}")
    print(f"    future:  {fut['time'].min()} ~ {fut['time'].max()}")
    assert len(hist) + len(fut) == len(df), "分割行数不匹配"
    assert hist["pcs_plan"].notna().all() and fut["pcs_plan"].notna().all(), "存在 NaN"


if __name__ == "__main__":
    for gate in ("A", "B"):
        print(f"=== {gate} 路 ===")
        split_route(gate)
    print("Done.")
