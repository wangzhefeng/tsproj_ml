# -*- coding: utf-8 -*-
"""endogenous_strategy_actual/raw/ 三段 PCS 数据拼接为单文件。

输入: dataset/aidc_ess_selfuse_load/endogenous_strategy_actual/raw/PCSMerged_{A,B}_5min_*.csv（3 段）
输出: dataset/aidc_ess_selfuse_load/endogenous_strategy_actual/{A,B}_PCSMerged_5min_20251001_20260728.csv

处理规则：
  - 按时间排序、去重（keep last）
  - 截断到 2026-07-28 23:55（与站用电 now_time 对齐，PCS 原始数据多出的 3 天是下载过程混入的）

用法（仓库根目录）：
    env -u PYTHONPATH .venv/bin/python config/aidc_ess_selfuse_load/merge_strategy_actual_raw.py
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "dataset/aidc_ess_selfuse_load/endogenous_strategy_actual"
CUTOFF = pd.Timestamp("2026-07-28 23:55:00")


def merge_route(gate: str) -> None:
    raw_dir = DATA_DIR / "raw"
    segments = sorted(raw_dir.glob(f"PCSMerged_{gate}_5min_*.csv"))
    if not segments:
        raise FileNotFoundError(f"{raw_dir} 下无 PCSMerged_{gate} 段文件")

    frames = []
    for seg in segments:
        df = pd.read_csv(seg)
        df.columns = df.columns.str.strip("﻿")  # 去 BOM
        frames.append(df)
        print(f"  {seg.name}: {len(df)} 行")

    merged = pd.concat(frames, ignore_index=True)
    merged["time"] = pd.to_datetime(merged["time"])
    merged = merged.sort_values("time").drop_duplicates("time", keep="last")

    # 截断到 07-28（PCS 原始数据到 07-31，多出 3 天为下载混入）
    before = len(merged)
    merged = merged[merged["time"] <= CUTOFF]
    dropped = before - len(merged)
    if dropped:
        print(f"  截断 07-28 后数据: 删除 {dropped} 行（{dropped / 288:.0f} 天）")

    out_path = DATA_DIR / f"{gate}_PCSMerged_5min_20251001_20260728.csv"
    merged.to_csv(out_path, index=False)
    print(f"  -> {out_path.name}: {len(merged)} 行, {merged['time'].min()} ~ {merged['time'].max()}")


if __name__ == "__main__":
    for gate in ("A", "B"):
        print(f"=== {gate} 路 ===")
        merge_route(gate)
    print("Done.")
