# -*- coding: utf-8 -*-
"""exogenous_strategy_plan/raw/ 计划数据上采样到 5min。

输入: exogenous_strategy_plan/raw/strategy_{A,B}_in_20251001_20260812.csv（日级调度表，15min/5min 混合频率）
输出: exogenous_strategy_plan/up_sampled/{A,B}_PCS_plan_5min_20251001_20260731.csv（5min 全网格）

处理规则：
  1. 展开为时间序列：effective_date + s_time → time，power → pcs_plan
  2. 早期人工设计段（≤2025-10-09 22:00，15min 频率）符号修正为负充正放：
     充电 → -abs(power)，放电 → +abs(power)，待机 → 0
  3. 上采样：15min 槽位的值覆盖其后 5min 间隙（ffill，不产生虚假中间值）
  4. 后段零星缺失填 0（无计划=待机）
  5. 截断到 2026-07-31 23:55（up_sampled 覆盖到 PCS 原始数据末点）

用法（仓库根目录）：
    uv run python config/aidc_ess_selfuse_load/upsample_strategy_plan.py
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "dataset/aidc_ess_selfuse_load/exogenous_strategy_plan"
EARLY_END = pd.Timestamp("2025-10-09 22:00:00")  # 人工设计段（15min 频率）
CUTOFF = pd.Timestamp("2026-07-31 23:55:00")     # up_sampled 覆盖末点


def upsample_route(gate: str) -> None:
    src = DATA_DIR / f"raw/strategy_{gate}_in_20251001_20260812.csv"
    s = pd.read_csv(src)
    s.columns = s.columns.str.strip("﻿")
    s["time"] = pd.to_datetime(s["effective_date"] + " " + s["s_time"])
    s = s.sort_values("time").drop_duplicates("time", keep="last")
    print(f"  原始: {len(s)} 条, {s['time'].min()} ~ {s['time'].max()}")

    # --- 早期人工段符号修正（负充正放，对齐 10-09 后真实段约定） ---
    early_mask = s["time"] <= EARLY_END
    ep = s.loc[early_mask, "power"].copy()
    lab = s.loc[early_mask, "strategy"]
    fixed = ep.copy()
    fixed[lab == "充电"] = -ep[lab == "充电"].abs()
    fixed[lab == "放电"] = ep[lab == "放电"].abs()
    fixed[lab == "待机"] = 0.0
    n_flip = int((fixed != ep).sum())
    s.loc[early_mask, "power"] = fixed
    print(f"  早期段符号修正: {n_flip} 点")

    # --- 上采样到 5min：ffill 前向填充 ---
    full_idx = pd.date_range(s["time"].min(), s["time"].max(), freq="5min")
    ts = s.set_index("time")["power"].reindex(full_idx)
    ts = ts.ffill()  # 15min 值覆盖其后 5min 间隙
    ts = ts.fillna(0.0)  # 残余 NaN（序列开头，实际无）

    # --- 截断到 07-31 ---
    ts = ts[ts.index <= CUTOFF]
    ts.index.name = "time"
    df = ts.reset_index().rename(columns={"power": "pcs_plan"})

    out_path = DATA_DIR / f"up_sampled/{gate}_PCS_plan_5min_20251001_20260731.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"  -> up_sampled/{out_path.name}: {len(df)} 行")
    vc = df["pcs_plan"].value_counts()
    print(f"     负值(充电): {(df['pcs_plan'] < 0).sum()}, 零值(待机): {(df['pcs_plan'] == 0).sum()}, 正值(放电): {(df['pcs_plan'] > 0).sum()}")


if __name__ == "__main__":
    for gate in ("A", "B"):
        print(f"=== {gate} 路 ===")
        upsample_route(gate)
    print("Done.")
