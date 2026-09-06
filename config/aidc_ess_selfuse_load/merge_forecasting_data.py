# -*- coding: utf-8 -*-
"""forecasting_data/ 下 ESS+PCS 拼接数据生成。

将站用电清洗数据（forecasting_data/{A,B}_GateEnergys_5min_*_remove_outlier.csv）
与 PCS 实际充放电清洗数据（endogenous_strategy_actual/outlier_analysis/{A,B}_PCSMerged_5min_*_remove_outlier.csv）
按时间列内连接拼接，输出多变量预测的目标+内生特征文件。

输出: dataset/aidc_ess_selfuse_load/forecasting_data/{A,B}_ESS_PCS_merged_5min_20251001_20260728.csv
列: time, ess_power（站用电 kW）, pcs_power（PCS 充放电功率 kW，负=充电，正=放电）

用法（仓库根目录）：
    env -u PYTHONPATH .venv/bin/python config/aidc_ess_selfuse_load/merge_forecasting_data.py
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "dataset/aidc_ess_selfuse_load"


def merge_route(gate: str) -> None:
    ess_path = DATA_DIR / f"forecasting_data/{gate}_GateEnergys_5min_20251001_20260728_remove_outlier.csv"
    pcs_path = DATA_DIR / f"endogenous_strategy_actual/outlier_analysis/{gate}_PCSMerged_5min_20251001_20260728_remove_outlier.csv"

    ess = pd.read_csv(ess_path)
    ess.columns = ess.columns.str.strip("﻿")
    ess["time"] = pd.to_datetime(ess["time"])
    ess = ess.rename(columns={"value": "ess_power"})

    pcs = pd.read_csv(pcs_path)
    pcs.columns = pcs.columns.str.strip("﻿")
    pcs["time"] = pd.to_datetime(pcs["time"])
    pcs = pcs.rename(columns={"value": "pcs_power"})

    # 内连接：站用电末点 07-28 23:55 决定拼接末点
    merged = ess.merge(pcs, on="time", how="inner")
    merged = merged.sort_values("time").reset_index(drop=True)

    out_path = DATA_DIR / f"forecasting_data/{gate}_ESS_PCS_merged_5min_20251001_20260728.csv"
    merged.to_csv(out_path, index=False)

    nan_cnt = merged.isna().sum().sum()
    print(f"  {gate}: ess={len(ess)} 行, pcs={len(pcs)} 行, merged={len(merged)} 行, NaN={nan_cnt}")
    print(f"    {merged['time'].min()} ~ {merged['time'].max()}")
    print(f"    ess_power: min={merged['ess_power'].min():.1f}, max={merged['ess_power'].max():.1f}")
    print(f"    pcs_power: min={merged['pcs_power'].min():.1f}, max={merged['pcs_power'].max():.1f}")
    print(f"  -> {out_path.name}")


if __name__ == "__main__":
    for gate in ("A", "B"):
        merge_route(gate)
    print("Done.")
