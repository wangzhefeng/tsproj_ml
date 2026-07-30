# -*- coding: utf-8 -*-
"""AIDC 5min 负荷缺失填充方法回测。

掩码真实观测点，用 data_process.data_aggregate 的 _FILLERS 注册表方法填充，
按缺口长度分桶对比 MAPE。新增填充方法加入注册表后自动纳入回测。

用法（仓库根目录）：
    uv run python data_process/fill_method_backtest.py
    uv run python data_process/fill_method_backtest.py --route A --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_process.data_aggregate import _FILLERS  # noqa: E402

DATASET_DIR = PROJECT_ROOT / "dataset" / "aidc_power"
DATE_RANGE = "20251001_20260728"
SOURCE_FREQ_MINUTES = 5
SLOTS_PER_DAY = 24 * 60 // SOURCE_FREQ_MINUTES  # 288

# 缺口形状：槽数 -> 随机起点样本数（None 表示用固定步长穷举）
GAP_SHAPES = {1: 300, 3: 150, 12: 80, 288: None}
FULL_DAY_STEP_DAYS = 5


def load_series(route: str) -> pd.Series:
    """读取一路 5min 负荷，规则化到 5min 网格并补齐为连续真值基准。"""
    frame = pd.read_csv(DATASET_DIR / f"{route}_Loads_5min_{DATE_RANGE}.csv")
    frame["time"] = pd.to_datetime(frame["time"])
    series = frame.sort_values("time").set_index("time")["value"].resample("5min").mean()
    return series.interpolate(method="time", limit_direction="both")


def mask_starts(length: int, n: int, margin: int, samples: int | None, rng) -> np.ndarray:
    """生成掩码起点：固定样本数随机取，或（全天缺口）按步长穷举。"""
    if samples is None:
        return np.arange(margin, n - margin - length, FULL_DAY_STEP_DAYS * SLOTS_PER_DAY)
    return rng.integers(margin, n - margin - length, samples)


def backtest_route(route: str, seed: int, margin_weeks: int) -> None:
    truth = load_series(route)
    n = len(truth)
    margin = margin_weeks * 7 * SLOTS_PER_DAY
    rng = np.random.default_rng(seed)

    print(f"=== {route} 路（{n} 槽, margin ±{margin_weeks} 周, seed {seed}）===")
    for length, samples in GAP_SHAPES.items():
        starts = mask_starts(length, n, margin, samples, rng)
        errors = {name: [] for name in _FILLERS}
        for start in starts:
            positions = np.arange(start, start + length)
            masked = truth.copy()
            masked.iloc[positions] = np.nan
            true_values = truth.iloc[positions].to_numpy()
            for name, filler in _FILLERS.items():
                filled = filler(masked, margin_weeks).iloc[positions].to_numpy()
                if np.isnan(filled).any():
                    continue
                errors[name].append(float(np.mean(np.abs(filled - true_values) / true_values) * 100))
        label = f"{length} 槽" if length < 288 else "288 槽(1天)"
        print(f"--- 缺口 {label}:")
        for name, values in errors.items():
            print(f"   {name:<14} MAPE={np.mean(values):6.3f}%  (n={len(values)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="AIDC 负荷缺失填充方法回测")
    parser.add_argument("--route", choices=["A", "B", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin-weeks", type=int, default=4,
                        help="掩码位置避开首尾的周数（seasonal_slot 的上下文需求）")
    args = parser.parse_args()

    routes = ("A", "B") if args.route == "both" else (args.route,)
    for route in routes:
        backtest_route(route, args.seed, args.margin_weeks)


if __name__ == "__main__":
    main()
