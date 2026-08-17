# -*- coding: utf-8 -*-
"""按 [当日22:00 ~ 次日21:55] 周期窗口绘制站用电 + PCS 实际/计划充放电曲线。

每个周期窗口输出一张图：
  - 左轴：站用电（ess_power，蓝色）
  - 右轴：PCS 实际充放电（红色）+ PCS 计划充放电（绿色虚线），零线标出

输入:
  forecasting_data/{A,B}_GateEnergys_5min_20251001_20260728_remove_outlier.csv
  endogenous_strategy_actual/outlier_analysis/{A,B}_PCSMerged_5min_20251001_20260728_remove_outlier.csv
  exogenous_strategy_plan/up_sampled/{A,B}_PCS_plan_5min_20251001_20260731.csv
输出:
  forecasting_data/daily_cycle_plots/{A,B}/<date>_cycle.png

用法（仓库根目录）：
    uv run python config/aidc_ess_selfuse_load/plot_daily_cycles.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "dataset/aidc_ess_selfuse_load"
OUT_DIR = DATA_DIR / "forecasting_data/daily_cycle_plots"

WINDOW_START_HOUR = 22  # 窗口起点：当日 22:00
WINDOW_POINTS = 288     # 5min × 288 = 24h


def load_route(gate: str):
    ess = pd.read_csv(DATA_DIR / f"forecasting_data/{gate}_GateEnergys_5min_20251001_20260728_remove_outlier.csv")
    ess["time"] = pd.to_datetime(ess["time"])
    ess = ess.rename(columns={"value": "ess_power"}).set_index("time")["ess_power"]

    pcs = pd.read_csv(DATA_DIR / f"endogenous_strategy_actual/outlier_analysis/{gate}_PCSMerged_5min_20251001_20260728_remove_outlier.csv")
    pcs["time"] = pd.to_datetime(pcs["time"])
    pcs = pcs.rename(columns={"value": "pcs_actual"}).set_index("time")["pcs_actual"]

    plan = pd.read_csv(DATA_DIR / f"exogenous_strategy_plan/up_sampled/{gate}_PCS_plan_5min_20251001_20260731.csv")
    plan["time"] = pd.to_datetime(plan["time"])
    plan = plan.set_index("time")["pcs_plan"]

    df = pd.concat([ess, pcs, plan], axis=1)
    return df


def plot_window(win: pd.DataFrame, date_str: str, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(14, 5), dpi=120)

    x = win.index
    ax1.plot(x, win["ess_power"], color="#1f77b4", lw=1.0, label="ESS self-use (left)")
    ax1.set_ylabel("ESS power (kW)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.plot(x, win["pcs_actual"], color="#d62728", lw=0.9, label="PCS actual (right)")
    ax2.plot(x, win["pcs_plan"], color="#2ca02c", lw=0.9, ls="--", label="PCS plan (right)")
    ax2.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax2.set_ylabel("PCS power (kW)  [neg=charge, pos=discharge]", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    # 合并图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, ncol=3)

    ax1.set_title(f"{date_str} 22:00 ~ next-day 21:55")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.set_xlabel("time of day")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def process_route(gate: str) -> None:
    df = load_route(gate)
    out_dir = OUT_DIR / gate
    out_dir.mkdir(parents=True, exist_ok=True)

    # 窗口起点：每日 22:00，从数据首日到最后一个完整窗口
    start = df.index.min().normalize() + pd.Timedelta(hours=WINDOW_START_HOUR)
    if df.index.min() > start:  # 首日 22:00 已过则从下一天开始
        start += pd.Timedelta(days=1)

    n_plot = n_skip = 0
    cur = start
    while cur + pd.Timedelta(minutes=5 * (WINDOW_POINTS - 1)) <= df.index.max():
        end = cur + pd.Timedelta(minutes=5 * (WINDOW_POINTS - 1))  # 次日 21:55
        win = df.loc[cur:end]
        if len(win) == WINDOW_POINTS and win["ess_power"].notna().all():
            date_str = cur.strftime("%Y-%m-%d")
            out_path = out_dir / f"{date_str}_cycle.png"
            if not out_path.exists():  # 幂等：已存在则跳过
                plot_window(win, date_str, out_path)
                n_plot += 1
            else:
                n_skip += 1
        cur += pd.Timedelta(days=1)

    total = len(list(out_dir.glob("*_cycle.png")))
    print(f"  {gate}: 新绘 {n_plot}, 跳过已存在 {n_skip}, 目录共 {total} 张 -> {out_dir}")


if __name__ == "__main__":
    for gate in ("A", "B"):
        print(f"=== {gate} 路 ===")
        process_route(gate)
    print("Done.")
