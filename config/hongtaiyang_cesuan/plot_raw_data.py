"""红太阳原始15min负荷/光伏全年总图和逐月分面图，不改变源数据。"""
import argparse
from pathlib import Path

import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import pandas as pd

from prepare import DATA, SOURCES, validate_frame


# -------------------- 原始序列绘图 --------------------
def build_raw_figures(frame: pd.DataFrame, *, title: str, color: str) -> tuple[Figure, Figure]:
    """两类图均绘制全部原始点；同一序列的逐月图共享纵轴尺度。"""
    data = validate_frame(frame, "15min")
    overview = Figure(figsize=(16, 4.5), layout="constrained")
    FigureCanvasAgg(overview)
    axis = overview.subplots()
    axis.plot(data.time, data.value, color=color, linewidth=0.55)
    axis.set(title=f"{title} | 2025 raw 15-minute observations",
             xlabel="Time", ylabel="Power (source units)",
             xlim=(pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")))
    axis.xaxis.set_major_locator(mdates.MonthLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.grid(alpha=0.25)
    axis.set_ylim(bottom=0)

    monthly = Figure(figsize=(16, 12), layout="constrained")
    FigureCanvasAgg(monthly)
    axes = monthly.subplots(4, 3, sharey=True)
    monthly.suptitle(f"{title} | Monthly detail, raw 15-minute observations", fontsize=15)
    for month, axis in enumerate(axes.flat, start=1):
        start = pd.Timestamp(year=2025, month=month, day=1)
        stop = start + pd.offsets.MonthBegin(1)
        subset = data.loc[(data.time >= start) & (data.time < stop)]
        axis.plot(subset.time, subset.value, color=color, linewidth=0.6)
        axis.set(title=f"2025-{month:02d}", xlim=(start, stop))
        # 只标本月日期，避免月末29日与下月1日刻度挤在一起。
        axis.set_xticks(pd.date_range(start, stop, freq="7D", inclusive="left"))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylim(bottom=0)
    monthly.supylabel("Power (source units)")
    monthly.supxlabel("Time")
    return overview, monthly


def plot_raw_data(data_root: Path = DATA) -> list[Path]:
    """先校验全部输入，再将六张PNG保存到各站点visualization目录。"""
    data_root = Path(data_root)
    inputs = [(site, target, validate_frame(pd.read_csv(data_root / site / f"{target}.csv")))
              for site, target in SOURCES]
    outputs = []
    for site, target, frame in inputs:
        is_pv = target == "pv_load"
        title = f"{site} / {'PV generation power' if is_pv else 'Load power'}"
        figures = build_raw_figures(frame, title=title, color="#d97706" if is_pv else "#2563eb")
        output_dir = data_root / site / "visualization"
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            for suffix, figure in zip(("raw", "raw_monthly"), figures):
                path = output_dir / f"{target}_2025_{suffix}.png"
                figure.savefig(path, dpi=160, facecolor="white")
                outputs.append(path)
        finally:
            for figure in figures:
                figure.clear()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA)
    args = parser.parse_args()
    for path in plot_raw_data(args.data_root):
        print(path)


if __name__ == "__main__":
    main()
