"""重叠滑窗回测：rolling 引擎的 sliding_window 形态。

stride_steps < horizon，测试折相互重叠（同一时刻被多个折预测）；逐折评分与
fixed_step 同一路径（loops/scoring.py），产物侧不拼接总图（逐窗图与 csv
照常），metadata mode=sliding_window。窗口构造见 contracts/windows.py。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from probabilistic.calibration import ConformalCalibrationTracker
from model_testing.contracts.protocols import BacktestRunner
from model_testing.loops.fixed_step import run_rolling_backtest


def run_sliding_window_backtest(
    runner: BacktestRunner, test_dir: Path, *, mode: str,
) -> tuple[dict[str, Any] | None, ConformalCalibrationTracker | None, tuple[Any, ...]]:
    """重叠滑窗回测入口：共享 rolling 引擎，重叠折不拼接总图。"""
    return run_rolling_backtest(
        runner, test_dir, mode=mode,
        stitch_overview=False, mode_label="sliding_window",
    )


__all__ = ["run_sliding_window_backtest"]
