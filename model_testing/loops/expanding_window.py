"""扩展窗回测：rolling 引擎的 expanding_window 形态。

每折训练集取全部合格历史候选（不截断 train_window_steps），随折扩大；
折间评估区间不重叠（stride 语义同 fixed-step），产物照常拼接总图，
metadata mode=expanding_window。final fit 无固定窗口语义，暂限
backtest-only（pipeline/runner.py 合同）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from probabilistic.calibration import ConformalCalibrationTracker
from model_testing.contracts.protocols import BacktestRunner
from model_testing.loops.fixed_step import run_rolling_backtest


def run_expanding_window_backtest(
    runner: BacktestRunner, test_dir: Path, *, mode: str,
) -> tuple[dict[str, Any] | None, ConformalCalibrationTracker | None, tuple[Any, ...]]:
    """扩展窗回测入口：共享 rolling 引擎，训练集逐折扩大。"""
    return run_rolling_backtest(
        runner, test_dir, mode=mode,
        stitch_overview=True, mode_label="expanding_window",
    )


__all__ = ["run_expanding_window_backtest"]
