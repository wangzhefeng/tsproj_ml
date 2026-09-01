# -*- coding: utf-8 -*-
"""B1 残差频谱诊断：对分解 residual 做 FFT/ACF，输出跨窗口稳定性证据。

设计目的：为 Phase 4 扩展方法（VMD/Wavelet 等分量级模型）提供
"residual 是否仍含稳定频带结构"的量化证据——若所有窗口的残差
FFT 主导周期一致且 ACF 峰显著，说明 decomposition 之后仍有
未被 linear/STL/MSTL 吸收的周期成分；反之则说明现有方法已足够。

本模块只做诊断，不改预测链路，不写入结果目录之外的位置。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decomposition.periods import acf_periods, fft_dominant_period


@dataclass
class ResidualDiagnosticsRow:
    """单个窗口的残差频谱诊断结果。"""

    window_idx: int
    fft_dominant_period_samples: float | None
    fft_dominant_amplitude: float | None
    acf_top_periods: list[tuple[int, float]] = field(default_factory=list)
    n_obs: int = 0


@dataclass
class ResidualDiagnosticsSummary:
    """跨窗口的残差频谱稳定性汇总。"""

    rows: list[ResidualDiagnosticsRow] = field(default_factory=list)
    fft_period_median: float | None = None
    fft_period_std: float | None = None
    fft_period_cv: float | None = None  # 变异系数 std/median，越小越稳定
    acf_max_value: float | None = None  # 最强 ACF 峰
    stable_band_detected: bool = False  # 跨窗口频带是否稳定


def diagnose_window_residual(
    residual: np.ndarray,
    window_idx: int,
    max_lags: int = 48,
    top_n: int = 3,
) -> ResidualDiagnosticsRow:
    """对单个窗口的分解残差做 FFT/ACF 诊断。"""
    residual = np.asarray(residual, dtype=float)
    n = len(residual)
    row = ResidualDiagnosticsRow(
        window_idx=window_idx,
        fft_dominant_period_samples=None,
        fft_dominant_amplitude=None,
        n_obs=n,
    )
    if n < 8:
        return row

    fft_result = fft_dominant_period(residual)
    row.fft_dominant_period_samples = fft_result["dominant_period_samples"]
    row.fft_dominant_amplitude = fft_result["dominant_amplitude"]
    row.acf_top_periods = acf_periods(residual, max_lags=max_lags, top_n=top_n)
    return row


def summarize_window_residuals(
    rows: list[ResidualDiagnosticsRow],
) -> ResidualDiagnosticsSummary:
    """汇总多个窗口的残差诊断，评估频带跨窗口稳定性。

    判定逻辑：
    - FFT 主导周期的变异系数（CV）< 0.5 → 频带跨窗口稳定；
    - 任一窗口的最强 ACF 峰 > 0.3 → 残差存在显著自相关周期；
    - 两者同时满足才标记 stable_band_detected=True（需进一步分解）。
    """
    summary = ResidualDiagnosticsSummary(rows=rows)
    valid_periods = [
        r.fft_dominant_period_samples
        for r in rows
        if r.fft_dominant_period_samples is not None
    ]
    if valid_periods:
        median = float(np.median(valid_periods))
        std = float(np.std(valid_periods))
        summary.fft_period_median = median
        summary.fft_period_std = std
        summary.fft_period_cv = std / (median + 1e-12)

    all_acf_values = [
        abs(val) for r in rows for _, val in r.acf_top_periods
    ]
    if all_acf_values:
        summary.acf_max_value = float(np.max(all_acf_values))

    if (
        summary.fft_period_cv is not None
        and summary.fft_period_cv < 0.5
        and summary.acf_max_value is not None
        and summary.acf_max_value > 0.3
    ):
        summary.stable_band_detected = True
    return summary


def write_residual_diagnostics(
    summary: ResidualDiagnosticsSummary,
    output_path: Path,
) -> Path:
    """把残差诊断结果写为 CSV（只新增，不覆盖旧文件）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in summary.rows:
        top = ";".join(f"{lag}:{val:.3f}" for lag, val in r.acf_top_periods[:2])
        rows.append(
            {
                "window_idx": r.window_idx,
                "fft_dominant_period_samples": r.fft_dominant_period_samples,
                "fft_dominant_amplitude": r.fft_dominant_amplitude,
                "acf_top_periods": top,
                "n_obs": r.n_obs,
            }
        )
    summary_row = {
        "window_idx": -1,  # 汇总行
        "fft_dominant_period_samples": summary.fft_period_median,
        "fft_dominant_amplitude": summary.fft_period_cv,  # 借用该列存 CV
        "acf_top_periods": f"max={summary.acf_max_value}",
        "n_obs": int(summary.stable_band_detected),
    }
    rows.append(summary_row)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")
    return output_path
