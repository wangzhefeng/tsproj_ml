# -*- coding: utf-8 -*-
"""分解诊断报告：trend/seasonal/residual 概览输出。

在 pipeline fit 后调用，输出 decomposition_diagnostics.csv 到
test_results_dir（只新增文件，不覆盖旧结果）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from decomposition.types import TARGET_KEY


def write_diagnostics_report(
    pipeline,
    history: pd.DataFrame,
    time_col: str,
    target_col: str,
    output_dir: Path,
    suffix: str = "",
) -> Path | None:
    """写入分解诊断报告；method=none 或未 fit 时跳过返回 None。

    suffix 用于区分多窗口（如 "_win3"），避免同目录逐窗口互相覆盖。
    """
    if not getattr(pipeline, "enabled", False) or not getattr(pipeline, "is_fitted", False):
        return None
    if pipeline.history_component is None or pipeline.history_component.empty:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"decomposition_diagnostics{suffix}.csv"

    y = pd.to_numeric(history[target_col], errors="coerce").to_numpy(dtype=float)
    deterministic = np.asarray(pipeline.history_component, dtype=float)
    residual = y - deterministic

    rows = [
        {"component": "y", "mean": float(np.nanmean(y)), "std": float(np.nanstd(y)),
         "min": float(np.nanmin(y)), "max": float(np.nanmax(y)), "n": int(len(y))},
        {"component": "deterministic", "mean": float(np.nanmean(deterministic)),
         "std": float(np.nanstd(deterministic)),
         "min": float(np.nanmin(deterministic)), "max": float(np.nanmax(deterministic)),
         "n": int(len(deterministic))},
        {"component": "residual", "mean": float(np.nanmean(residual)),
         "std": float(np.nanstd(residual)),
         "min": float(np.nanmin(residual)), "max": float(np.nanmax(residual)),
         "n": int(len(residual))},
    ]
    # 趋势/季节分量（extractor components 拆分）
    extractor = getattr(pipeline, "_extractor", None)
    if extractor is not None and hasattr(extractor, "components"):
        for name, series in extractor.components().items():
            if name == TARGET_KEY:
                continue
            arr = np.asarray(series, dtype=float)
            rows.append({
                "component": name,
                "mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)),
                "min": float(np.nanmin(arr)), "max": float(np.nanmax(arr)),
                "n": int(len(arr)),
            })

    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    return out
