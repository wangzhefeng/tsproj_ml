# -*- coding: utf-8 -*-
"""窗口/quantile/output/model 多层并行的统一预算。"""

from typing import Any


def apply_window_parallel_budget(args: Any, window_workers: int) -> Any:
    """窗口并行启用时，把所有内部并行维度压到 1。"""
    if int(window_workers or 1) <= 1:
        return args
    args.multi_output_n_jobs = 1
    args.quantile_parallel_workers = 1
    args.ensemble_parallel_workers = 1
    args.model_thread_count = 1
    return args
