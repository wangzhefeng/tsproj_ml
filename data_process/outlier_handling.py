# -*- coding: utf-8 -*-
"""滑窗训练段异常处理（legacy 身份，2026-08-29 架构收敛自 data_provider/ 迁入）。

身份声明（架构收敛方案 D2/D12）：本模块是 legacy 训练窗口清洗实现，canonical
runtime 不接线（canonical 信息集默认「缺失/异常 = RAISE」的预测性维护契约见
AGENTS.md「包间分层规则」）。保留原因是报告 schema（TRAIN_OUTLIER_REPORT_COLUMNS）
仍被算力场景结果融合脚本消费；`enable_train_outlier_handling` 已被 run.py 列为
canonical CLI 不支持项。禁止把本模块接入 canonical runtime。
"""

from typing import List, Set, Tuple

import numpy as np
import pandas as pd

from utils.log_util import logger


TRAIN_OUTLIER_REPORT_COLUMNS = [
    "window",
    "time",
    "raw_y",
    "clean_y",
    "outlier_type",
    "run_length",
    "rule",
]


def empty_train_outlier_report() -> pd.DataFrame:
    return pd.DataFrame(columns=TRAIN_OUTLIER_REPORT_COLUMNS)


def _iter_true_runs(mask: pd.Series) -> List[List[int]]:
    runs = []
    start = None
    values = mask.fillna(False).to_numpy(dtype=bool)
    for pos, is_true in enumerate(values):
        if is_true and start is None:
            start = pos
        if start is not None and (not is_true or pos == len(values) - 1):
            end = pos if is_true else pos - 1
            runs.append(list(range(start, end + 1)))
            start = None
    return runs


def _detect_drop_rebound_positions(
    y: pd.Series,
    max_run_points: int,
    min_abs_diff: float,
) -> Set[int]:
    positions = set()
    values = y.to_numpy(dtype=float)
    n_values = len(values)
    for start in range(1, n_values - 1):
        for run_len in range(1, max_run_points + 1):
            end = start + run_len - 1
            next_pos = end + 1
            if next_pos >= n_values:
                continue
            prev_value = values[start - 1]
            next_value = values[next_pos]
            run_values = values[start : end + 1]
            if not np.isfinite(prev_value) or not np.isfinite(next_value) or not np.all(np.isfinite(run_values)):
                continue
            drop = prev_value - run_values[0]
            rebound = next_value - run_values[-1]
            if (
                drop >= min_abs_diff
                and rebound >= min_abs_diff
                and np.nanmax(run_values) < min(prev_value, next_value)
            ):
                positions.update(range(start, end + 1))
    return positions


def _detect_rise_rebound_positions(
    y: pd.Series,
    max_run_points: int,
    min_abs_diff: float,
) -> Set[int]:
    positions = set()
    values = y.to_numpy(dtype=float)
    n_values = len(values)
    for start in range(1, n_values - 1):
        for run_len in range(1, max_run_points + 1):
            end = start + run_len - 1
            next_pos = end + 1
            if next_pos >= n_values:
                continue
            prev_value = values[start - 1]
            next_value = values[next_pos]
            run_values = values[start : end + 1]
            if not np.isfinite(prev_value) or not np.isfinite(next_value) or not np.all(np.isfinite(run_values)):
                continue
            rise = run_values[0] - prev_value
            fallback = run_values[-1] - next_value
            if (
                rise >= min_abs_diff
                and fallback >= min_abs_diff
                and np.nanmin(run_values) > max(prev_value, next_value)
            ):
                positions.update(range(start, end + 1))
    return positions


def _interpolate_target(
    df: pd.DataFrame,
    target_feature: str,
    outlier_positions: List[int],
) -> pd.DataFrame:
    cleaned = df.copy()
    if not outlier_positions:
        return cleaned

    original_index = cleaned.index
    target = pd.to_numeric(cleaned[target_feature], errors="coerce")
    target.iloc[outlier_positions] = np.nan
    if "time" in cleaned.columns:
        time_index = pd.to_datetime(cleaned["time"])
        target.index = time_index
        target = target.interpolate(method="time", limit_direction="both")
        target.index = original_index
    else:
        target = target.interpolate(method="linear", limit_direction="both")
    cleaned[target_feature] = target
    return cleaned


def handle_train_outliers(
    args,
    df_history_train: pd.DataFrame,
    target_feature: str,
    window: int,
    log_prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean target outliers only inside a sliding training window.
    """
    report = empty_train_outlier_report()
    if not getattr(args, "enable_train_outlier_handling", False):
        return df_history_train, report
    if getattr(args, "train_outlier_method", "local_interpolate") != "local_interpolate":
        logger.warning(
            f"{log_prefix} Unsupported train_outlier_method="
            f"{getattr(args, 'train_outlier_method', None)}. Skip train outlier handling."
        )
        return df_history_train, report
    if target_feature not in df_history_train.columns:
        logger.warning(f"{log_prefix} Target feature '{target_feature}' not found. Skip train outlier handling.")
        return df_history_train, report

    y = pd.to_numeric(df_history_train[target_feature], errors="coerce")
    high_threshold = getattr(args, "high_outlier_threshold", None)
    high_max_run = int(getattr(args, "high_outlier_max_run_points", 4) or 0)
    drop_max_run = int(getattr(args, "drop_outlier_max_run_points", 2) or 0)
    drop_min_abs_diff = float(getattr(args, "drop_rebound_min_abs_diff", 900.0))
    low_threshold = getattr(args, "low_outlier_threshold", None)
    low_max_run = int(getattr(args, "low_outlier_max_run_points", 4) or 0)
    rise_max_run = int(getattr(args, "rise_outlier_max_run_points", 2) or 0)
    rise_min_abs_diff = float(getattr(args, "rise_rebound_min_abs_diff", 900.0))

    outlier_meta = {}
    if high_threshold is not None and high_max_run > 0:
        for run in _iter_true_runs(y > high_threshold):
            if len(run) <= high_max_run:
                for pos in run:
                    outlier_meta[pos] = (
                        "high_short_run",
                        len(run),
                        f"y > {high_threshold:g} and run_length <= {high_max_run}",
                    )

    if low_threshold is not None and low_max_run > 0:
        for run in _iter_true_runs(y < low_threshold):
            if len(run) <= low_max_run:
                for pos in run:
                    outlier_meta[pos] = (
                        "low_short_run",
                        len(run),
                        f"y < {low_threshold:g} and run_length <= {low_max_run}",
                    )

    if drop_max_run > 0:
        drop_positions = _detect_drop_rebound_positions(
            y=y,
            max_run_points=drop_max_run,
            min_abs_diff=drop_min_abs_diff,
        )
        drop_mask = pd.Series([idx in drop_positions for idx in range(len(y))])
        for run in _iter_true_runs(drop_mask):
            if len(run) <= drop_max_run:
                for pos in run:
                    outlier_meta.setdefault(
                        pos,
                        (
                            "drop_rebound",
                            len(run),
                            f"drop and rebound >= {drop_min_abs_diff:g} within {drop_max_run} points",
                        ),
                    )

    if rise_max_run > 0:
        rise_positions = _detect_rise_rebound_positions(
            y=y,
            max_run_points=rise_max_run,
            min_abs_diff=rise_min_abs_diff,
        )
        rise_mask = pd.Series([idx in rise_positions for idx in range(len(y))])
        for run in _iter_true_runs(rise_mask):
            if len(run) <= rise_max_run:
                for pos in run:
                    outlier_meta.setdefault(
                        pos,
                        (
                            "rise_rebound",
                            len(run),
                            f"rise and fallback >= {rise_min_abs_diff:g} within {rise_max_run} points",
                        ),
                    )

    outlier_positions = sorted(outlier_meta)
    cleaned = _interpolate_target(df_history_train, target_feature, outlier_positions)
    if not outlier_positions:
        return cleaned, report

    rows = []
    for pos in outlier_positions:
        outlier_type, run_length, rule = outlier_meta[pos]
        rows.append(
            {
                "window": window,
                "time": df_history_train.iloc[pos]["time"] if "time" in df_history_train.columns else df_history_train.index[pos],
                "raw_y": y.iloc[pos],
                "clean_y": cleaned.iloc[pos][target_feature],
                "outlier_type": outlier_type,
                "run_length": run_length,
                "rule": rule,
            }
        )
    report = pd.DataFrame(rows, columns=TRAIN_OUTLIER_REPORT_COLUMNS)
    logger.info(f"{log_prefix} Train outlier handling window {window}: cleaned {len(report)} points.")
    return cleaned, report
