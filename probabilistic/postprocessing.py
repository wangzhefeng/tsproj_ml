# -*- coding: utf-8 -*-
"""分位数 crossing 诊断与 q50 锚定修复。"""

import re
from typing import List, Tuple

import numpy as np
import pandas as pd


_QUANTILE_COLUMN_RE = re.compile(r"^predict_q(?P<token>\d+(?:p\d+)?)$")


def _parse_quantile_columns(frame: pd.DataFrame) -> List[Tuple[float, str]]:
    parsed: List[Tuple[float, str]] = []
    for column in frame.columns:
        match = _QUANTILE_COLUMN_RE.match(str(column))
        if match is None:
            continue
        level = float(match.group("token").replace("p", ".")) / 100.0
        parsed.append((level, str(column)))
    parsed.sort(key=lambda item: item[0])
    if len({level for level, _ in parsed}) != len(parsed):
        raise ValueError("quantile columns contain duplicate numeric levels")
    return parsed


def repair_quantile_crossing(
    frame: pd.DataFrame,
    enabled: bool = False,
    point_column: str = "predict_value",
) -> pd.DataFrame:
    """以 q50 为固定锚点修复 crossing，并同步点预测。"""
    quantile_columns = _parse_quantile_columns(frame)
    if not quantile_columns:
        return frame

    median_matches = [
        index
        for index, (level, _) in enumerate(quantile_columns)
        if np.isclose(level, 0.5, rtol=0.0, atol=1e-12)
    ]
    if len(median_matches) != 1:
        raise ValueError("quantile columns must contain exactly one q50 column")

    result = frame.copy()
    columns = [column for _, column in quantile_columns]
    values = result[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("quantile columns must contain only finite values")

    median_index = median_matches[0]
    median = values[:, median_index].copy()
    if enabled:
        if median_index > 0:
            lower = np.minimum(values[:, :median_index], median[:, None])
            values[:, :median_index] = np.maximum.accumulate(lower, axis=1)
        if median_index + 1 < values.shape[1]:
            upper = np.maximum(values[:, median_index + 1:], median[:, None])
            values[:, median_index + 1:] = np.maximum.accumulate(upper, axis=1)
        values[:, median_index] = median
        result[columns] = values

    result[point_column] = median
    return result
