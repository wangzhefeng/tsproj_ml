# -*- coding: utf-8 -*-
"""多步预测输入、schema、模型输出的严格契约。"""

from typing import Any, Iterable

import numpy as np
import pandas as pd


def require_future_horizon(df_future: pd.DataFrame, horizon: int) -> None:
    actual = len(df_future)
    if actual != int(horizon):
        raise ValueError(
            f"future frame length mismatch: expected horizon={int(horizon)}, got {actual}."
        )


def require_predictor_columns(columns: Iterable[str]) -> tuple[str, ...]:
    resolved = tuple(columns or ())
    if not resolved:
        raise ValueError("predictor feature set is empty.")
    return resolved


def require_schema_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    required = tuple(columns or ())
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"forecast feature schema missing columns: {missing}.")


def require_endogenous_history(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"history missing endogenous columns: {missing}.")
    invalid = [column for column in columns if not np.isfinite(pd.to_numeric(frame[column], errors="coerce")).any()]
    if invalid:
        raise ValueError(f"history has no finite endogenous values: {invalid}.")


def _finite(values: np.ndarray, label: str) -> np.ndarray:
    result = values.astype(float, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} prediction contains non-finite values.")
    return result


def require_pointwise_output(prediction: Any, horizon: int) -> np.ndarray:
    arr = np.asarray(prediction)
    if arr.shape == (horizon,):
        return _finite(arr, "pointwise")
    if arr.shape == (horizon, 1):
        return _finite(arr[:, 0], "pointwise")
    raise ValueError(
        f"pointwise prediction length mismatch: "
        f"expected {horizon}, got {len(arr.reshape(-1)) if arr.size else arr.shape}."
    )


def require_exact_vector_output(
    prediction: Any,
    expected_length: int,
    label: str,
) -> np.ndarray:
    arr = np.asarray(prediction)
    if arr.shape == (expected_length,):
        return _finite(arr, label)
    if arr.shape == (1, expected_length):
        return _finite(arr[0], label)
    if arr.shape == (expected_length, 1):
        return _finite(arr[:, 0], label)
    flat = arr.reshape(-1)
    raise ValueError(
        f"{label} prediction length mismatch: "
        f"expected {expected_length}, got {len(flat) if flat.size else arr.shape}."
    )


def require_direct_output(prediction: Any, horizon: int, label: str = "direct") -> np.ndarray:
    return require_exact_vector_output(
        prediction,
        horizon,
        label=f"{label} direct",
    )


def require_recursive_output(prediction: Any) -> float:
    arr = np.asarray(prediction)
    if arr.ndim == 0:
        value = float(arr)
    elif arr.shape in {(1,), (1, 1)}:
        value = float(arr.reshape(-1)[0])
    else:
        raise ValueError(f"recursive prediction shape mismatch: expected one value, got {arr.shape}.")
    if not np.isfinite(value):
        raise ValueError("recursive prediction contains a non-finite value.")
    return value


def require_dirrec_output(prediction: Any, block_size: int) -> np.ndarray:
    try:
        return require_direct_output(prediction, block_size)
    except ValueError as exc:
        raise ValueError(
            f"dirrec prediction shape mismatch: expected block width {block_size}, "
            f"got {np.asarray(prediction).shape}."
        ) from exc