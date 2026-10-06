"""预测张量的不可变存储基础；不依赖预测类型。"""

from dataclasses import FrozenInstanceError
from typing import Any
import numpy as np


def _validate_float_array(
    values: np.ndarray,
    ndim: int,
    name: str,
    *,
    check_finite: bool = True,
) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if values.ndim != ndim:
        raise ValueError(f"{name} must have exactly {ndim} dimensions")
    if any(size == 0 for size in values.shape):
        raise ValueError(f"{name} axes must be nonempty")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError(f"{name} must have a floating dtype")
    if check_finite and not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def _immutable_array_storage(values: np.ndarray) -> tuple[bytes, str, tuple[int, ...]]:
    return values.tobytes(order="C"), values.dtype.str, values.shape


def _array_from_storage(
    value_bytes: bytes,
    value_dtype: str,
    value_shape: tuple[int, ...],
) -> np.ndarray:
    return np.frombuffer(value_bytes, dtype=np.dtype(value_dtype)).reshape(value_shape)


class _FrozenTensor:
    __slots__ = ()

    def __setattr__(self, name: str, value: Any) -> None:
        raise FrozenInstanceError(f"cannot assign to field '{name}'")

    def __delattr__(self, name: str) -> None:
        raise FrozenInstanceError(f"cannot delete field '{name}'")
