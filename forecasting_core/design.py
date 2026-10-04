"""只读按列索引设计：共享特征事实，估计器边界才展开行列。"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def retained_array_bytes(arrays: Sequence[np.ndarray]) -> int:
    """Unique ndarray backing storage, not process RSS or Python object overhead."""
    blocks = {}
    for array in arrays:
        root = current = array
        seen = set()
        while getattr(current, "base", None) is not None and id(current) not in seen:
            seen.add(id(current))
            current = getattr(current, "base")
            if isinstance(current, np.ndarray):
                root = current
        blocks[id(root)] = root
    return sum(array.nbytes for array in blocks.values())


class IndexedDesign:
    """Each column is a shared 1-D value block plus an integer row offset.

    ``nbytes`` is the logical dense size; ``retained_bytes`` is the unique block
    footprint. Row slicing preserves blocks, and NumPy conversion is explicit
    materialization. No fitted state or data availability policy lives here.
    """

    def __init__(self, columns: Sequence[tuple[np.ndarray, int]], *,
                 start: int, stop: int, rows: np.ndarray | None = None) -> None:
        if start < 0 or stop < start:
            raise ValueError("invalid indexed design bounds")
        normalized = []
        frozen = {}
        for values, offset in columns:
            array = np.asarray(values)
            if array.ndim != 1 or not np.issubdtype(array.dtype, np.number):
                raise ValueError("indexed columns must be one-dimensional numeric arrays")
            if not isinstance(offset, (int, np.integer)):
                raise TypeError("column offset must be an integer")
            if stop > start and (start + offset < 0 or stop + offset > len(array)):
                raise ValueError("indexed column access is outside its value block")
            if array.flags.writeable:
                key = id(array)
                if key not in frozen:
                    frozen[key] = array.copy()
                    frozen[key].flags.writeable = False
                array = frozen[key]
            normalized.append((array, int(offset)))
        if not normalized:
            raise ValueError("indexed design needs at least one column")
        self.columns = tuple(normalized)
        self.start, self.stop = int(start), int(stop)
        if rows is not None:
            rows = np.asarray(rows, dtype=np.int64).copy()
            if rows.ndim != 1 or np.any((rows < start) | (rows >= stop)):
                raise ValueError("row selector is outside indexed design bounds")
            rows.flags.writeable = False
        self.rows = rows

    @property
    def shape(self) -> tuple[int, int]:
        return (self.stop - self.start if self.rows is None else len(self.rows), len(self.columns))

    ndim = 2
    dtype = np.dtype(float)

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def nbytes(self) -> int:
        return self.shape[0] * self.shape[1] * self.dtype.itemsize

    @property
    def retained_bytes(self) -> int:
        blocks = {id(values): values for values, _ in self.columns}
        return sum(v.nbytes for v in blocks.values()) + (0 if self.rows is None else self.rows.nbytes)

    def column(self, index: int) -> np.ndarray:
        values, offset = self.columns[index]
        if self.rows is None:
            return values[self.start + offset:self.stop + offset]
        return values[self.rows + offset]

    def is_finite(self) -> bool:
        return all(np.isfinite(self.column(i)).all() for i in range(self.shape[1]))

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        if copy is False:
            raise ValueError("indexed design requires materialization")
        result = np.empty(self.shape, dtype=self.dtype if dtype is None else dtype)
        self.copy_into(result)
        return result

    def copy_into(self, output: np.ndarray) -> None:
        if output.shape != self.shape:
            raise ValueError("output shape differs from indexed design")
        for index in range(self.shape[1]):
            output[:, index] = self.column(index)

    def __getitem__(self, selector):
        if isinstance(selector, tuple):
            # Column selection is consumed by feature selection/diagnostics.
            row, col = selector
            if isinstance(col, (int, np.integer)):
                return self.column(int(col))[row]
            return np.asarray(self)[selector]
        if isinstance(selector, slice) and self.rows is None:
            begin, end, step = selector.indices(len(self))
            if step == 1:
                return IndexedDesign(self.columns, start=self.start + begin,
                                     stop=self.start + max(begin, end))
        rows = (np.arange(self.start, self.stop) if self.rows is None else self.rows)[selector]
        if np.ndim(rows) == 0:
            return np.array([v[int(rows) + offset] for v, offset in self.columns])
        return IndexedDesign(self.columns, start=self.start, stop=self.stop, rows=rows)
