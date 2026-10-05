"""一次 registry 运行内的文件读取与验证帧缓存。"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from threading import RLock

import numpy as np
import pandas as pd

from data_loading.processing.validation import validate_frame
from forecasting_core.specs.data import DataSourceSpec

if TYPE_CHECKING:
    from data_loading.information.information_set import InformationSetRequest

FrameReader = Callable[[Path], pd.DataFrame]
SourceGenerator = Callable[[DataSourceSpec, "InformationSetRequest"], pd.DataFrame]


class SourceFrames:
    """状态只属于一个 registry；返回值/键与原读取链保持一致。"""

    def __init__(self, base_dir: Path, reader: FrameReader) -> None:
        self._base_dir = base_dir
        self._reader = reader
        self._raw_cache: dict[Path, pd.DataFrame] = {}
        # bool 区分 history 推导 available_at；不改变原键或扩大共享范围。
        self._validated_cache: dict[tuple[Path, str, bool], pd.DataFrame] = {}
        self._numeric_cache = {}
        self._versions = {}
        self._lock = RLock()

    def numeric_history(self, source: DataSourceSpec, column: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
        """One immutable source-time numeric snapshot per registry lifetime."""
        if source.availability.value != "source_time" or source.series_id_cols or source.history_path is None:
            raise ValueError("numeric snapshots require local source-time history")
        with self._lock:
            frame = self.read_validated(source, source.history_path, "history")
            path = Path(source.history_path)
            path = (path if path.is_absolute() else self._base_dir / path).resolve()
            self._check_version(path)
            key = (path, source.name, column)
            if key not in self._numeric_cache:
                values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float, copy=True)
                values.flags.writeable = False
                self._numeric_cache[key] = (pd.DatetimeIndex(frame[source.time_col]), values)
            return self._numeric_cache[key]

    def _check_version(self, path: Path) -> None:
        stat = path.stat()
        version = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if path in self._versions and self._versions[path] != version:
            raise ValueError(f"source changed during registry snapshot: {path}")
        self._versions[path] = version

    def read_path(self, configured_path: str) -> pd.DataFrame:
        path = Path(configured_path)
        if not path.is_absolute():
            path = self._base_dir / path
        path = path.resolve()
        self._check_version(path)
        if path not in self._raw_cache:
            frame = self._reader(path)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"reader must return a DataFrame for {path}")
            self._raw_cache[path] = frame.copy(deep=True)
        return self._raw_cache[path].copy(deep=True)

    def read_validated(self, source: DataSourceSpec, configured_path: str, version: str) -> pd.DataFrame:
        resolved_path = Path(configured_path)
        if not resolved_path.is_absolute():
            resolved_path = self._base_dir / resolved_path
        resolved_path = resolved_path.resolve()
        self._check_version(resolved_path)
        cache_key = (resolved_path, source.name, version == "history")
        cached = self._validated_cache.get(cache_key)
        if cached is None:
            raw = self.read_path(configured_path)
            cached = validate_frame(source, raw, generated=False, path_version=version)
            self._validated_cache[cache_key] = cached
        return cached
