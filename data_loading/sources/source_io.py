"""一次 registry 运行内的文件读取与验证帧缓存。"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

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

    def read_path(self, configured_path: str) -> pd.DataFrame:
        path = Path(configured_path)
        if not path.is_absolute():
            path = self._base_dir / path
        path = path.resolve()
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
        cache_key = (resolved_path, source.name, version == "history")
        cached = self._validated_cache.get(cache_key)
        if cached is None:
            raw = self.read_path(configured_path)
            cached = validate_frame(source, raw, generated=False, path_version=version)
            self._validated_cache[cache_key] = cached
        return cached
