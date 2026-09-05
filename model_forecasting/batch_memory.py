"""Bounded cache accounting and sampled RSS for batch admission.

Limits describe retained Python payload estimates and planned task working sets,
not an OS hard memory cap. Native estimator allocations remain externally owned.
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import fields, is_dataclass
from threading import Event, Lock, Thread
from typing import Any, Callable

import numpy as np
import psutil


def retained_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Count retained arrays (including view bases) once, plus Python containers."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        return size + (retained_bytes(value.base, seen) if value.base is not None else 0)
    if isinstance(value, dict):
        return size + sum(retained_bytes(k, seen) + retained_bytes(v, seen) for k, v in value.items())
    if isinstance(value, (tuple, list, set, frozenset)):
        return size + sum(retained_bytes(v, seen) for v in value)
    if is_dataclass(value) and not isinstance(value, type):
        return size + sum(retained_bytes(getattr(value, f.name), seen) for f in fields(value))
    if hasattr(value, "__dict__") and not callable(value):
        return size + retained_bytes(vars(value), seen)
    return size


class BoundedPayloadCache:
    """Single-flight LRU; eviction drops only cache ownership, never mutates data."""

    def __init__(self, *, max_bytes: int = 256 << 20, max_entries: int = 64) -> None:
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (max_bytes, max_entries)):
            raise ValueError("cache max_bytes/max_entries must be positive integers")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self._values: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self._lock = Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.oversized_skips = 0
        self.resident_bytes = 0

    def get_or_create(self, key: str, factory: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._values:
                value, _ = self._values[key]
                self._values.move_to_end(key)
                self.hits += 1
                return value
            value = factory()
            size = retained_bytes(value)
            self.misses += 1
            if size > self.max_bytes:
                self.oversized_skips += 1
                return value
            while self._values and (
                len(self._values) >= self.max_entries or self.resident_bytes + size > self.max_bytes
            ):
                _, (_, removed_size) = self._values.popitem(last=False)
                self.resident_bytes -= removed_size
                self.evictions += 1
            self._values[key] = (value, size)
            self.resident_bytes += size
            return value

    def payload(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._values), "hits": self.hits, "misses": self.misses,
                "resident_bytes": self.resident_bytes, "max_bytes": self.max_bytes,
                "max_entries": self.max_entries, "evictions": self.evictions,
                "oversized_skips": self.oversized_skips,
            }


class SampledRSS:
    """Measure a labelled sampled peak, including pool execution, every 20ms."""

    interval_seconds = 0.02

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self._stop = Event()
        self._thread = Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> "SampledRSS":
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.peak = max(self.peak, self.process.memory_info().rss)
        self._stop.set()
        self._thread.join()
