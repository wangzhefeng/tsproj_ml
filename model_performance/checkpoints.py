"""Trusted-local, content-addressed completed model fits (not estimator resume).

One atomic envelope binds schema, identity and payload SHA256. Orphan temporary
files are ignored, never interpreted as completion. Pickles are trusted-local
artifacts, not a format for accepting untrusted remote checkpoints.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar
from uuid import uuid4

import numpy as np

from forecasting_core.checkpoints import FitCheckpointError
from utils.runtime_env import RUNTIME_DEPENDENCY_PACKAGES

try:
    import fcntl
except ImportError:  # Keep non-checkpoint runtime importable on Windows.
    fcntl = None

T = TypeVar("T")
CHECKPOINT_SCHEMA_VERSION = 1

# 本地 checkpoint 库存的保留上限：超龄先删，再按最旧优先压到字节预算内。
# 删除是安全的——miss 只是重新拟合，不删除任何非 .fit/.tmp/.lock 文件。
FIT_CHECKPOINT_MAX_BYTES = 10 << 30
FIT_CHECKPOINT_MAX_AGE_SECONDS = 30 * 24 * 3600
_ORPHAN_TMP_MAX_AGE_SECONDS = 3600


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def prune_fit_checkpoints(
    root: str | Path,
    *,
    max_bytes: int = FIT_CHECKPOINT_MAX_BYTES,
    max_age_seconds: int = FIT_CHECKPOINT_MAX_AGE_SECONDS,
    orphan_tmp_age_seconds: int = _ORPHAN_TMP_MAX_AGE_SECONDS,
    now: float | None = None,
) -> dict[str, int]:
    """Bounded retention for the trusted-local checkpoint store.

    Removes ``*.fit`` entries older than ``max_age_seconds`` first, then evicts
    oldest-write-first until the remainder fits ``max_bytes``. Orphan ``*.tmp``
    files (abandoned writes) older than ``orphan_tmp_age_seconds`` are removed;
    fresh ones may belong to an in-flight write and are left alone. A removed
    ``.fit`` takes its sibling ``.lock`` along. Everything else (batch state
    JSON, unrelated files) is never touched. Deletion is safe by construction:
    a missing checkpoint is a miss that refits, never corrupted state.
    """
    root = Path(root)
    stats = {"scanned_fits": 0, "removed_fits": 0,
             "removed_orphan_tmps": 0, "removed_bytes": 0}
    if not root.is_dir():
        return stats
    now = time.time() if now is None else float(now)
    entries: list[tuple[float, int, Path]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".tmp":
            info = path.stat()
            if now - info.st_mtime > orphan_tmp_age_seconds:
                path.unlink()
                stats["removed_orphan_tmps"] += 1
                stats["removed_bytes"] += info.st_size
            continue
        if path.suffix != ".fit":
            continue
        info = path.stat()
        entries.append((info.st_mtime, info.st_size, path))
    stats["scanned_fits"] = len(entries)

    def _remove(size: int, path: Path) -> None:
        path.unlink()
        path.with_suffix(".lock").unlink(missing_ok=True)
        stats["removed_fits"] += 1
        stats["removed_bytes"] += size

    keep: list[tuple[float, int, Path]] = []
    for mtime, size, path in entries:
        if now - mtime > max_age_seconds:
            _remove(size, path)
        else:
            keep.append((mtime, size, path))
    total = sum(size for _, size, _ in keep)
    for _mtime, size, path in sorted(keep):
        if total <= max_bytes:
            break
        _remove(size, path)
        total -= size
    return stats


def implementation_fingerprint() -> str:
    """Hash actual implementation files and installed numerical library versions.

    进程内缓存（2026-09-27）：输入是全部实现包的 .py 文件 + 已装库版本，
    进程生命周期内必然不变；此前批跑 171 任务会重复全仓哈希 171 次。
    """
    cached = _IMPLEMENTATION_FINGERPRINT_CACHE.get("value")
    if cached is not None:
        return cached
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for package in ("forecasting_core", "data_loading", "feature_engineering",
                    "model_training", "model_predicting", "model_performance",
                    "pipeline", "probabilistic", "models", "decomposition",
                    "model_building", "model_testing", "utils"):
        for path in sorted((root / package).rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    versions = {"python": platform.python_version()}
    for name in RUNTIME_DEPENDENCY_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "absent"
    digest.update(_json(versions))
    value = digest.hexdigest()
    _IMPLEMENTATION_FINGERPRINT_CACHE["value"] = value
    return value


_IMPLEMENTATION_FINGERPRINT_CACHE: dict[str, str] = {}


def runtime_checkpoint_errors(method):
    """Add structured phase failures without changing non-checkpoint callers."""
    @wraps(method)
    def wrapped(*call_args, **kwargs):
        runner = call_args[0] if call_args else kwargs.get("config")
        args = call_args[1:]
        try:
            return method(*call_args, **kwargs)
        except FitCheckpointError:
            raise
        except Exception as exc:
            root = kwargs.get("checkpoint_root", getattr(runner, "checkpoint_root", None))
            if root is None:
                raise
            config = (runner if method.__name__ == "run_canonical_config"
                      else getattr(runner, "config", None))
            if config is None and method.__name__ == "__init__":
                config = kwargs.get("config", args[0] if args else None)
            fingerprint = getattr(config, "fingerprint", None)
            fold = None
            if method.__name__ == "fit":
                fold = list(kwargs.get("train_indices", args[0] if args else ()))
            elif method.__name__ == "fit_final":
                fold = "final"
            raise FitCheckpointError(
                context={"config": fingerprint() if callable(fingerprint) else None,
                         "fold": fold}, model=None, phase=method.__name__,
                key="unresolved", reason=str(exc),
            ) from exc
    return wrapped


class FileFitCheckpoint:
    """Storage owned by L2; child contexts isolate config/fold/group/level."""

    def __init__(self, root: str | Path, context: Mapping[str, Any]) -> None:
        self.root = Path(root)
        self.context = json.loads(_json(dict(context)))

    def child(self, **context: Any) -> "FileFitCheckpoint":
        return type(self)(self.root, {**self.context, **context})

    def run(self, *, identity: Mapping[str, Any],
            arrays: Sequence[np.ndarray | None], fit: Callable[[], T]) -> T:
        phase = "identity"
        key = "unresolved"
        try:
            if fcntl is None:
                raise NotImplementedError("fit checkpoints require POSIX process locking")
            array_hashes = []
            for value in arrays:
                if value is None:
                    array_hashes.append(None)
                    continue
                array = np.ascontiguousarray(value)
                if array.dtype.hasobject:
                    raise TypeError("checkpoint arrays must not contain Python objects")
                array_hashes.append({"shape": array.shape, "dtype": array.dtype.str,
                                     "sha256": hashlib.sha256(array.tobytes()).hexdigest()})
            descriptor = {"schema": CHECKPOINT_SCHEMA_VERSION, "context": self.context,
                          "identity": dict(identity), "arrays": array_hashes}
            key = hashlib.sha256(_json(descriptor)).hexdigest()
            directory = self.root / key[:2]
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{key}.fit"
            # Separate opens + flock also serialize competing threads/processes.
            with (directory / f"{key}.lock").open("ab") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                phase = "load"
                if path.exists():
                    with path.open("rb") as handle:
                        header = json.loads(handle.readline())
                        payload = handle.read()
                    if header.get("schema") != CHECKPOINT_SCHEMA_VERSION:
                        raise ValueError("checkpoint schema mismatch")
                    if header.get("key") != key or header.get("descriptor") != json.loads(_json(descriptor)):
                        raise ValueError("checkpoint identity mismatch")
                    if header.get("sha256") != hashlib.sha256(payload).hexdigest():
                        raise ValueError("checkpoint payload hash mismatch")
                    return pickle.loads(payload)
                phase = "fit"
                result = fit()
                phase = "write"
                payload = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
                header = _json({"schema": CHECKPOINT_SCHEMA_VERSION, "key": key,
                                "descriptor": descriptor,
                                "sha256": hashlib.sha256(payload).hexdigest()})
                temporary = directory / f"{key}.{uuid4().hex}.tmp"
                with temporary.open("xb") as handle:
                    handle.write(header + b"\n" + payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                directory_fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                return result
        except FitCheckpointError:
            raise
        except Exception as exc:
            raise FitCheckpointError(context=self.context, model=identity.get("model", dict(identity)),
                                     phase=phase, key=key, reason=str(exc)) from exc
