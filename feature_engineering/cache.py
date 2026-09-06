"""Content-addressed cache for compiled canonical training designs."""
from __future__ import annotations

import errno
import hashlib
import inspect
import json
import os
import pickle
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from time import sleep
from typing import Any, BinaryIO, Iterator, Mapping, cast

if os.name == "nt":
    import msvcrt as _process_lock_backend
else:
    import fcntl as _process_lock_backend

from forecasting_core.specs import ForecastConfigSpec


COMPILED_CACHE_SCHEMA_VERSION = 2
COMPILED_CACHE_DIR_NAME = "_compiled_features"
COMPILED_PAYLOAD_FILE = "compiled.pkl"
COMPILED_METADATA_FILE = "metadata.json"
_REQUIRED_FILES = (COMPILED_PAYLOAD_FILE, COMPILED_METADATA_FILE)
_PROCESS_LOCKS: dict[Path, threading.Lock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


def _acquire_process_lock(lock_file: BinaryIO) -> None:
    if os.name != "nt":
        _process_lock_backend.flock(
            lock_file.fileno(),
            _process_lock_backend.LOCK_EX,
        )
        return
    lock_file.seek(0, os.SEEK_END)
    if lock_file.tell() == 0:
        lock_file.write(b"\0")
        lock_file.flush()
    lock_file.seek(0)
    while True:
        try:
            _process_lock_backend.locking(
                lock_file.fileno(),
                _process_lock_backend.LK_NBLCK,
                1,
            )
            return
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EDEADLK}:
                raise
            sleep(0.05)


def _release_process_lock(lock_file: BinaryIO) -> None:
    if os.name != "nt":
        _process_lock_backend.flock(
            lock_file.fileno(),
            _process_lock_backend.LOCK_UN,
        )
        return
    lock_file.seek(0)
    _process_lock_backend.locking(
        lock_file.fileno(),
        _process_lock_backend.LK_UNLCK,
        1,
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_path(base_dir: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _source_hashes(
    config: ForecastConfigSpec,
    base_dir: Path,
) -> dict[str, str]:
    hashes = {}
    for source in config.data.sources:
        if source.source_type != "file":
            continue
        for path_role in ("history_path", "backtest_path", "future_path"):
            configured_path = getattr(source, path_role)
            if configured_path is None:
                continue
            hashes[f"{source.name}:{path_role}"] = file_sha256(
                _resolved_path(base_dir, configured_path)
            )
    return hashes


def _generator_hashes(
    config: ForecastConfigSpec,
    generators: Mapping[str, Any],
) -> dict[str, str]:
    hashes = {}
    for source in config.data.sources:
        if source.source_type != "generated":
            continue
        generator_name = source.generator or ""
        generator = generators.get(generator_name)
        if generator is None:
            raise ValueError(
                f"no generator registered for compiled cache source {source.name!r}"
            )
        try:
            callable_implementation = inspect.getsource(generator)
        except (OSError, TypeError):
            callable_implementation = (
                f"{generator.__module__}.{generator.__qualname__}"
            )
        try:
            source_file = inspect.getsourcefile(generator)
        except TypeError:
            source_file = None
        module_hash = (
            file_sha256(source_file)
            if source_file is not None and Path(source_file).is_file()
            else None
        )
        implementation = json.dumps(
            {
                "callable": callable_implementation,
                "module_sha256": module_hash,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        hashes[source.name] = hashlib.sha256(implementation).hexdigest()
    return hashes


def _environment_hashes(base_dir: Path) -> dict[str, str]:
    hashes = {}
    for name in ("pyproject.toml", "uv.lock"):
        path = base_dir / name
        if path.is_file():
            hashes[name] = file_sha256(path)
    return hashes


def _compilation_implementation_hashes() -> dict[str, str]:
    """绑定原始设计的代码来源，不导入运行层，也不绑定无关模型实现。"""
    root = Path(__file__).resolve().parents[1]
    # R6（2026-09-06）：design.py 已迁 model_pipeline/supervised_design.py
    paths = [root / "model_pipeline" / "supervised_design.py"]
    for package in (
        "forecasting_core", "data_loading", "feature_engineering",
        "model_training/strategies", "model_testing", "decomposition",
    ):
        paths.extend((root / package).rglob("*.py"))
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(set(paths))
    }


def _raw_feature_payload(config: ForecastConfigSpec) -> dict[str, object]:
    payload = config.features.canonical_payload()
    payload.pop("selection", None)
    transformations = dict(
        cast(Mapping[str, Any], payload["transformations"])
    )
    transformations.pop("feature_scaling", None)
    transformations.pop("target", None)
    payload["transformations"] = transformations
    return payload


def _raw_validation_payload(config: ForecastConfigSpec) -> dict[str, Any]:
    validation = config.validation.canonical_payload()
    raw_fields = (
        "schedule_mode",
        "horizon_mode",
        "history_steps",
        "train_window_steps",
        "fold_count",
        "stride_steps",
        "train_window_days",
        "stride_months",
        "training_scope",
    )
    return {field: validation[field] for field in raw_fields if field in validation}


def raw_design_provenance(
    config: ForecastConfigSpec,
    *,
    base_dir: str | Path,
    origin: Any,
    generators: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(base_dir).resolve()
    if config.strategy is None:
        raise ValueError("raw design fingerprint requires a strategy")
    payload = {
        "schema_version": COMPILED_CACHE_SCHEMA_VERSION,
        "problem": config.problem.canonical_payload(),
        "data": config.data.canonical_payload(),
        "features": _raw_feature_payload(config),
        "strategy": config.strategy.canonical_payload(),
        "forecast_origin": str(origin),
        "validation": _raw_validation_payload(config),
        "source_hashes": _source_hashes(config, root),
        "generator_hashes": _generator_hashes(config, generators),
        "environment_hashes": _environment_hashes(root),
        "compilation_implementation_hashes": _compilation_implementation_hashes(),
    }
    return payload


def compute_raw_design_fingerprint(
    config: ForecastConfigSpec,
    *,
    base_dir: str | Path,
    origin: Any,
    generators: Mapping[str, Any],
) -> str:
    encoded = json.dumps(
        raw_design_provenance(config, base_dir=base_dir, origin=origin, generators=generators),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_dir(results_root: str | Path, fingerprint: str) -> Path:
    return Path(results_root) / COMPILED_CACHE_DIR_NAME / fingerprint


@contextmanager
def raw_design_cache_lock(
    results_root: str | Path,
    fingerprint: str,
) -> Iterator[None]:
    """Serialize same-key cache misses across threads and processes."""
    directory = cache_dir(results_root, fingerprint).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".lock"
    with _PROCESS_LOCKS_GUARD:
        process_lock = _PROCESS_LOCKS.setdefault(lock_path, threading.Lock())
    with process_lock, lock_path.open("a+b") as lock_file:
        _acquire_process_lock(lock_file)
        try:
            yield
        finally:
            _release_process_lock(lock_file)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def save_compiled_cache(
    results_root: str | Path,
    fingerprint: str,
    payload: Mapping[str, Any],
) -> Path:
    directory = cache_dir(results_root, fingerprint)
    blob = pickle.dumps(dict(payload), protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        "schema_version": COMPILED_CACHE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "payload_sha256": hashlib.sha256(blob).hexdigest(),
    }
    _atomic_write(directory / COMPILED_PAYLOAD_FILE, blob)
    _atomic_write(
        directory / COMPILED_METADATA_FILE,
        json.dumps(metadata, sort_keys=True).encode("utf-8"),
    )
    return directory


def load_compiled_cache(
    results_root: str | Path,
    fingerprint: str,
) -> dict[str, Any]:
    directory = cache_dir(results_root, fingerprint)
    for name in _REQUIRED_FILES:
        if not (directory / name).is_file():
            raise FileNotFoundError(
                f"compiled feature cache incomplete at {directory}: missing {name}"
            )
    try:
        metadata = json.loads(
            (directory / COMPILED_METADATA_FILE).read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"invalid compiled feature cache metadata at {directory}") from exc
    if int(metadata.get("schema_version", -1)) != COMPILED_CACHE_SCHEMA_VERSION:
        raise ValueError(f"compiled feature cache schema mismatch at {directory}")
    if str(metadata.get("fingerprint")) != fingerprint:
        raise ValueError(f"compiled feature cache fingerprint mismatch at {directory}")
    blob = (directory / COMPILED_PAYLOAD_FILE).read_bytes()
    if metadata.get("payload_sha256") != hashlib.sha256(blob).hexdigest():
        raise ValueError(f"compiled feature cache payload hash mismatch at {directory}")
    try:
        payload = pickle.loads(blob)
    except (pickle.PickleError, EOFError, AttributeError, ValueError) as exc:
        raise ValueError(f"invalid compiled feature cache payload at {directory}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"compiled feature cache payload must be a mapping at {directory}")
    return payload


__all__ = [
    "COMPILED_CACHE_DIR_NAME",
    "COMPILED_CACHE_SCHEMA_VERSION",
    "cache_dir",
    "compute_raw_design_fingerprint",
    "file_sha256",
    "load_compiled_cache",
    "raw_design_cache_lock",
    "save_compiled_cache",
]
