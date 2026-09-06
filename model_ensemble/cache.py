"""OOF cache: content-addressed storage keyed by an oof fingerprint (v4 §7.2).

Fingerprint inputs: ordered member names + single-model semantic fingerprints,
the shared problem/probabilistic payload, OOF geometry, the SHA-256 of every
actually-read source file, and the OOF schema/logic version. Fusion methods are
deliberately excluded because they only consume OOF predictions. Corrupt or
incomplete cache entries raise — partial reuse is never attempted.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from time import sleep
from typing import Any, BinaryIO, Callable, Iterator, Mapping

if os.name == "nt":
    import msvcrt as _process_lock_backend
else:
    import fcntl as _process_lock_backend

import numpy as np
import pandas as pd

from data_loading.sources.provenance import file_sha256
from model_ensemble.artifacts import OOFPredictionArtifact
from model_ensemble.specs import OOF_GAP_SEMANTICS

OOF_SCHEMA_VERSION = 2

OOF_PREDICTIONS_FILE = "oof_predictions.csv"
OOF_METADATA_FILE = "oof_metadata.json"
MEMBER_MANIFEST_FILE = "member_manifest.json"
_REQUIRED_FILES = (OOF_PREDICTIONS_FILE, OOF_METADATA_FILE, MEMBER_MANIFEST_FILE)
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


def compute_oof_fingerprint(
    *,
    members: Mapping[str, str],
    ensemble_payload: Mapping[str, Any],
    oof_payload: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> str:
    payload = {
        "schema_version": OOF_SCHEMA_VERSION,
        "members": {name: members[name] for name in sorted(members)},
        "ensemble": dict(ensemble_payload),
        "oof": dict(oof_payload),
        "source_hashes": {name: source_hashes[name] for name in sorted(source_hashes)},
    }
    if oof_payload.get("gap_steps", 0) > 0:
        payload["oof_gap_semantics"] = OOF_GAP_SEMANTICS
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def cache_dir(results_root: str | Path, oof_fingerprint: str) -> Path:
    return Path(results_root) / "_ensemble_oof" / oof_fingerprint


@contextmanager
def oof_cache_lock(
    results_root: str | Path,
    oof_fingerprint: str,
) -> Iterator[None]:
    """Serialize same-key OOF cache access across threads and processes."""
    directory = cache_dir(results_root, oof_fingerprint).resolve()
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


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _save_oof_cache_unlocked(
    results_root: str | Path,
    artifact: OOFPredictionArtifact,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Path:
    directory = cache_dir(results_root, artifact.oof_fingerprint)
    member_frames = []
    for name in artifact.member_order:
        values = artifact.values_by_member[name]
        flat = values.reshape(values.shape[0], -1)
        columns = pd.MultiIndex.from_product(
            [[name], [f"v{i}" for i in range(flat.shape[1])]]
        )
        member_frames.append(pd.DataFrame(flat, columns=columns))
    stacked = pd.concat(member_frames, axis=1)
    buffer = stacked.to_csv(index=False).encode("utf-8")
    _atomic_write_bytes(directory / OOF_PREDICTIONS_FILE, buffer)

    metadata = {
        "oof_schema_version": OOF_SCHEMA_VERSION,
        "oof_fingerprint": artifact.oof_fingerprint,
        "member_order": list(artifact.member_order),
        "targets": list(artifact.targets),
        "horizon": artifact.horizon,
        "quantile_levels": (
            list(artifact.quantile_levels) if artifact.quantile_levels else None
        ),
        "series_ids": [list(s) if isinstance(s, tuple) else s for s in artifact.series_ids],
        "folds": list(artifact.folds),
        **(dict(extra_metadata) if extra_metadata else {}),
        "execution_evidence": list(artifact.execution_evidence),
    }
    _atomic_write_bytes(
        directory / OOF_METADATA_FILE,
        json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode("utf-8"),
    )
    manifest = {
        name: {
            "sha256": hashlib.sha256(
                artifact.values_by_member[name].tobytes()
            ).hexdigest(),
            "shape": list(artifact.values_by_member[name].shape),
        }
        for name in artifact.member_order
    }
    _atomic_write_bytes(
        directory / MEMBER_MANIFEST_FILE,
        json.dumps(manifest, sort_keys=True).encode("utf-8"),
    )
    return directory


def save_oof_cache(
    results_root: str | Path,
    artifact: OOFPredictionArtifact,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically persist one complete OOF artifact under its same-key lock."""
    with oof_cache_lock(results_root, artifact.oof_fingerprint):
        return _save_oof_cache_unlocked(results_root, artifact, extra_metadata)


def _load_oof_cache_unlocked(
    results_root: str | Path,
    oof_fingerprint: str,
) -> OOFPredictionArtifact:
    directory = cache_dir(results_root, oof_fingerprint)
    present = {name for name in _REQUIRED_FILES if (directory / name).exists()}
    if not present:
        raise FileNotFoundError(f"OOF cache not found at {directory}")
    missing = set(_REQUIRED_FILES) - present
    if missing:
        raise ValueError(
            f"OOF cache incomplete at {directory}: missing {sorted(missing)}"
        )
    metadata = json.loads((directory / OOF_METADATA_FILE).read_text("utf-8"))
    if int(metadata.get("oof_schema_version", -1)) != OOF_SCHEMA_VERSION:
        raise ValueError(
            f"OOF cache schema version mismatch at {directory}"
        )
    if str(metadata.get("oof_fingerprint")) != oof_fingerprint:
        raise ValueError(f"OOF cache fingerprint mismatch at {directory}")

    frame = pd.read_csv(directory / OOF_PREDICTIONS_FILE, header=[0, 1])
    member_order = tuple(metadata["member_order"])
    n_samples = int(frame.shape[0])
    horizon = int(metadata["horizon"])
    targets = tuple(metadata["targets"])
    levels = metadata.get("quantile_levels")
    depth = 4 if levels else 3
    width = horizon * len(targets) * (len(levels) if levels else 1)

    values_by_member = {}
    manifest = json.loads((directory / MEMBER_MANIFEST_FILE).read_text("utf-8"))
    for name in member_order:
        if name not in frame.columns.get_level_values(0):
            raise ValueError(
                f"OOF cache missing member block {name!r} at {directory}"
            )
        block = frame[name]
        if list(block.columns) != [f"v{i}" for i in range(width)]:
            raise ValueError(
                f"OOF cache column set mismatch for member {name!r} at {directory}"
            )
        values = block.to_numpy(dtype=float).reshape(
            (n_samples, horizon, len(targets), len(levels)) if levels
            else (n_samples, horizon, len(targets))
        )
        expected = manifest.get(name, {})
        if expected.get("sha256") != hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest():
            raise ValueError(
                f"OOF cache hash mismatch for member {name!r} at {directory}"
            )
        values_by_member[name] = values

    missing_members = set(member_order) - set(values_by_member)
    if missing_members:
        raise ValueError(
            f"OOF cache incomplete member set at {directory}: {sorted(missing_members)}"
        )
    return OOFPredictionArtifact(
        values_by_member=values_by_member,
        member_order=member_order,
        targets=targets,
        horizon=horizon,
        quantile_levels=tuple(float(v) for v in levels) if levels else None,
        oof_fingerprint=oof_fingerprint,
        folds=tuple(metadata.get("folds", ())),
        series_ids=tuple(metadata.get("series_ids", ())),
        execution_evidence=tuple(metadata.get("execution_evidence", ())),
    )


def load_oof_cache(
    results_root: str | Path,
    oof_fingerprint: str,
) -> OOFPredictionArtifact:
    """Load and fully validate one OOF artifact under its same-key lock.

    Raises FileNotFoundError for a missing cache and ValueError for any
    corruption (missing files, incomplete columns, hash mismatch).
    """
    with oof_cache_lock(results_root, oof_fingerprint):
        return _load_oof_cache_unlocked(results_root, oof_fingerprint)


def get_or_create_oof_cache(
    results_root: str | Path,
    oof_fingerprint: str,
    factory: Callable[[], OOFPredictionArtifact],
) -> tuple[OOFPredictionArtifact, bool]:
    """Load an OOF artifact or generate it exactly once for a shared key.

    The boolean return value reports whether the artifact came from cache.
    Corrupt entries raise instead of being silently regenerated.
    """
    with oof_cache_lock(results_root, oof_fingerprint):
        try:
            return _load_oof_cache_unlocked(results_root, oof_fingerprint), True
        except FileNotFoundError:
            artifact = factory()
            if not isinstance(artifact, OOFPredictionArtifact):
                raise TypeError("OOF cache factory must return OOFPredictionArtifact")
            artifact = replace(artifact, oof_fingerprint=oof_fingerprint)
            _save_oof_cache_unlocked(results_root, artifact)
            return artifact, False


__all__ = [
    "OOF_SCHEMA_VERSION",
    "cache_dir",
    "compute_oof_fingerprint",
    "file_sha256",
    "get_or_create_oof_cache",
    "load_oof_cache",
    "oof_cache_lock",
    "save_oof_cache",
]
