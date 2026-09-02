"""Content-addressed cache for compiled canonical training designs."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Mapping

from forecasting_core.specs import ForecastConfigSpec


COMPILED_CACHE_SCHEMA_VERSION = 1
COMPILED_CACHE_DIR_NAME = "_compiled_features"
COMPILED_PAYLOAD_FILE = "compiled.pkl"
COMPILED_METADATA_FILE = "metadata.json"
_REQUIRED_FILES = (COMPILED_PAYLOAD_FILE, COMPILED_METADATA_FILE)


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
            implementation = inspect.getsource(generator).encode("utf-8")
        except (OSError, TypeError):
            implementation = (
                f"{generator.__module__}.{generator.__qualname__}"
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


def compute_compiled_fingerprint(
    config: ForecastConfigSpec,
    *,
    base_dir: str | Path,
    origin: Any,
    generators: Mapping[str, Any],
) -> str:
    root = Path(base_dir).resolve()
    payload = {
        "schema_version": COMPILED_CACHE_SCHEMA_VERSION,
        "config_fingerprint": config.fingerprint(),
        "forecast_origin": str(origin),
        "backtest_geometry": config.validation.canonical_payload(),
        "source_hashes": _source_hashes(config, root),
        "generator_hashes": _generator_hashes(config, generators),
        "environment_hashes": _environment_hashes(root),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_dir(results_root: str | Path, fingerprint: str) -> Path:
    return Path(results_root) / COMPILED_CACHE_DIR_NAME / fingerprint


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
    "compute_compiled_fingerprint",
    "file_sha256",
    "load_compiled_cache",
    "save_compiled_cache",
]
