"""输入源内容及生成器实现身份；不组合特征/训练设计的 fingerprint。"""
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping

from forecasting_core.specs.data import DataSpec


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_path(base_dir: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def source_hashes(
    data_spec: DataSpec,
    base_dir: Path,
) -> dict[str, str]:
    hashes = {}
    for source in data_spec.sources:
        if source.source_type != "file":
            continue
        for path_role in ("history_path", "backtest_path", "future_path"):
            configured_path = getattr(source, path_role)
            if configured_path is None:
                continue
            hashes[f"{source.name}:{path_role}"] = file_sha256(
                resolved_path(base_dir, configured_path)
            )
    return hashes


def generator_hashes(
    data_spec: DataSpec,
    generators: Mapping[str, Any],
) -> dict[str, str]:
    hashes = {}
    for source in data_spec.sources:
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
