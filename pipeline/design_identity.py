"""Raw-design identity for in-memory sharing and execution evidence; no array persistence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, cast

from data_loading.sources.provenance import file_sha256, source_hashes, generator_hashes
from forecasting_core.specs import ForecastConfigSpec

RAW_DESIGN_IDENTITY_VERSION = 3


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
    # R6（2026-09-06）：design.py 已迁 pipeline/supervised_design.py
    paths = [root / "pipeline" / "supervised_design.py"]
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
        "train_history_steps",
        "fold_count",
        "stride_steps",
        "train_window_days",
        "stride_months",
        "training_scope",
        "training_window",
        "forecast_window",
    )
    payload = {field: validation[field] for field in raw_fields if field in validation}
    if validation.get("training_window") is not None:
        payload["origin_sampling"] = validation.get("training", {}).get("origin_sampling")
    return payload


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
        "schema_version": RAW_DESIGN_IDENTITY_VERSION,
        "problem": config.problem.canonical_payload(),
        "data": config.data.canonical_payload(),
        "features": _raw_feature_payload(config),
        "strategy": config.strategy.canonical_payload(),
        "forecast_origin": str(origin),
        "validation": _raw_validation_payload(config),
        "data_phase": "historical",
        "source_hashes": source_hashes(config.data, root, data_phase="historical"),
        "generator_hashes": generator_hashes(config.data, generators),
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
