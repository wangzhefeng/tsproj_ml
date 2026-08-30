"""Reference-based ensemble runtime: YAML -> members -> OOF -> fuser -> persist.

Orchestration only: member lifecycles run through CanonicalBaseModelRunner,
output contracts (long schema, bundle layout, fingerprints) mirror the
single-model canonical runtime so downstream tooling stays uniform (v4 §8).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from model_ensemble import cache as oof_cache
from model_ensemble.loader import load_ensemble_config, resolve_members
from model_ensemble.predictor import combine_members
from model_ensemble.specs import EnsembleConfigSpec, EnsembleSpecError
from model_ensemble.trainer import fit_ensemble
from model_testing.backtest import resolve_origin
from data_loading import SourceRegistry
from model_forecasting.runtime import CanonicalBaseModelRunner
from model_forecasting.specs.config import parse_model_config


def run_ensemble_config_file(
    config_path: str | Path,
    output_root: str | Path | None = None,
) -> Any:
    config = load_ensemble_config(config_path)
    return run_ensemble_config(
        config,
        output_root=output_root,
        base_dir=Path(config_path).resolve().parent,
    )


def run_ensemble_config(
    config: EnsembleConfigSpec,
    output_root: str | Path | None = None,
    *,
    base_dir: str | Path = ".",
    use_oof_cache: bool = True,
) -> Any:
    """Execute one reference-based ensemble configuration end to end."""
    resolved = resolve_members(config, base_dir=base_dir)
    if len(resolved) < 2:
        raise EnsembleSpecError("ensemble requires at least two valid members")

    member_configs: dict[str, Any] = {}
    runners: dict[str, CanonicalBaseModelRunner] = {}
    member_fingerprints: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    registry_by_member: dict[str, SourceRegistry] = {}
    for member in config.members:
        member_raw = resolved[member.name]
        member_config = parse_model_config(member_raw, source=member.config_ref)
        member_configs[member.name] = member_config
        member_fingerprints[member.name] = member_config.fingerprint()
        registry = SourceRegistry(member_config.data, Path(base_dir).resolve())
        registry_by_member[member.name] = registry
        runners[member.name] = CanonicalBaseModelRunner(
            member_config,
            registry,
            _member_origin(member_config, registry),
        )
        for source in member_config.data.sources:
            if source.source_type != "file":
                raise EnsembleSpecError(
                    "ensemble OOF caching supports file sources only; generator "
                    "sources must RAISE (v4 §7.2)"
                )
            source_hashes[source.name] = oof_cache.file_sha256(
                Path(base_dir).resolve() / source.history_path
            )

    ens_payload = {
        "members": [member.payload() for member in config.members],
        "method": config.method.payload(),
        "problem": config.problem,
        "probabilistic": config.probabilistic,
    }
    oof_payload = config.oof.payload()
    fingerprint = oof_cache.compute_oof_fingerprint(
        members=member_fingerprints,
        ensemble_payload=ens_payload,
        oof_payload=oof_payload,
        source_hashes=source_hashes,
    )

    oof = None
    if use_oof_cache:
        try:
            oof = oof_cache.load_oof_cache(output_root or "results", fingerprint)
        except FileNotFoundError:
            oof = None
        except ValueError:
            raise

    artifact, oof, final_values, audit = fit_ensemble(
        config,
        runners,
        oof=oof,
        outer_cutoff_origin=None,
    )
    if use_oof_cache and output_root is not None:
        # cache key = outer fingerprint (includes member + source-file hashes)
        oof_for_cache = replace(oof, oof_fingerprint=fingerprint)
        oof_cache.save_oof_cache(output_root, oof_for_cache)

    combined = combine_members(artifact, final_values)
    return {
        "artifact": artifact,
        "oof": oof,
        "oof_fingerprint": fingerprint,
        "member_final_values": final_values,
        "combined_values": combined,
        "fused_oof_scores": audit.get("fused_oof_scores"),
        "forecast_times": runners[config.members[0].name].forecast_times(
            runners[config.members[0].name].origin
        ),
        "audit": audit,
    }


def _member_origin(member_config, registry):
    return resolve_origin(registry, member_config.validation.get("forecast_origin"))


__all__ = ["run_ensemble_config", "run_ensemble_config_file"]
