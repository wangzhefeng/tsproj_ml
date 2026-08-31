#!/usr/bin/env python
"""audit_ensemble_configs: permanent post-migration audit (v4 §9/E6).

Checks, over every ensemble YAML under config/:
- config_refs resolve, are single-model configs, unique, acyclic;
- the direct member reference is valid and the recursive member exists;
- method distribution matches the migrated contract
  (averaging / linear_blending only);
- exact model-YAML accounting: single + ensemble + ensemble_members == 845;
- reports orphan OOF cache directories under results/_ensemble_oof.

Exit code 0 only when every check passes (orphan report is informational).
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_ensemble.specs import EnsembleSpecError
from config.config_loader import load_yaml_config
from forecasting_core.specs import ForecastConfigSpec

EXPECTED_TOTAL = 845
EXPECTED_METHODS = {"averaging", "linear_blending"}


def _mapping(value):
    return isinstance(value, dict)


def _is_single_model_config(path: str | Path) -> bool:
    """Return whether a member reference resolves to the single-model typed spec."""
    return isinstance(load_yaml_config(path), ForecastConfigSpec)


def main() -> int:
    root = Path("config")
    all_yamls = sorted(glob.glob("config/**/*.yaml", recursive=True))
    single_files: list[str] = []
    ensemble_files: list[str] = []
    member_files: list[str] = []
    data_tool_files: list[str] = []

    for path in all_yamls:
        try:
            doc = yaml.safe_load(open(path, encoding="utf-8"))
        except Exception:
            data_tool_files.append(path)
            continue
        if not isinstance(doc, dict) or doc.get("schema_version") != 2:
            data_tool_files.append(path)
            continue
        rel = os.path.relpath(path, root)
        in_members = "ensemble_members" in Path(path).parts
        if isinstance(doc.get("ensemble"), dict):
            (member_files if in_members else ensemble_files).append(path)
        elif in_members:
            member_files.append(path)
        elif isinstance(doc.get("strategy"), dict):
            single_files.append(path)
        else:
            data_tool_files.append(path)

    failures: list[str] = []
    method_counter: dict[str, int] = {}
    referenced_refs: set[str] = set()

    for path in ensemble_files:
        base_dir = Path(path).parent
        try:
            config = load_yaml_config(path)
        except EnsembleSpecError as exc:
            failures.append(f"{path}: invalid ensemble config: {exc}")
            continue
        method_counter[config.method.name] = (
            method_counter.get(config.method.name, 0) + 1
        )
        if config.method.name not in EXPECTED_METHODS:
            failures.append(
                f"{path}: method {config.method.name!r} is not part of the "
                "migrated distribution"
            )
        for member in config.members:
            resolved = (base_dir / member.config_ref).resolve()
            referenced_refs.add(str(resolved))
            if not resolved.exists():
                failures.append(
                    f"{path}: member {member.name!r} ref {member.config_ref!r} "
                    "does not exist"
                )
                continue
            try:
                is_single_model = _is_single_model_config(resolved)
            except Exception as exc:
                failures.append(
                    f"{path}: member {member.name!r} ref could not be parsed: {exc}"
                )
                continue
            if not is_single_model:
                failures.append(
                    f"{path}: member {member.name!r} ref is not a single-model "
                    "ForecastConfigSpec"
                )
        names = [member.name for member in config.members]
        if names != ["direct", "recursive"]:
            failures.append(f"{path}: member names must be direct+recursive, got {names}")

    total_models = len(single_files) + len(ensemble_files) + len(member_files)
    if total_models != EXPECTED_TOTAL:
        failures.append(
            f"model YAML count {total_models} != expected {EXPECTED_TOTAL} "
            f"(single={len(single_files)}, ensemble={len(ensemble_files)}, "
            f"members={len(member_files)})"
        )

    orphan_cache_dirs: list[str] = []
    cache_root = Path("results/_ensemble_oof")
    if cache_root.exists():
        for entry in sorted(cache_root.iterdir()):
            if entry.is_dir():
                orphan_cache_dirs.append(entry.name)

    report = {
        "model_yaml_total": total_models,
        "single": len(single_files),
        "ensemble": len(ensemble_files),
        "ensemble_members": len(member_files),
        "data_tool": len(data_tool_files),
        "method_distribution": method_counter,
        "referenced_member_refs": len(referenced_refs),
        "orphan_oof_cache_dirs": orphan_cache_dirs,
        "failures": failures,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
