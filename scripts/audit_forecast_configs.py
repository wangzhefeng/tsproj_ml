#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a deterministic, read-only catalog of active canonical model YAMLs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config_loader import is_model_yaml, load_yaml_config  # noqa: E402
from forecasting_core.specs import ForecastConfigSpec  # noqa: E402
from model_ensemble.specs import EnsembleConfigSpec  # noqa: E402


REQUIRED_KEYS = frozenset(
    {
        "path",
        "schema_version",
        "config_kind",
        "result_identity",
        "fingerprint",
        "freq",
        "horizon",
        "targets",
        "training_scope",
        "series_id_cols",
        "probabilistic_mode",
        "quantiles",
        "strategy",
        "output_chunk_length",
        "model_type",
        "target_adapter",
        "ensemble_method",
        "member_count",
        "member_names",
        "oof_geometry",
        "validation_geometry",
        "lags",
        "scenario_subpath",
    }
)


def _common_row(path: Path, root: Path, cfg: Any) -> dict[str, Any]:
    validation_geometry = (
        asdict(cfg.validation.backtest)
        if cfg.validation.backtest is not None
        else None
    )
    return {
        "path": path.relative_to(root).as_posix(),
        "schema_version": cfg.schema_version,
        "result_identity": cfg.result_identity(),
        "fingerprint": cfg.fingerprint(),
        "freq": cfg.problem.freq,
        "horizon": cfg.problem.horizon,
        "targets": list(cfg.problem.targets),
        "training_scope": cfg.problem.training_scope,
        "series_id_cols": list(cfg.problem.series_id_cols),
        "probabilistic_mode": str(cfg.probabilistic.get("mode", "point")),
        "quantiles": list(cfg.probabilistic.get("quantiles", ())),
        "validation_geometry": validation_geometry,
        "scenario_subpath": str(cfg.output.get("scenario_subpath", "")),
    }


def _build_single_row(
    path: Path,
    root: Path,
    cfg: ForecastConfigSpec,
) -> dict[str, Any]:
    row = _common_row(path, root, cfg)
    strategy = cfg.strategy
    if strategy is None:
        raise AssertionError("ForecastConfigSpec requires strategy")
    row.update(
        {
            "config_kind": "single_model",
            "strategy": strategy.name.value,
            "output_chunk_length": strategy.output_chunk_length,
            "model_type": cfg.estimator.model_type,
            "target_adapter": cfg.estimator.target_adapter.value,
            "ensemble_method": None,
            "member_count": 0,
            "member_names": [],
            "oof_geometry": None,
            "lags": sorted(
                {
                    int(lag)
                    for values in cfg.features.target_lags.values()
                    for lag in values
                }
            ),
        }
    )
    return row


def _build_ensemble_row(
    path: Path,
    root: Path,
    cfg: EnsembleConfigSpec,
) -> dict[str, Any]:
    row = _common_row(path, root, cfg)
    row.update(
        {
            "config_kind": "ensemble",
            "strategy": None,
            "output_chunk_length": None,
            "model_type": None,
            "target_adapter": None,
            "ensemble_method": cfg.method.name,
            "member_count": len(cfg.members),
            "member_names": [member.name for member in cfg.members],
            "oof_geometry": cfg.oof.payload(),
            "lags": [],
        }
    )
    return row


def _build_row(path: Path, root: Path) -> dict[str, Any]:
    cfg = load_yaml_config(path)
    if isinstance(cfg, ForecastConfigSpec):
        row = _build_single_row(path, root, cfg)
    elif isinstance(cfg, EnsembleConfigSpec):
        row = _build_ensemble_row(path, root, cfg)
    else:
        raise TypeError(f"non-canonical model config rejected: {path}")
    if set(row) != set(REQUIRED_KEYS):
        raise AssertionError(f"catalog row schema drift for {path}")
    return row


def build_catalog(root: str | Path = "config") -> list[dict[str, Any]]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Config root does not exist: {root_path}")
    if not root_path.is_dir():
        raise ValueError(f"Config root must be a directory: {root_path}")
    paths = sorted(
        path
        for path in root_path.rglob("*.yaml")
        if is_model_yaml(path)
    )
    return [_build_row(path, root_path) for path in paths]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="config", help="Config tree to audit")
    parser.add_argument("--output", help="Write JSON to this path instead of stdout")
    args = parser.parse_args(argv)

    catalog = build_catalog(args.root)
    serialized = json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(serialized, encoding="utf-8")
    else:
        sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
