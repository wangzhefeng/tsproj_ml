#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a deterministic, read-only catalog of forecast model YAML configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config_loader import is_model_yaml, load_yaml_config  # noqa: E402
from model_forecasting.specs import ColumnRole, ForecastConfigSpec  # noqa: E402


REQUIRED_KEYS = frozenset(
    {
        "path",
        "canonical_identity",
        "base_config",
        "target",
        "target_series_numeric_features",
        "predict_type",
        "direct_strategy",
        "effective_block_size",
        "blend_weight_strategy",
        "blend_weights",
        "endogenous_backfill_strategy",
        "enable_global_training",
        "series_id_feature",
        "horizon",
        "lags",
        "scenario_subpath",
        "setting_suffix",
    }
)


def _build_row(path: Path, root: Path) -> dict[str, Any]:
    cfg = load_yaml_config(path)
    if not isinstance(cfg, ForecastConfigSpec):
        raise TypeError(f"non-canonical model config rejected: {path}")
    return _build_model_row(path, root, cfg)


def _build_model_row(
    path: Path,
    root: Path,
    cfg: ForecastConfigSpec,
) -> dict[str, Any]:
    identity = (
        "ensemble" if cfg.ensemble is not None else cfg.strategy.name.value
    )
    lags = sorted(
        {
            int(lag)
            for values in cfg.features.target_lags.values()
            for lag in values
        }
    )
    observed_sources = [
        source
        for source in cfg.data.sources
        if any(column.role is ColumnRole.OBSERVED_PAST for column in source.columns)
    ]
    provider = next(
        (source.provider for source in observed_sources if source.provider is not None),
        "none",
    )
    direct_layout = str(cfg.features.transformations.get("direct_layout", "standard"))
    effective_block_size = (
        cfg.strategy.output_chunk_length
        if cfg.strategy is not None
        else None
    )
    blend_weights = list(cfg.ensemble.weights) if cfg.ensemble is not None else []
    blend_strategy = cfg.ensemble.method if cfg.ensemble is not None else "none"
    series_id_feature = (
        cfg.problem.series_id_cols[0]
        if len(cfg.problem.series_id_cols) == 1
        else None
    )
    return {
        "path": path.relative_to(root).as_posix(),
        "canonical_identity": identity,
        "base_config": "canonical",
        "target": cfg.problem.targets[0],
        "target_series_numeric_features": list(
            cfg.data.role_columns(ColumnRole.OBSERVED_PAST)
        ),
        "predict_type": str(cfg.probabilistic.get("mode", "point")),
        "direct_strategy": direct_layout,
        "effective_block_size": effective_block_size,
        "blend_weight_strategy": blend_strategy,
        "blend_weights": blend_weights,
        "endogenous_backfill_strategy": provider,
        "enable_global_training": cfg.problem.is_global,
        "series_id_feature": series_id_feature,
        "horizon": cfg.problem.horizon,
        "lags": lags,
        "scenario_subpath": str(cfg.output.get("scenario_subpath", "")),
        "setting_suffix": str(cfg.output.get("setting_suffix", "")),
    }


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
