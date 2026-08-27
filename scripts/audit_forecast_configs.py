#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a deterministic, read-only catalog of forecast model YAML configs."""

from __future__ import annotations

import argparse
import calendar
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from config.config_loader import load_yaml_config  # noqa: E402
from models.multistep.resolve import _resolve_block_size  # noqa: E402
from models.multistep.spec import RolloutFamily, get_strategy_spec  # noqa: E402
from scripts.check_model_configs import is_model_yaml  # noqa: E402


REQUIRED_KEYS = frozenset(
    {
        "path",
        "legacy_method",
        "method_code",
        "base_config",
        "target",
        "target_series_numeric_features",
        "predict_type",
        "direct_strategy",
        "block_size",
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


def _is_tool_yaml(path: Path) -> bool:
    stem = path.stem.lower()
    return stem.startswith(("aggregate", "outlier")) or "detect" in stem


def _base_config(path: Path) -> str:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return str(payload.get("base_config") or "config.univariate_config")


def _resolve_horizon(cfg: Any) -> int:
    if str(getattr(cfg, "horizon_mode", "fixed_steps") or "fixed_steps").lower() != "calendar_month":
        return int(cfg.predict_steps)
    forecast_start = cfg.now_time.date() + timedelta(days=1)
    return calendar.monthrange(forecast_start.year, forecast_start.month)[1]


def _build_row(path: Path, root: Path) -> dict[str, Any]:
    cfg = load_yaml_config(path)
    legacy_method = str(cfg.pred_method)
    strategy = get_strategy_spec(legacy_method)
    horizon = _resolve_horizon(cfg)
    lags = [int(value) for value in (cfg.lags or [])]
    effective_block_size = None
    if strategy.rollout == RolloutFamily.DIRREC:
        effective_block_size = _resolve_block_size(cfg, horizon, tuple(lags))

    return {
        "path": path.relative_to(root).as_posix(),
        "legacy_method": legacy_method,
        "method_code": strategy.code,
        "base_config": _base_config(path),
        "target": cfg.target,
        "target_series_numeric_features": list(cfg.target_series_numeric_features or []),
        "predict_type": cfg.predict_type,
        "direct_strategy": cfg.direct_strategy,
        "block_size": int(cfg.block_size or 0),
        "effective_block_size": effective_block_size,
        "blend_weight_strategy": cfg.blend_weight_strategy,
        "blend_weights": list(cfg.blend_weights or []),
        "endogenous_backfill_strategy": cfg.endogenous_backfill_strategy,
        "enable_global_training": bool(cfg.enable_global_training),
        "series_id_feature": cfg.series_id_feature,
        "horizon": horizon,
        "lags": lags,
        "scenario_subpath": cfg.scenario_subpath,
        "setting_suffix": cfg.setting_suffix,
    }


def build_catalog(root: str | Path = "config") -> list[dict[str, Any]]:
    root_path = Path(root)
    paths = sorted(
        path
        for path in root_path.rglob("*.yaml")
        if not _is_tool_yaml(path) and is_model_yaml(path)
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
