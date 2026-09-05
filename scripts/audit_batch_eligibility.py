# -*- coding: utf-8 -*-
"""Audit supervised compiler batch eligibility for active model configs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, cast

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config_loader import load_yaml_config  # noqa: E402
from feature_engineering.compiler import FeatureCompiler  # noqa: E402
from forecasting_core.specs import ForecastConfigSpec  # noqa: E402
from model_ensemble.specs import EnsembleConfigSpec  # noqa: E402
from model_training.strategies import target_plan_for_config  # noqa: E402


def _request_for_config(config: ForecastConfigSpec):
    from data_loading import InformationSetRequest

    origin = cast(
        pd.Timestamp,
        pd.Timestamp(config.validation["forecast_origin"]),
    )
    return InformationSetRequest(
        forecast_origin=origin,
        forecast_times=pd.date_range(
            origin,
            periods=config.problem.horizon + 1,
            freq=config.problem.freq,
        )[1:],
        series_ids=(),
        target_access="history_only",
    )


def _estimated_training_origins(config: ForecastConfigSpec) -> int:
    validation = config.validation
    history_steps = validation.get("history_steps")
    if isinstance(history_steps, int) and not isinstance(history_steps, bool):
        return history_steps
    train_window_days = validation.get("train_window_days")
    if isinstance(train_window_days, int) and not isinstance(train_window_days, bool):
        return train_window_days
    return 1


def audit_configs(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Return one structured eligibility row per canonical model config."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        config = load_yaml_config(path)
        if isinstance(config, EnsembleConfigSpec):
            continue
        if not isinstance(config, ForecastConfigSpec):
            raise TypeError(f"unexpected config type for {path}: {type(config)!r}")
        if config.strategy is None:
            raise ValueError(f"forecast config requires strategy: {path}")
        plan = target_plan_for_config(config)
        call_steps = tuple(
            coordinates[0].horizon_step for coordinates in plan.call_coordinates
        )
        eligibility = FeatureCompiler(config).batch_eligibility(
            (_request_for_config(config),),
            horizon_steps=call_steps,
        )
        row = eligibility.to_payload()
        origin_count = _estimated_training_origins(config)
        row.update(
            {
                "path": str(path),
                "strategy": config.strategy.name,
                "model_type": config.estimator.model_type,
                "horizon": config.problem.horizon,
                "origin_count": origin_count,
                "estimated_origin_call_count": origin_count
                * eligibility.call_count,
            }
        )
        rows.append(row)
    return rows


def _canonical_model_paths(root: Path) -> tuple[Path, ...]:
    paths = []
    for path in sorted((root / "config").rglob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("schema_version") == 2:
            paths.append(path)
    return tuple(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    paths = _canonical_model_paths(args.root)
    rows = audit_configs(paths)
    reason_counts: dict[str, int] = {}
    for row in rows:
        for reason in row["reason_codes"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    payload = {
        "canonical_yaml_count": len(paths),
        "forecast_config_count": len(rows),
        "ensemble_config_count": len(paths) - len(rows),
        "eligible_count": sum(bool(row["eligible"]) for row in rows),
        "fallback_count": sum(not bool(row["eligible"]) for row in rows),
        "reason_counts": dict(sorted(reason_counts.items())),
        "rows": rows,
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
