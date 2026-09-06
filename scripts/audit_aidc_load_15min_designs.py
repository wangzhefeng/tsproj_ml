#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""真实编译三个 AIDC 15min 场景的全部活动单模型训练设计。

本门禁不训练 estimator；每份配置选择一个标签完整可见的真实训练 origin，
通过生产 ``SourceRegistry`` 与 ``model_pipeline.supervised_design`` 设计 builder 完成
信息集物化、provider 构造、全部策略 call 的特征编译及标签构造。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config_loader import is_model_yaml, load_yaml_config  # noqa: E402
from data_loading import BUILTIN_GENERATORS, SourceRegistry  # noqa: E402
from forecasting_core.specs import ForecastConfigSpec  # noqa: E402
from model_pipeline.supervised_design import probe_training_design  # noqa: E402


SCENARIOS = (
    "aidc_load_15min_daily",
    "aidc_load_15min_rolling",
    "aidc_load_15min_short",
)
EXPECTED_SINGLE_MODELS = 4617


def _probe_origin(config: ForecastConfigSpec) -> pd.Timestamp:
    forecast_origin = pd.Timestamp(config.validation["forecast_origin"])
    if not isinstance(forecast_origin, pd.Timestamp):
        raise ValueError("validation.forecast_origin must be a valid timestamp")
    offset = pd.tseries.frequencies.to_offset(config.problem.freq)
    return forecast_origin - config.problem.horizon * offset


def audit_aidc_load_15min_designs(
    *,
    repository_root: str | Path = ROOT,
) -> dict[str, Any]:
    """Compile one real training design for every active single-model YAML."""
    repo = Path(repository_root).resolve()
    config_root = repo / "config"
    paths = sorted(
        path
        for scenario in SCENARIOS
        for path in (config_root / scenario).rglob("*.yaml")
        if is_model_yaml(path)
    )

    single_model_count = 0
    compiled_count = 0
    failures: list[dict[str, str]] = []
    for path in paths:
        config = load_yaml_config(path)
        if not isinstance(config, ForecastConfigSpec):
            continue
        single_model_count += 1
        relative = path.relative_to(repo).as_posix()
        try:
            registry = SourceRegistry(
                config.data,
                repo,
                generators=BUILTIN_GENERATORS,
            )
            probe_training_design(config, registry, _probe_origin(config))
        except Exception as exc:  # 审计必须汇总全部失败配置
            failures.append(
                {
                    "config": relative,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        else:
            compiled_count += 1

    return {
        "scenario_count": len(SCENARIOS),
        "single_model_count": single_model_count,
        "compiled_count": compiled_count,
        "failure_count": len(failures),
        "failures": failures,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = audit_aidc_load_15min_designs()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "AIDC 15min design audit: "
            f"scenarios={payload['scenario_count']} "
            f"single={payload['single_model_count']} "
            f"compiled={payload['compiled_count']} "
            f"failures={payload['failure_count']}"
        )
        for failure in payload["failures"]:
            print(
                f"FAIL {failure['config']}: "
                f"{failure['error_type']}: {failure['message']}"
            )
    return 1 if (
        payload["single_model_count"] != EXPECTED_SINGLE_MODELS
        or payload["compiled_count"] != EXPECTED_SINGLE_MODELS
        or payload["failure_count"]
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
