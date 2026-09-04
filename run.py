# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-11
# * Version     : 2.0.0
# * Description : CLI entry for ML time-series forecasting
# ***************************************************


import argparse
import os
import random
import sys
from pathlib import Path

from config.config_loader import load_yaml_config
from forecasting_core.specs import ForecastConfigSpec
from model_ensemble.contracts import EnsembleRuntimeServices
from model_ensemble.runtime import run_ensemble_config_file
from model_ensemble.specs import EnsembleConfigSpec
from model_forecasting.runtime import (
    CanonicalBaseModelRunner,
    persist_model_bundle,
    run_canonical_config,
)
from model_forecasting.resource_planner import plan_ensemble_resources

ENSEMBLE_RUNTIME_SERVICES = EnsembleRuntimeServices(
    runner_factory=CanonicalBaseModelRunner,
    persist_bundle=persist_model_bundle,
    plan_resources=plan_ensemble_resources,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger

def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        logger.warning("[run.py] numpy is not installed, only Python random seed is set.")

def _load_config(config_yaml: str):
    """加载 YAML 配置文件并返回配置实例。base_config 从 YAML 内部读取。"""
    return load_yaml_config(config_yaml)

def args_parse():
    parser = argparse.ArgumentParser(description="Machine Learning Time Series Forecasting CLI")

    parser.add_argument(
        "--config-yaml",
        type=str,
        required=True,
        help="Canonical schema-2 YAML config path.",
    )

    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Unified root directory for model, test, and forecast artifacts.",
    )

    return parser.parse_args()

def run(args):
    cfg = _load_config(args.config_yaml)

    if isinstance(cfg, EnsembleConfigSpec):
        logger.info(
            "[run.py] config=%s schema=2 kind=ensemble method=%s",
            args.config_yaml,
            cfg.method.name,
        )
        result = run_ensemble_config_file(
            args.config_yaml,
            output_root=args.output_root,
            services=ENSEMBLE_RUNTIME_SERVICES,
        )
        fused = result.get("fused_oof_scores") or {}
        point_scores = fused.get("point")
        if point_scores is not None:
            aggregate = point_scores[point_scores["scope"] == "aggregate"]
            if not aggregate.empty:
                row = aggregate.iloc[0]
                logger.info(
                    "[run.py] ensemble fused OOF: MAE=%.4f RMSE=%.4f MAPE=%.4f",
                    row["MAE"],
                    row["RMSE"],
                    row["MAPE"],
                )
        logger.info(
            "[run.py] ensemble run complete: oof_fingerprint=%s",
            result["oof_fingerprint"],
        )
        return result

    if not isinstance(cfg, ForecastConfigSpec):
        raise TypeError(
            "only canonical ForecastConfigSpec can enter the runtime; "
            f"got {type(cfg).__name__}"
        )
    if cfg.strategy is None:
        raise ValueError("canonical config requires a strategy")
    identity = cfg.strategy.name.value
    logger.info(
        "[run.py] config=%s schema=2 model_type=%s identity=%s",
        args.config_yaml,
        cfg.estimator.model_type,
        identity,
    )

    return run_canonical_config(cfg, output_root=args.output_root)

def main():
    args = args_parse()
    _set_seed(args.seed)
    run(args)

if __name__ == "__main__":
    main()
