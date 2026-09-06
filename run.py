# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-11
# * Version     : 3.0.0
# * Description : 统一 canonical 入口（单模型 + 引用式 Ensemble 分派）
# ***************************************************

"""Canonical forecasting entrypoint: single-model and reference-based ensemble."""

import argparse
import os
import random
import warnings
from pathlib import Path

from config.config_loader import load_yaml_config
from forecasting_core.specs import ForecastConfigSpec
from model_ensemble.contracts import EnsembleRuntimeServices
from model_ensemble.runtime import run_ensemble_config_file
from model_ensemble.specs import EnsembleConfigSpec
from model_pipeline.runner import (
    CanonicalBaseModelRunner,
    persist_model_bundle,
    run_canonical_config,
)
from model_performance.resource_planner import (
    plan_ensemble_resources,
    runtime_budget_for_config,
)
from utils.runtime_env import ensure_runtime_environment

warnings.filterwarnings("ignore")

ENSEMBLE_RUNTIME_SERVICES = EnsembleRuntimeServices(
    runner_factory=CanonicalBaseModelRunner,
    persist_bundle=persist_model_bundle,
    plan_resources=plan_ensemble_resources,
    resolve_budget=runtime_budget_for_config,
)

# global variable
LOGGING_LABEL = Path(__file__).stem
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


class CanonicalModel:
    """Single-model canonical wrapper: logging identity + result handle."""

    def __init__(self, args: ForecastConfigSpec):
        if not isinstance(args, ForecastConfigSpec):
            raise TypeError("CanonicalModel requires ForecastConfigSpec")
        self.args = args
        self.setting = args.result_identity()
        self.log_prefix = f"[{self.setting}]"
        self.result = None

    def run(self, output_root=None):
        # output_root=None 时由 runtime 按 output.directories/results_root 解析，
        # 与 main.py 历史分支语义完全一致（model_pipeline.lifecycle._output_paths）。
        self.result = run_canonical_config(self.args, output_root=output_root)
        return self.result


def build_model(args: ForecastConfigSpec) -> CanonicalModel:
    """Build the single-model runtime wrapper; non-canonical configs are rejected."""
    if isinstance(args, EnsembleConfigSpec):
        raise TypeError(
            "reference-based ensemble configs are dispatched to "
            "model_ensemble.runtime by run.py; build_model accepts "
            "single-model configs only"
        )
    if not isinstance(args, ForecastConfigSpec):
        raise TypeError(
            "only canonical ForecastConfigSpec can enter the runtime; "
            f"got {type(args).__name__}"
        )
    return CanonicalModel(args)


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

    model = build_model(cfg)
    try:
        model.run(output_root=args.output_root)
    except Exception as exc:
        logger.error(
            f"{model.log_prefix} Pipeline FAILED: {exc}",
            exc_info=True,
        )
        raise
    logger.info(f"{model.log_prefix} {'#' * 85}")
    logger.info(f"{model.log_prefix} 模型预测流程完成！")
    logger.info(f"{model.log_prefix} {'#' * 85}")
    return model.result

def main():
    args = args_parse()
    ensure_runtime_environment()
    _set_seed(args.seed)
    run(args)

if __name__ == "__main__":
    main()
