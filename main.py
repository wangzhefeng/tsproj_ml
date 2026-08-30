# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-11
# * Version     : 2.0
# * Description : 基于机器学习回归器的时间序列预测框架（canonical 入口）
# ***************************************************

"""Canonical forecasting entrypoint."""

import os
import sys
import warnings
from collections.abc import Mapping
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from config.config_loader import load_yaml_config
from model_forecasting.runtime import run_canonical_config
from model_forecasting.specs import ForecastConfigSpec

warnings.filterwarnings("ignore")

LOGGING_LABEL = Path(__file__).stem
os.environ.setdefault("LOG_NAME", LOGGING_LABEL)
from utils.log_util import logger


class CanonicalModel:
    """Canonical entrypoint backed only by the canonical runtime."""

    def __init__(self, args: ForecastConfigSpec):
        if not isinstance(args, ForecastConfigSpec):
            raise TypeError("CanonicalModel requires ForecastConfigSpec")
        self.args = args
        if args.strategy is None and args.ensemble is None:
            raise ValueError("canonical config requires strategy or ensemble")
        output_identity = args.output.get("identity", {})
        if isinstance(output_identity, Mapping):
            self.scenario_subpath = str(
                output_identity.get(
                    "scenario_subpath",
                    args.output.get("scenario_subpath", "canonical"),
                )
                or "canonical"
            ).strip("/")
        else:
            self.scenario_subpath = "canonical"
        if not self.scenario_subpath:
            self.scenario_subpath = "canonical"
        self.setting = args.result_identity()
        self.log_prefix = f"[{self.setting}]"
        self.result = None

    def run(self):
        directories = self.args.output.get("directories")
        legacy_directories = {
            "checkpoints_dir",
            "test_results_dir",
            "pred_results_dir",
        }
        if isinstance(directories, Mapping) or legacy_directories.issubset(self.args.output):
            self.result = run_canonical_config(self.args)
        else:
            output_root = Path(str(self.args.output.get("results_root", "results")))
            self.result = run_canonical_config(self.args, output_root=output_root)
        return self.result


def build_model(args: ForecastConfigSpec) -> CanonicalModel:
    """Build the only executable runtime; non-canonical configs are rejected."""
    from model_ensemble.specs import EnsembleConfigSpec

    if isinstance(args, EnsembleConfigSpec):
        raise TypeError(
            "reference-based ensemble configs run through model_ensemble.runtime "
            "(see run.py dispatch); main.py executes single-model configs only"
        )
    if not isinstance(args, ForecastConfigSpec):
        raise TypeError(
            "only canonical ForecastConfigSpec can enter the runtime; "
            f"got {type(args).__name__}"
        )
    return CanonicalModel(args)


CONFIG_YAML = "config/aidc_load_15min_short/route_A/baseline/st_usmr_mean.yaml"


def main() -> None:
    """Load one canonical schema YAML and execute the canonical runtime."""
    from utils.runtime_env import ensure_runtime_environment

    ensure_runtime_environment()
    args = load_yaml_config(CONFIG_YAML)
    model = build_model(args)
    try:
        model.run()
    except Exception as exc:
        logger.error(
            f"{model.log_prefix} Pipeline FAILED: {exc}",
            exc_info=True,
        )
        raise
    logger.info(f"{model.log_prefix} {'#' * 85}")
    logger.info(f"{model.log_prefix} 模型预测流程完成！")
    logger.info(f"{model.log_prefix} {'#' * 85}")


if __name__ == "__main__":
    main()
