# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-11
# * Version     : 2.0.0
# * Description : CLI entry for ML time-series forecasting
# ***************************************************


import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import argparse
import datetime
import json
import random
from typing import Any
from config.config_loader import load_yaml_config
from model_forecasting.specs import ForecastConfigSpec, parse_model_config

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

def _apply_overrides(cfg, args):
    """Apply CLI overrides; ensemble configs pass through untouched (v4 E6)."""
    from model_ensemble.specs import EnsembleConfigSpec

    if isinstance(cfg, EnsembleConfigSpec):
        if any(
            getattr(args, name, None) is not None
            for name in (
                "model_type",
                "pred_method",
                "predict_steps",
                "history_length",
                "window_length",
                "lags",
                "model_params",
            )
        ):
            raise ValueError(
                "CLI model overrides are not valid for ensemble configs; "
                "edit the model_ensemble/member YAML instead"
            )
        return cfg
    if not isinstance(cfg, ForecastConfigSpec):
        raise TypeError(
            "only canonical ForecastConfigSpec can enter the runtime; "
            f"got {type(cfg).__name__}"
        )
    return _apply_canonical_overrides(cfg, args)

def _apply_canonical_overrides(
    cfg: ForecastConfigSpec,
    args,
) -> ForecastConfigSpec:
    payload = cfg.canonical_payload()
    value = lambda name: getattr(args, name, None)
    output = payload["output"]
    identity = output.setdefault(
        "identity",
        {
            "scenario_subpath": str(output.pop("scenario_subpath", "") or ""),
            "setting_suffix": str(output.pop("setting_suffix", "") or ""),
        },
    )
    if not isinstance(identity, dict):
        raise TypeError("canonical output.identity must be a mapping")
    directories = output.setdefault("directories", {})
    if not isinstance(directories, dict):
        raise TypeError("canonical output.directories must be a mapping")
    legacy_directories = {
        "checkpoints_dir": "checkpoints",
        "test_results_dir": "tests",
        "pred_results_dir": "forecast",
    }
    for legacy_name, canonical_name in legacy_directories.items():
        if legacy_name in output:
            directories.setdefault(canonical_name, str(output.pop(legacy_name)))

    target_sources = [
        source
        for source in payload["data"]["sources"]
        if any(column["role"] == "target" for column in source["columns"])
    ]
    if len(target_sources) != 1:
        raise ValueError("CLI overrides require exactly one canonical target source")
    target_source = target_sources[0]

    if value("data_dir") is not None or value("data_path") is not None:
        current = Path(target_source["history_path"])
        directory = Path(value("data_dir")) if value("data_dir") is not None else current.parent
        filename = Path(value("data_path")) if value("data_path") is not None else Path(current.name)
        target_source["history_path"] = (directory / filename).as_posix()
    if value("target") is not None:
        if len(payload["problem"]["targets"]) != 1:
            raise ValueError("--target only supports a single-target canonical config")
        old_target = payload["problem"]["targets"][0]
        new_target = str(value("target"))
        payload["problem"]["targets"] = [new_target]
        for column in target_source["columns"]:
            if column["name"] == old_target and column["role"] == "target":
                column["name"] = new_target
        lags = payload["features"]["target_lags"].pop(old_target, [])
        payload["features"]["target_lags"][new_target] = lags
    if value("target_ts_feat") is not None:
        payload["problem"]["time_col"] = str(value("target_ts_feat"))
        target_source["time_col"] = str(value("target_ts_feat"))
    if value("freq") is not None:
        payload["problem"]["freq"] = str(value("freq"))
    if value("model_type") is not None:
        payload["estimator"]["model_type"] = str(value("model_type"))
    if value("pred_method") is not None:
        if isinstance(payload.get("ensemble"), dict):
            raise ValueError("--pred-method cannot replace an ensemble config")
        payload["strategy"] = {"name": str(value("pred_method"))}
    if value("predict_steps") is not None:
        payload["problem"]["horizon"] = int(value("predict_steps"))
    if value("history_length") is not None:
        payload["validation"]["history_length"] = int(value("history_length"))
    if value("window_length") is not None:
        payload["validation"]["window_length"] = int(value("window_length"))
    if value("now_time") is not None:
        payload["validation"]["forecast_origin"] = datetime.datetime.fromisoformat(
            value("now_time")
        ).isoformat()
    if value("lags") is not None:
        lags = [int(item.strip()) for item in value("lags").split(",") if item.strip()]
        payload["features"]["target_lags"] = {
            target: list(lags) for target in payload["problem"]["targets"]
        }
    if value("model_params") is not None:
        params = json.loads(value("model_params"))
        if not isinstance(params, dict):
            raise ValueError("--model-params must be a JSON object")
        payload["estimator"]["params"] = params
    directory_overrides = {
        "checkpoints_dir": "checkpoints",
        "test_results_dir": "tests",
        "pred_results_dir": "forecast",
    }
    if any(value(field) is not None for field in directory_overrides):
        for field, canonical_name in directory_overrides.items():
            if value(field) is not None:
                directories[canonical_name] = str(value(field))
        for legacy_field in directory_overrides:
            output.pop(legacy_field, None)

    unsupported = [
        name
        for name in (
            "scale",
            "encode_categorical_features",
            "perform_tuning",
            "enable_data_augmentation",
            "enable_feature_selection",
            "enable_auto_learning_rate",
            "enable_train_outlier_handling",
        )
        if value(name) is not None
    ]
    if unsupported:
        raise ValueError(
            "legacy CLI enhancement overrides are not valid for the canonical schema: "
            + ", ".join(unsupported)
        )
    return parse_model_config(payload, source="<cli-overrides>")

def args_parse():
    parser = argparse.ArgumentParser(description="Machine Learning Time Series Forecasting CLI")

    parser.add_argument(
        "--config-yaml",
        type=str,
        required=True,
        help="YAML config file path (base_config is read from inside the YAML).",
    )

    parser.add_argument("--seed", type=int, default=2025)

    parser.add_argument("--model-type", type=str, default=None, help="lightgbm/xgboost/catboost")
    parser.add_argument("--pred-method", type=str, default=None)
    parser.add_argument("--model-params", type=str, default=None, help='JSON string, e.g. \'{"num_leaves":63}\'')

    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--target-ts-feat", type=str, default=None)
    parser.add_argument("--freq", type=str, default=None)

    parser.add_argument("--history-length", type=int, default=None)
    parser.add_argument("--predict-steps", type=int, default=None)
    parser.add_argument("--window-length", type=int, default=None)
    parser.add_argument("--lags", type=str, default=None, help="comma separated list, e.g. 288,576")

    parser.add_argument("--scale", default=None, help="bool flag, supports 1/0/true/false")
    parser.add_argument("--encode-categorical-features", default=None, help="bool flag")
    parser.add_argument("--perform-tuning", default=None, help="bool flag")
    parser.add_argument("--enable-data-augmentation", default=None, help="bool flag")
    parser.add_argument("--augmentation-ratio", type=float, default=None)
    parser.add_argument("--augmentation-feature-noise-std", type=float, default=None)
    parser.add_argument("--augmentation-target-noise-std", type=float, default=None)
    parser.add_argument("--augmentation-random-state", type=int, default=None)
    parser.add_argument("--enable-feature-selection", default=None, help="bool flag")
    parser.add_argument("--feature-selection-method", type=str, default=None, help="f_regression/mutual_info")
    parser.add_argument("--feature-selection-max-features", type=int, default=None)
    parser.add_argument("--feature-selection-min-features", type=int, default=None)
    parser.add_argument("--enable-auto-learning-rate", default=None, help="bool flag")
    parser.add_argument("--auto-lr-min", type=float, default=None)
    parser.add_argument("--auto-lr-max", type=float, default=None)
    parser.add_argument("--huber-delta", type=float, default=None)
    parser.add_argument("--enable-train-outlier-handling", default=None, help="bool flag")
    parser.add_argument("--train-outlier-method", type=str, default=None)
    parser.add_argument("--high-outlier-threshold", type=float, default=None)
    parser.add_argument("--high-outlier-max-run-points", type=int, default=None)
    parser.add_argument("--drop-outlier-max-run-points", type=int, default=None)
    parser.add_argument("--drop-rebound-min-abs-diff", type=float, default=None)
    parser.add_argument("--low-outlier-threshold", type=float, default=None)
    parser.add_argument("--low-outlier-max-run-points", type=int, default=None)
    parser.add_argument("--rise-outlier-max-run-points", type=int, default=None)
    parser.add_argument("--rise-rebound-min-abs-diff", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)

    parser.add_argument("--now-time", type=str, default=None, help="ISO datetime, e.g. 2025-12-27T00:00:00")

    parser.add_argument("--checkpoints-dir", type=str, default=None)
    parser.add_argument("--test-results-dir", type=str, default=None)
    parser.add_argument("--pred-results-dir", type=str, default=None)

    return parser.parse_args()

def run(args):
    cfg = _load_config(args.config_yaml)
    cfg = _apply_overrides(cfg, args)

    from model_ensemble.specs import EnsembleConfigSpec

    if isinstance(cfg, EnsembleConfigSpec):
        # reference-based ensemble: dispatch to the ensemble runtime (v4 E6)
        from model_ensemble.runtime import run_ensemble_config_file

        logger.info(
            "[run.py] config=%s schema=2 kind=ensemble method=%s",
            args.config_yaml,
            cfg.method.name,
        )
        result = run_ensemble_config_file(
            args.config_yaml,
            output_root=None,
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

    if cfg.strategy is None:
        raise ValueError("canonical config requires a strategy")
    identity = cfg.strategy.name.value
    logger.info(
        "[run.py] config=%s schema=2 model_type=%s identity=%s",
        args.config_yaml,
        cfg.estimator.model_type,
        identity,
    )

    from main import build_model

    model = build_model(cfg)
    model.run()

def main():
    args = args_parse()
    _set_seed(args.seed)
    run(args)

if __name__ == "__main__":
    main()
