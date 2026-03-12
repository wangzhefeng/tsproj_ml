# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-03-12
# * Version     : 2.0.0
# * Description : CLI entry for ML time-series forecasting
# ***************************************************

import argparse
import datetime
import importlib
import json
import random
from typing import Any

from utils.log_util import logger


def _parse_bool_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean flag value: {value}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        logger.warning("[run.py] numpy is not installed, only Python random seed is set.")


def _load_config(config_module: str, config_class: str):
    module = importlib.import_module(config_module)
    if not hasattr(module, config_class):
        raise AttributeError(f"Config class '{config_class}' not found in module '{config_module}'.")
    config_cls = getattr(module, config_class)
    return config_cls()


def _apply_overrides(cfg, args):
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.data_path is not None:
        cfg.data_path = args.data_path
    if args.data is not None:
        cfg.data = args.data
    if args.target is not None:
        cfg.target = args.target
    if args.target_ts_feat is not None:
        cfg.target_ts_feat = args.target_ts_feat
    if args.freq is not None:
        cfg.freq = args.freq
    if args.freq_minutes is not None:
        cfg.freq_minutes = args.freq_minutes

    if args.model_type is not None:
        cfg.model_type = args.model_type
    if args.pred_method is not None:
        cfg.pred_method = args.pred_method

    if args.history_days is not None:
        cfg.history_days = args.history_days
    if args.predict_days is not None:
        cfg.predict_days = args.predict_days
    if args.window_days is not None:
        cfg.window_days = args.window_days

    if args.is_testing is not None:
        cfg.is_testing = _parse_bool_flag(args.is_testing)
    if args.is_forecasting is not None:
        cfg.is_forecasting = _parse_bool_flag(args.is_forecasting)

    if args.scale is not None:
        cfg.scale = _parse_bool_flag(args.scale)
    if args.encode_categorical_features is not None:
        cfg.encode_categorical_features = _parse_bool_flag(args.encode_categorical_features)
    if args.perform_tuning is not None:
        cfg.perform_tuning = _parse_bool_flag(args.perform_tuning)

    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.patience is not None:
        cfg.patience = args.patience

    if args.checkpoints_dir is not None:
        cfg.checkpoints_dir = args.checkpoints_dir
    if args.test_results_dir is not None:
        cfg.test_results_dir = args.test_results_dir
    if args.pred_results_dir is not None:
        cfg.pred_results_dir = args.pred_results_dir

    if args.now_time is not None:
        cfg.now_time = datetime.datetime.fromisoformat(args.now_time)

    if args.lags is not None:
        cfg.lags = [int(x.strip()) for x in args.lags.split(",") if x.strip()]

    if args.model_params is not None:
        loaded = json.loads(args.model_params)
        if not isinstance(loaded, dict):
            raise ValueError("--model-params must be a JSON object.")
        cfg.model_params = loaded

    return cfg


def args_parse():
    parser = argparse.ArgumentParser(description="Machine Learning Time Series Forecasting CLI")

    parser.add_argument(
        "--config-module",
        type=str,
        default="config.model_config",
        help="Python module path for config, e.g. config.model_config_lgbm_usmd_A",
    )
    parser.add_argument(
        "--config-class",
        type=str,
        default="ModelConfig_univariate",
        help="Config class name in config module",
    )

    parser.add_argument("--seed", type=int, default=2025)

    parser.add_argument("--is-testing", default=None, help="bool flag, supports 1/0/true/false")
    parser.add_argument("--is-forecasting", default=None, help="bool flag, supports 1/0/true/false")

    parser.add_argument("--model-type", type=str, default=None, help="lightgbm/xgboost/catboost")
    parser.add_argument("--pred-method", type=str, default=None)
    parser.add_argument("--model-params", type=str, default=None, help='JSON string, e.g. \'{"num_leaves":63}\'')

    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--target-ts-feat", type=str, default=None)
    parser.add_argument("--freq", type=str, default=None)
    parser.add_argument("--freq-minutes", type=int, default=None)

    parser.add_argument("--history-days", type=int, default=None)
    parser.add_argument("--predict-days", type=int, default=None)
    parser.add_argument("--window-days", type=int, default=None)
    parser.add_argument("--lags", type=str, default=None, help="comma separated list, e.g. 288,576")

    parser.add_argument("--scale", default=None, help="bool flag, supports 1/0/true/false")
    parser.add_argument("--encode-categorical-features", default=None, help="bool flag")
    parser.add_argument("--perform-tuning", default=None, help="bool flag")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)

    parser.add_argument("--now-time", type=str, default=None, help="ISO datetime, e.g. 2025-12-27T00:00:00")

    parser.add_argument("--checkpoints-dir", type=str, default=None)
    parser.add_argument("--test-results-dir", type=str, default=None)
    parser.add_argument("--pred-results-dir", type=str, default=None)

    return parser.parse_args()


def run(args):
    cfg = _load_config(args.config_module, args.config_class)
    cfg = _apply_overrides(cfg, args)

    logger.info(
        "[run.py] config=%s.%s, model_type=%s, pred_method=%s, is_testing=%s, is_forecasting=%s",
        args.config_module,
        args.config_class,
        cfg.model_type,
        cfg.pred_method,
        cfg.is_testing,
        cfg.is_forecasting,
    )

    from main import Model

    model = Model(cfg)
    model.run()


def main():
    args = args_parse()
    _set_seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
