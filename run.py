# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-03-12
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

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
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


def _load_config(config_yaml: str):
    """加载 YAML 配置文件并返回配置实例。base_config 从 YAML 内部读取。"""
    return load_yaml_config(config_yaml)


def _apply_overrides(cfg, args):
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.data_path is not None:
        cfg.data_path = args.data_path
    if args.target is not None:
        cfg.target = args.target
    if args.target_ts_feat is not None:
        cfg.target_ts_feat = args.target_ts_feat
    if args.freq is not None:
        cfg.freq = args.freq

    if args.model_type is not None:
        cfg.model_type = args.model_type
    if args.pred_method is not None:
        cfg.pred_method = args.pred_method

    if args.history_days is not None:
        cfg.history_days = args.history_days
    if args.predict_steps is not None:
        cfg.predict_steps = args.predict_steps
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
    if args.enable_data_augmentation is not None:
        cfg.enable_data_augmentation = _parse_bool_flag(args.enable_data_augmentation)
    if args.augmentation_ratio is not None:
        cfg.augmentation_ratio = args.augmentation_ratio
    if args.augmentation_feature_noise_std is not None:
        cfg.augmentation_feature_noise_std = args.augmentation_feature_noise_std
    if args.augmentation_target_noise_std is not None:
        cfg.augmentation_target_noise_std = args.augmentation_target_noise_std
    if args.augmentation_random_state is not None:
        cfg.augmentation_random_state = args.augmentation_random_state
    if args.enable_feature_selection is not None:
        cfg.enable_feature_selection = _parse_bool_flag(args.enable_feature_selection)
    if args.feature_selection_method is not None:
        cfg.feature_selection_method = args.feature_selection_method
    if args.feature_selection_max_features is not None:
        cfg.feature_selection_max_features = args.feature_selection_max_features
    if args.feature_selection_min_features is not None:
        cfg.feature_selection_min_features = args.feature_selection_min_features
    if args.enable_auto_learning_rate is not None:
        cfg.enable_auto_learning_rate = _parse_bool_flag(args.enable_auto_learning_rate)
    if args.auto_lr_min is not None:
        cfg.auto_lr_min = args.auto_lr_min
    if args.auto_lr_max is not None:
        cfg.auto_lr_max = args.auto_lr_max
    if args.huber_delta is not None:
        cfg.huber_delta = args.huber_delta
    if args.enable_train_outlier_handling is not None:
        cfg.enable_train_outlier_handling = _parse_bool_flag(args.enable_train_outlier_handling)
    if args.train_outlier_method is not None:
        cfg.train_outlier_method = args.train_outlier_method
    if args.high_outlier_threshold is not None:
        cfg.high_outlier_threshold = args.high_outlier_threshold
    if args.high_outlier_max_run_points is not None:
        cfg.high_outlier_max_run_points = args.high_outlier_max_run_points
    if args.drop_outlier_max_run_points is not None:
        cfg.drop_outlier_max_run_points = args.drop_outlier_max_run_points
    if args.drop_rebound_min_abs_diff is not None:
        cfg.drop_rebound_min_abs_diff = args.drop_rebound_min_abs_diff
    if args.low_outlier_threshold is not None:
        cfg.low_outlier_threshold = args.low_outlier_threshold
    if args.low_outlier_max_run_points is not None:
        cfg.low_outlier_max_run_points = args.low_outlier_max_run_points
    if args.rise_outlier_max_run_points is not None:
        cfg.rise_outlier_max_run_points = args.rise_outlier_max_run_points
    if args.rise_rebound_min_abs_diff is not None:
        cfg.rise_rebound_min_abs_diff = args.rise_rebound_min_abs_diff

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
        "--config-yaml",
        type=str,
        required=True,
        help="YAML config file path (base_config is read from inside the YAML).",
    )

    parser.add_argument("--seed", type=int, default=2025)

    parser.add_argument("--is-testing", default=None, help="bool flag, supports 1/0/true/false")
    parser.add_argument("--is-forecasting", default=None, help="bool flag, supports 1/0/true/false")

    parser.add_argument("--model-type", type=str, default=None, help="lightgbm/xgboost/catboost")
    parser.add_argument("--pred-method", type=str, default=None)
    parser.add_argument("--model-params", type=str, default=None, help='JSON string, e.g. \'{"num_leaves":63}\'')

    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--target-ts-feat", type=str, default=None)
    parser.add_argument("--freq", type=str, default=None)

    parser.add_argument("--history-days", type=int, default=None)
    parser.add_argument("--predict-steps", type=int, default=None)
    parser.add_argument("--window-days", type=int, default=None)
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

    logger.info(
        "[run.py] config=%s, model_type=%s, pred_method=%s, is_testing=%s, is_forecasting=%s",
        args.config_yaml,
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
