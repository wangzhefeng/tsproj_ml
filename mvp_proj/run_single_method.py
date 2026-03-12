# -*- coding: utf-8 -*-
"""Run one forecasting method and save test results."""

import argparse

from config.univariate_config import ModelConfig_univariate, ModelConfig_multivariate
from main import Model


def build_args(group: str, method: str):
    if group == "univariate":
        args = ModelConfig_univariate()
        args.lags = [1 * 288, 2 * 288, 3 * 288]
    else:
        args = ModelConfig_multivariate()
        args.lags = [96, 192, 288]

    args.pred_method = method
    args.is_testing = True
    args.is_forecasting = False
    args.enable_ensemble = False
    args.perform_tuning = False
    args.disable_plotting = True
    args.history_days = 12
    args.window_days = 10
    args.predict_days = 1
    args.model_type = "lightgbm"
    args.model_params = {"n_estimators": 80, "learning_rate": 0.05}
    args.scale = False
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["univariate", "multivariate"], required=True)
    parser.add_argument("--method", required=True)
    parsed = parser.parse_args()

    args = build_args(parsed.group, parsed.method)
    model = Model(args)
    model.run()
    print(f"Done: {parsed.group} | {parsed.method} -> {args.test_results_dir}")


if __name__ == "__main__":
    main()
