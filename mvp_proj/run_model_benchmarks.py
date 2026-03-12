# -*- coding: utf-8 -*-
"""Batch benchmark runner for all 7 forecasting strategies."""

import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from config.univariate_config import ModelConfig_univariate, ModelConfig_multivariate
from main import Model


UNIVARIATE_METHODS = [
    "univariate-single-multistep-direct-output",
    "univariate-single-multistep-direct",
    "univariate-single-multistep-recursive",
    "univariate-single-multistep-direct-recursive",
]

MULTIVARIATE_METHODS = [
    "multivariate-single-multistep-direct",
    "multivariate-single-multistep-recursive",
    "multivariate-single-multistep-direct-recursive",
]


def _common_overrides(args) -> None:
    """Set stable experiment defaults for reproducible local testing."""
    args.is_testing = True
    args.is_forecasting = False
    args.enable_ensemble = False
    args.perform_tuning = False
    args.scale = False
    args.disable_plotting = True
    args.history_days = 12
    args.window_days = 10
    args.predict_days = 1
    args.model_type = "lightgbm"
    args.model_params = {"n_estimators": 80, "learning_rate": 0.05}


def _configure_univariate(method: str):
    args = ModelConfig_univariate()
    _common_overrides(args)
    args.pred_method = method
    # Ensure lag-based methods have lag features enabled in practice.
    args.lags = [1 * 288, 2 * 288, 3 * 288]
    return args


def _configure_multivariate(method: str):
    args = ModelConfig_multivariate()
    _common_overrides(args)
    args.pred_method = method
    args.lags = [96, 192, 288]
    return args


def _extract_mean_metrics(csv_path: Path) -> Dict[str, float]:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if df.empty:
        return {}
    if "time_range" in df.columns and (df["time_range"] == "均值").any():
        row = df[df["time_range"] == "均值"].iloc[0]
    else:
        row = df.iloc[-1]
    metrics = {}
    for key in ["R2", "MSE", "RMSE", "MAE", "MAPE", "MAPE Accuracy"]:
        metrics[key] = float(row[key]) if key in row and pd.notna(row[key]) else float("nan")
    return metrics


def run_all() -> pd.DataFrame:
    records: List[Dict] = []

    plan = [("univariate", m) for m in UNIVARIATE_METHODS] + [("multivariate", m) for m in MULTIVARIATE_METHODS]
    for group, method in plan:
        try:
            args = _configure_univariate(method) if group == "univariate" else _configure_multivariate(method)
            runner = Model(args)
            runner.run()
            metrics = _extract_mean_metrics(args.test_results_dir.joinpath("test_scores_df.csv"))
            records.append(
                {
                    "group": group,
                    "dataset": args.data_path,
                    "method": method,
                    **metrics,
                    "status": "ok",
                    "results_dir": str(args.test_results_dir),
                }
            )
        except Exception as exc:
            records.append(
                {
                    "group": group,
                    "dataset": getattr(args, "data_path", ""),
                    "method": method,
                    "status": f"failed: {exc}",
                    "results_dir": str(getattr(args, "test_results_dir", "")),
                }
            )

    return pd.DataFrame(records)


def main():
    summary_df = run_all()
    out_dir = Path("saved_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir.joinpath(f"benchmark_summary_{ts}.csv")
    md_path = out_dir.joinpath(f"benchmark_summary_{ts}.md")

    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(summary_df.to_string(index=False), encoding="utf-8")

    print(f"Saved benchmark summary csv: {csv_path}")
    print(f"Saved benchmark summary md : {md_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
