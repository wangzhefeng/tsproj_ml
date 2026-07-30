# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from data_provider.outlier_handling import empty_train_outlier_report
from models.ModelTesting import Tester


DEFAULT_RESULTS_ROOT = Path(
    "results/results_test/aidc_electricity_computility/electricity/2026-06-11"
)
DEFAULT_SCENARIOS = ["AIDC/route_A", "AIDC/route_B"]
MODEL_A_NAME = "lightgbm-df_power-usmd-15"
MODEL_B_NAME = "lightgbm-df_power-usmdp-2"
FUSED_MODEL_NAME = "lightgbm-df_power-usmd-15__lightgbm-df_power-usmdp-2__avg_0505"


def _read_curve_df(results_root: Path, scenario: str, model_name: str) -> pd.DataFrame:
    curve_path = results_root.joinpath(scenario, model_name, "cv_plot_df.csv")
    curve_df = pd.read_csv(curve_path)
    curve_df["time"] = pd.to_datetime(curve_df["time"])
    return curve_df.sort_values("time").reset_index(drop=True)


def _validate_alignment(curve_a_df: pd.DataFrame, curve_b_df: pd.DataFrame, scenario: str):
    merged_df = curve_a_df.merge(
        curve_b_df,
        on="time",
        how="inner",
        suffixes=("_a", "_b"),
    )
    if merged_df.empty:
        raise ValueError(f"{scenario}: no overlapping timestamps between source models")
    if not np.allclose(
        merged_df["Y_trues_a"].to_numpy(dtype=float),
        merged_df["Y_trues_b"].to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise ValueError(f"{scenario}: Y_trues mismatch between source models")
    return merged_df


def _build_fused_curve_df(merged_df: pd.DataFrame) -> pd.DataFrame:
    fused_df = pd.DataFrame()
    fused_df["time"] = merged_df["time"]
    fused_df["Y_trues"] = merged_df["Y_trues_a"].astype(float)
    fused_df["Y_preds"] = (
        merged_df["Y_preds_a"].astype(float).to_numpy()
        + merged_df["Y_preds_b"].astype(float).to_numpy()
    ) / 2.0
    return fused_df


def _build_daily_scores_and_plot_df(fused_curve_df: pd.DataFrame):
    daily_scores = []
    daily_plots = []
    for window, (_, day_df) in enumerate(fused_curve_df.groupby(fused_curve_df["time"].dt.date), start=1):
        day_df = day_df.sort_values("time").reset_index(drop=True)
        df_history_test = pd.DataFrame({"time": day_df["time"]})
        y_true = day_df["Y_trues"].to_numpy(dtype=float)
        y_pred = day_df["Y_preds"].to_numpy(dtype=float)
        daily_scores.append(
            Tester._evaluate_score(
                y_test=y_true,
                y_pred=y_pred,
                window=window,
                df_history_test=df_history_test,
                log_prefix="[fusion]",
            )
        )
        daily_plots.append(
            Tester._evaluate_result(
                y_test=y_true,
                y_pred=y_pred,
                df_history_test=df_history_test,
                log_prefix="[fusion]",
            )
        )

    test_scores_df = pd.concat(daily_scores, axis=0).reset_index(drop=True)
    test_scores_median = test_scores_df.drop(columns=["time_range"]).median(numeric_only=True).to_frame().T
    test_scores_median["time_range"] = "中位数"
    test_scores_median = test_scores_median[["time_range"] + [c for c in test_scores_df.columns if c != "time_range"]]
    test_scores_df = pd.concat([test_scores_df, test_scores_median], axis=0, ignore_index=True)
    cv_plot_df = pd.concat(daily_plots, axis=0).reset_index(drop=True)
    return test_scores_df, cv_plot_df


def fuse_scenario_results(
    results_root: Path,
    scenario: str,
    model_a_name: str = MODEL_A_NAME,
    model_b_name: str = MODEL_B_NAME,
) -> Path:
    curve_a_df = _read_curve_df(results_root, scenario, model_a_name)
    curve_b_df = _read_curve_df(results_root, scenario, model_b_name)
    merged_df = _validate_alignment(curve_a_df, curve_b_df, scenario)
    fused_curve_df = _build_fused_curve_df(merged_df)
    test_scores_df, cv_plot_df = _build_daily_scores_and_plot_df(fused_curve_df)

    fused_dir = results_root.joinpath(scenario, FUSED_MODEL_NAME)
    fused_dir.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(test_results_dir=fused_dir)
    Tester.test_results_save(
        args=args,
        log_prefix="[fusion]",
        test_scores_df=test_scores_df,
        cv_plot_df=cv_plot_df,
        train_outlier_report=empty_train_outlier_report(),
    )
    return fused_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse AIDC route test results with fixed 0.5/0.5 weights.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--scenario", action="append", dest="scenarios")
    return parser.parse_args()


def main():
    args = parse_args()
    scenarios = args.scenarios or DEFAULT_SCENARIOS
    for scenario in scenarios:
        fused_dir = fuse_scenario_results(args.results_root, scenario)
        print(f"saved fused result: {fused_dir}")


if __name__ == "__main__":
    main()
