# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from data_process.outlier_handling import empty_train_outlier_report
from model_evaluation.point import evaluate_point_forecasts
from model_forecasting.results import (
    CanonicalResultReader,
    write_backtest_results,
)
from model_forecasting.tensors import PointForecastTensor


DEFAULT_RESULTS_ROOT = Path(
    "results/results_test/aidc_electricity_computility/electricity/2026-06-11"
)
DEFAULT_SCENARIOS = ["AIDC/route_A", "AIDC/route_B"]
MODEL_A_NAME = "lightgbm-df_power-usmd-15"
MODEL_B_NAME = "lightgbm-df_power-usmdp-2"
FUSED_MODEL_NAME = "lightgbm-df_power-usmd-15__lightgbm-df_power-usmdp-2__avg_0505"


def _read_curve_df(results_root: Path, scenario: str, model_name: str) -> pd.DataFrame:
    curve_path = results_root.joinpath(scenario, model_name, "cv_plot_df.csv")
    curve_df = CanonicalResultReader.read_backtest(curve_path)
    return curve_df.sort_values(
        ["window", "series_id", "time", "target"]
    ).reset_index(drop=True)


def _validate_alignment(curve_a_df: pd.DataFrame, curve_b_df: pd.DataFrame, scenario: str):
    merged_df = curve_a_df.merge(
        curve_b_df,
        on=["series_id", "time", "target", "window"],
        how="inner",
        suffixes=("_a", "_b"),
    )
    if merged_df.empty:
        raise ValueError(f"{scenario}: no overlapping timestamps between source models")
    if not np.allclose(
        merged_df["actual_value_a"].to_numpy(dtype=float),
        merged_df["actual_value_b"].to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise ValueError(f"{scenario}: actual_value mismatch between source models")
    return merged_df


def _build_fused_curve_df(merged_df: pd.DataFrame) -> pd.DataFrame:
    fused_df = merged_df[["series_id", "time", "target", "window"]].copy()
    fused_df["actual_value"] = merged_df["actual_value_a"].astype(float)
    fused_df["predict_value"] = (
        merged_df["predict_value_a"].astype(float).to_numpy()
        + merged_df["predict_value_b"].astype(float).to_numpy()
    ) / 2.0
    fused_df["plot_valid"] = (
        merged_df["plot_valid_a"].astype(bool)
        & merged_df["plot_valid_b"].astype(bool)
    )
    return fused_df


def _frame_tensor(frame: pd.DataFrame, value_column: str) -> PointForecastTensor:
    series_ids = tuple(dict.fromkeys(frame["series_id"].tolist()))
    times = pd.DatetimeIndex(sorted(pd.to_datetime(frame["time"]).unique()))
    targets = tuple(dict.fromkeys(frame["target"].astype(str).tolist()))
    indexed = frame.set_index(["series_id", "time", "target"])
    values = np.empty((len(series_ids), len(times), len(targets)), dtype=float)
    for series_index, series_id in enumerate(series_ids):
        for time_index, timestamp in enumerate(times):
            for target_index, target in enumerate(targets):
                values[series_index, time_index, target_index] = float(
                    indexed.loc[(series_id, timestamp, target), value_column]
                )
    return PointForecastTensor(
        values=values,
        series_ids=series_ids,
        forecast_times=times,
        targets=targets,
    )


def _build_scores(fused_curve_df: pd.DataFrame) -> pd.DataFrame:
    scores = []
    targets = tuple(dict.fromkeys(fused_curve_df["target"].astype(str)))
    weights = {target: 1.0 / len(targets) for target in targets}
    for window, window_df in fused_curve_df.groupby("window", sort=True):
        scores.append(
            evaluate_point_forecasts(
                _frame_tensor(window_df, "actual_value"),
                _frame_tensor(window_df, "predict_value"),
                aggregate_weighting=weights,
                window=int(window),
            )
        )
    return pd.concat(scores, ignore_index=True)


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
    test_scores_df = _build_scores(fused_curve_df)

    fused_dir = results_root.joinpath(scenario, FUSED_MODEL_NAME)
    fused_dir.mkdir(parents=True, exist_ok=True)
    targets = tuple(dict.fromkeys(fused_curve_df["target"].astype(str)))
    write_backtest_results(
        fused_dir,
        fused_curve_df,
        test_scores_df,
        aggregate_weighting={target: 1.0 / len(targets) for target in targets},
        metadata={"fusion_weights": [0.5, 0.5]},
    )
    empty_train_outlier_report().to_csv(
        fused_dir / "train_outlier_report.csv",
        index=False,
        encoding="utf_8_sig",
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
