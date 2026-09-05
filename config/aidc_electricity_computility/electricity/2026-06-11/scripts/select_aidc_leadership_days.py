# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from model_forecasting.results import CanonicalResultReader


DEFAULT_RESULTS_ROOT = Path(
    "results/results_test/aidc_electricity_computility/electricity/2026-06-11"
)
DEFAULT_OUTPUT_ROOT = Path(
    "results/leadership_selection/aidc_electricity_computility/2026-06-11"
)
DEFAULT_REPORT_ROOT = Path(
    "results/reports/aidc_electricity_computility/2026-06-11"
)
DEFAULT_CANDIDATE_TOP_K = 5
DEFAULT_TAIL_POINTS = 48
DEFAULT_FILL_METHOD = "linear_ffill_bfill"
SCENARIOS = [
    "A1_01a",
    "A1_201",
    "A1_IT",
    "A3_01e",
    "AIDC/route_A",
    "AIDC/route_B",
]
MODEL_CANDIDATES = [
    "lightgbm-df_power-usmd-15",
    "lightgbm-df_power-usmdp-2",
]
ARCHIVE_SCENARIOS = {"A1_201", "A3_01e"}
POOR_FIT_SCENARIOS = {"A1_IT", "AIDC/route_A", "AIDC/route_B"}
FINAL_SUMMARY_COLUMNS = [
    "scenario",
    "selected_model",
    "selected_date",
    "time_range",
    "R2",
    "MSE",
    "RMSE",
    "MAE",
    "MAPE",
    "MAPE Accuracy",
    "MAPE Threshold",
    "MAPE Valid Points",
    "MAPE Excluded Points",
    "MAPE Excluded Ratio",
    "corr",
    "tail_MAE",
    "tail_bias",
    "plot_nan",
    "selection_reason",
    "note",
]
REPORT_METRICS_COLUMNS = [
    "scenario",
    "R2",
    "MSE",
    "RMSE",
    "MAE",
    "MAPE",
    "MAPE Accuracy",
]
REPORT_SELECTIONS = {
    "A1_01a": "candidates/A1_01a/2026-05-24-lightgbm-df_power-usmd-15.png",
    "A1_201": "candidates/A1_201/2026-04-24-lightgbm-df_power-usmd-15.png",
    "A1_IT": "final/A1_IT/leadership_day_plot.png",
    "A3_01e": "candidates/A3_01e/2026-04-14-lightgbm-df_power-usmdp-2.png",
    "AIDC/route_A": "candidates/AIDC/route_A/2026-04-24-lightgbm-df_power-usmd-15.png",
    "AIDC/route_B": "final/AIDC/route_B/leadership_day_plot.png",
}
REPORT_PLOT_META = {
    "A1_01a": {
        "title": "A1 楼 01a 集群功率预测图",
        "filename": "A1 楼 01a 集群功率预测图.png",
    },
    "A1_201": {
        "title": "A1 楼 201 机房功率预测图",
        "filename": "A1 楼 201 机房功率预测图.png",
    },
    "A1_IT": {
        "title": "A1 楼 IT 总负荷功率预测图",
        "filename": "A1 楼 IT 总负荷功率预测图.png",
    },
    "A3_01e": {
        "title": "A3 楼 01e 集群功率预测图",
        "filename": "A3 楼 01e 集群功率预测图.png",
    },
    "AIDC/route_A": {
        "title": "AIDC A 路总负荷功率预测图",
        "filename": "AIDC A 路总负荷功率预测图.png",
    },
    "AIDC/route_B": {
        "title": "AIDC B 路总负荷功率预测图",
        "filename": "AIDC B 路总负荷功率预测图.png",
    },
}
CANDIDATE_REPORT_DEFAULTS = {
    "selection_reason": "manual_candidate_pick",
    "note": "manually selected from candidate pool",
}
REPORT_FONT_CANDIDATES = [
    "Hiragino Sans GB",
    "STHeiti",
    "Songti SC",
    "Arial Unicode MS",
]


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 1.0 if np.allclose(y_true, y_pred, equal_nan=True) else 0.0
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)


def fill_display_series(series: pd.Series, fill_method: str = DEFAULT_FILL_METHOD) -> pd.Series:
    filled = pd.Series(series, copy=True).astype(float)
    if fill_method != DEFAULT_FILL_METHOD:
        raise ValueError(f"Unsupported fill_method: {fill_method}")
    filled = filled.interpolate(method="linear", limit_direction="both")
    filled = filled.ffill().bfill()
    return filled


def resolve_report_font() -> font_manager.FontProperties:
    for font_name in REPORT_FONT_CANDIDATES:
        try:
            font_path = font_manager.findfont(font_name, fallback_to_default=False)
        except ValueError:
            continue
        if Path(font_path).exists():
            return font_manager.FontProperties(fname=font_path)
    raise ValueError(
        "No usable Chinese font found for report plots. "
        f"Tried: {', '.join(REPORT_FONT_CANDIDATES)}"
    )


def _read_daily_scores(results_root: Path, scenario: str, model_name: str) -> pd.DataFrame:
    score_path = results_root.joinpath(scenario, model_name, "test_scores_df.csv")
    score_df = CanonicalResultReader.read_scores(score_path)
    if {"window", "scope", "target", "MAE", "RMSE", "MAPE"}.issubset(score_df):
        daily_df = score_df[score_df["scope"] == "target"].copy()
        curve_df = CanonicalResultReader.read_backtest(
            results_root.joinpath(scenario, model_name, "cv_plot_df.csv"),
        )
        ranges = (
            curve_df.groupby(["window", "target"], sort=True)["time"]
            .agg(["min", "max"])
            .reset_index()
        )
        ranges["time_range"] = (
            pd.to_datetime(ranges["min"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            + "~"
            + pd.to_datetime(ranges["max"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        )
        daily_df = daily_df.merge(
            ranges[["window", "target", "time_range"]],
            on=["window", "target"],
            how="left",
            validate="one_to_one",
        )
        daily_df["MSE"] = np.square(pd.to_numeric(daily_df["RMSE"], errors="coerce"))
        daily_df["R2"] = np.nan
        daily_df["MAPE Accuracy"] = daily_df.get("Accuracy", np.nan)
        daily_df["MAPE Threshold"] = np.nan
        daily_df["MAPE Valid Points"] = daily_df.get("Valid Points", 0)
        daily_df["MAPE Excluded Points"] = (
            pd.to_numeric(daily_df.get("n_points", 0), errors="coerce")
            - pd.to_numeric(daily_df["MAPE Valid Points"], errors="coerce")
        )
        denominator = pd.to_numeric(daily_df.get("n_points", 0), errors="coerce")
        daily_df["MAPE Excluded Ratio"] = np.where(
            denominator > 0,
            daily_df["MAPE Excluded Points"] / denominator,
            np.nan,
        )
    else:
        daily_df = score_df[
            score_df["time_range"].astype(str).str.contains("~", na=False)
        ].copy()
    if daily_df.empty:
        raise ValueError(f"No daily score rows found: scenario={scenario}, model={model_name}")
    numeric_columns = [
        "R2",
        "MSE",
        "RMSE",
        "MAE",
        "MAPE",
        "MAPE Accuracy",
        "MAPE Threshold",
        "MAPE Valid Points",
        "MAPE Excluded Points",
        "MAPE Excluded Ratio",
    ]
    for column in numeric_columns:
        if column in daily_df.columns:
            daily_df[column] = pd.to_numeric(daily_df[column], errors="coerce")
    return daily_df


def _extract_selected_date(time_range: str) -> str:
    return str(time_range).split("~", 1)[0].split(" ", 1)[0]


def load_day_curve(
    results_root: Path, scenario: str, model_name: str, selected_date: str
) -> pd.DataFrame:
    curve_path = results_root.joinpath(scenario, model_name, "cv_plot_df.csv")
    curve_df = CanonicalResultReader.read_backtest(curve_path)
    targets = tuple(dict.fromkeys(curve_df["target"].astype(str)))
    if len(targets) != 1:
        raise ValueError(
            f"Leadership selection requires one target, got {targets}"
        )
    selected_day_df = curve_df[
        curve_df["time"].dt.strftime("%Y-%m-%d") == selected_date
    ].copy()
    if len(selected_day_df) != 288:
        raise ValueError(
            "Expected 288 points for selected day: "
            f"scenario={scenario}, model={model_name}, selected_date={selected_date}, "
            f"actual_points={len(selected_day_df)}"
        )
    return selected_day_df.sort_values("time").reset_index(drop=True)


def _compute_candidate_features(
    results_root: Path,
    scenario: str,
    model_name: str,
    score_row: pd.Series,
    day_curve_df: pd.DataFrame,
    tail_points: int,
) -> dict:
    y_true = day_curve_df["actual_value"].astype(float).to_numpy()
    y_pred = day_curve_df["predict_value"].astype(float).to_numpy()
    tail_true = y_true[-tail_points:]
    tail_pred = y_pred[-tail_points:]
    errors = np.abs(y_pred - y_true)
    tail_errors = np.abs(tail_pred - tail_true)

    candidate = score_row.to_dict()
    candidate["scenario"] = scenario
    candidate["selected_model"] = model_name
    candidate["selected_date"] = _extract_selected_date(str(score_row["time_range"]))
    candidate["corr"] = _safe_corr(y_true, y_pred)
    candidate["tail_MAE"] = float(tail_errors.mean())
    candidate["tail_bias"] = float(tail_pred.mean() - tail_true.mean())
    candidate["plot_nan"] = int((~day_curve_df["plot_valid"].astype(bool)).sum())
    candidate["display_true_nan"] = int(day_curve_df["actual_value"].isna().sum())
    candidate["display_pred_nan"] = int(day_curve_df["predict_value"].isna().sum())
    candidate["daily_MAE_from_curve"] = float(errors.mean())
    candidate["original_plot_path"] = str(
        results_root.joinpath(scenario, model_name, "test_prediction.png")
    )
    return candidate


def build_candidate_pool(
    results_root: Path,
    scenario: str,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    tail_points: int = DEFAULT_TAIL_POINTS,
) -> pd.DataFrame:
    candidates = []
    for model_name in MODEL_CANDIDATES:
        daily_df = _read_daily_scores(results_root, scenario, model_name)
        for _, score_row in daily_df.iterrows():
            selected_date = _extract_selected_date(str(score_row["time_range"]))
            day_curve_df = load_day_curve(results_root, scenario, model_name, selected_date)
            candidates.append(
                _compute_candidate_features(
                    results_root,
                    scenario,
                    model_name,
                    score_row,
                    day_curve_df,
                    tail_points,
                )
            )

    candidate_df = pd.DataFrame(candidates)
    if candidate_df.empty:
        raise ValueError(f"No candidate rows built for scenario={scenario}")

    index_sets = []
    index_sets.append(
        candidate_df.sort_values(["MAPE", "MAE", "RMSE", "time_range"]).head(candidate_top_k).index
    )
    index_sets.append(
        candidate_df.sort_values(["corr", "MAPE", "MAE"], ascending=[False, True, True]).head(candidate_top_k).index
    )
    index_sets.append(
        candidate_df.sort_values(["tail_MAE", "MAPE", "MAE"]).head(candidate_top_k).index
    )
    candidate_df = candidate_df.assign(abs_tail_bias=candidate_df["tail_bias"].abs())
    index_sets.append(
        candidate_df.sort_values(["abs_tail_bias", "corr", "MAPE"], ascending=[True, False, True]).head(candidate_top_k).index
    )

    keep_indexes = []
    seen = set()
    for index_set in index_sets:
        for idx in index_set.tolist():
            if idx not in seen:
                keep_indexes.append(idx)
                seen.add(idx)

    candidate_df = candidate_df.loc[keep_indexes].copy().reset_index(drop=True)
    candidate_df = candidate_df.sort_values(["MAPE", "MAE", "RMSE", "time_range"]).reset_index(drop=True)
    return candidate_df


def _selection_tuple(scenario: str, row: pd.Series) -> tuple:
    if scenario == "A1_01a":
        return (
            float(row["tail_MAE"]),
            abs(float(row["tail_bias"])),
            -float(row["corr"]),
            float(row["MAPE"]),
            float(row["MAE"]),
            str(row["time_range"]),
        )
    if scenario in {"A1_201", "A3_01e"}:
        return (
            -float(row["corr"]),
            float(row["tail_MAE"]),
            abs(float(row["tail_bias"])),
            float(row["MAPE"]),
            float(row["MAE"]),
            str(row["time_range"]),
        )
    return (
        -float(row["corr"]),
        abs(float(row["tail_bias"])),
        float(row["tail_MAE"]),
        float(row["MAE"]),
        float(row["MAPE"]),
        str(row["time_range"]),
    )


def _selection_reason(scenario: str, selected_row: pd.Series, metric_best_row: pd.Series) -> str:
    if scenario == "A1_01a" and str(selected_row["time_range"]) != str(metric_best_row["time_range"]):
        return "lower_tail_error_than_metric_best"
    if scenario in ARCHIVE_SCENARIOS and str(selected_row["time_range"]) != str(metric_best_row["time_range"]):
        return "better_shape_match_than_previous_pick"
    if scenario in POOR_FIT_SCENARIOS:
        return "least_bad_visual_fit_among_two_models"
    return "best_metric_and_visual_balance"


def _selection_note(scenario: str, row: pd.Series) -> str:
    if scenario in POOR_FIT_SCENARIOS:
        return "relative best among two candidate models; overall fit still weak"
    if scenario == "A1_01a":
        return "selected to reduce tail drift instead of raw lowest MAPE"
    if scenario in ARCHIVE_SCENARIOS:
        return "reselected after shape-focused candidate review"
    return "selected after metric prescreen and visual proxy review"


def _plot_title(selection: dict) -> str:
    return (
        f"{selection['scenario']} | {selection['selected_model']} | {selection['selected_date']} | "
        f"MAPE={float(selection['MAPE']):.6f}, MAE={float(selection['MAE']):.3f}, "
        f"corr={float(selection['corr']):.3f}, tail_MAE={float(selection['tail_MAE']):.3f}"
    )


def _plot_day_curve(
    output_path: Path,
    selection: dict,
    day_curve_df: pd.DataFrame,
    fill_method: str,
) -> None:
    plot_df = day_curve_df.sort_values("time").copy()
    plot_df["display_true"] = fill_display_series(plot_df["actual_value"], fill_method=fill_method)
    plot_df["display_pred"] = fill_display_series(plot_df["predict_value"], fill_method=fill_method)
    if plot_df["display_true"].isna().any() or plot_df["display_pred"].isna().any():
        raise ValueError(f"Display series still contain NaN for scenario={selection['scenario']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(16, 6))
    plt.plot(plot_df["time"], plot_df["display_true"], label="Trues", lw=1.7)
    plt.plot(plot_df["time"], plot_df["display_pred"], label="Preds", lw=1.7, ls="-.")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(_plot_title(selection))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def _plot_report_day_curve(
    output_path: Path,
    scenario: str,
    day_curve_df: pd.DataFrame,
    fill_method: str,
) -> None:
    plot_df = day_curve_df.sort_values("time").copy()
    plot_df["display_true"] = fill_display_series(plot_df["actual_value"], fill_method=fill_method)
    plot_df["display_pred"] = fill_display_series(plot_df["predict_value"], fill_method=fill_method)
    if plot_df["display_true"].isna().any() or plot_df["display_pred"].isna().any():
        raise ValueError(f"Display series still contain NaN for report scenario={scenario}")

    x_values = np.arange(1, len(plot_df) + 1)
    font_prop = resolve_report_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(16, 6))
    plt.plot(x_values, plot_df["display_true"], label="真实值", lw=1.7)
    plt.plot(x_values, plot_df["display_pred"], label="预测值", lw=1.7, ls="-.")
    plt.legend(prop=font_prop)
    plt.xlabel("时间", fontproperties=font_prop)
    plt.ylabel("功率（kW）", fontproperties=font_prop)
    plt.title(REPORT_PLOT_META[scenario]["title"], fontproperties=font_prop)
    plt.xlim(1, len(plot_df))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def _copy_original_plot(results_root: Path, scenario: str, model_name: str, output_path: Path) -> None:
    original_plot = results_root.joinpath(scenario, model_name, "test_prediction.png")
    if not original_plot.exists():
        original_plot = results_root.joinpath(
            scenario,
            model_name,
            "target_plots",
            "power.png",
        )
    if original_plot.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_plot, output_path)


def _candidate_plot_name(row: pd.Series) -> str:
    model_slug = str(row["selected_model"]).replace("/", "_")
    return f"{row['selected_date']}-{model_slug}.png"


def _parse_candidate_filename(path: Path) -> tuple[str, str]:
    stem = path.stem
    prefix = f"{stem[:10]}-"
    if len(stem) <= 11 or not prefix.startswith("20"):
        raise ValueError(f"Cannot parse candidate filename: {path}")
    return stem[:10], stem[11:]


def _require_existing_file(path: Path, scenario: str) -> Path:
    if not path.is_file():
        raise ValueError(f"Missing selected source image for {scenario}: {path}")
    return path


def _load_selected_candidate_row(selection_root: Path, scenario: str, relative_path: str) -> dict:
    source_path = _require_existing_file(selection_root.joinpath(relative_path), scenario)
    selected_date, selected_model = _parse_candidate_filename(source_path)
    details_path = source_path.parent.joinpath("candidate_details.csv")
    details_df = pd.read_csv(details_path)
    matched = details_df[
        (details_df["selected_date"] == selected_date)
        & (details_df["selected_model"] == selected_model)
    ].copy()
    if len(matched) != 1:
        raise ValueError(
            f"Expected exactly one candidate detail row for {scenario}, "
            f"selected_date={selected_date}, selected_model={selected_model}, got {len(matched)}"
        )
    row = matched.iloc[0].to_dict()
    for key, value in CANDIDATE_REPORT_DEFAULTS.items():
        row.setdefault(key, value)
    return {column: row[column] for column in FINAL_SUMMARY_COLUMNS}


def _load_selected_final_row(selection_root: Path, scenario: str, relative_path: str) -> dict:
    source_path = _require_existing_file(selection_root.joinpath(relative_path), scenario)
    summary_path = selection_root.joinpath("final", "leadership_day_summary.csv")
    summary_df = pd.read_csv(summary_path)
    matched = summary_df[summary_df["scenario"] == scenario].copy()
    if len(matched) != 1:
        raise ValueError(f"Expected exactly one final summary row for {scenario}, got {len(matched)}")
    row = matched.iloc[0].to_dict()
    if str(source_path.parent.relative_to(selection_root.joinpath("final"))) != scenario:
        raise ValueError(f"Final source path does not align with scenario={scenario}: {source_path}")
    return {column: row[column] for column in FINAL_SUMMARY_COLUMNS}


def build_report_package(
    selection_root: Path = DEFAULT_OUTPUT_ROOT,
    report_root: Path = DEFAULT_REPORT_ROOT,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    fill_method: str = DEFAULT_FILL_METHOD,
) -> pd.DataFrame:
    report_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for scenario, relative_path in REPORT_SELECTIONS.items():
        if relative_path.startswith("candidates/"):
            summary_row = _load_selected_candidate_row(selection_root, scenario, relative_path)
        elif relative_path.startswith("final/"):
            summary_row = _load_selected_final_row(selection_root, scenario, relative_path)
        else:
            raise ValueError(f"Unsupported report selection source for {scenario}: {relative_path}")

        day_curve_df = load_day_curve(
            results_root,
            scenario,
            summary_row["selected_model"],
            summary_row["selected_date"],
        )
        scenario_report_dir = report_root.joinpath(scenario)
        legacy_plot = scenario_report_dir.joinpath("leadership_day_plot.png")
        if legacy_plot.exists():
            legacy_plot.unlink()
        target_path = scenario_report_dir.joinpath(REPORT_PLOT_META[scenario]["filename"])
        _plot_report_day_curve(target_path, scenario, day_curve_df, fill_method)
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows, columns=FINAL_SUMMARY_COLUMNS)
    summary_df.to_csv(report_root.joinpath("leadership_day_summary.csv"), index=False, encoding="utf-8")
    export_report_metrics_summary(report_root, summary_df)
    return summary_df


def export_report_metrics_summary(
    report_root: Path = DEFAULT_REPORT_ROOT,
    summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if summary_df is None:
        summary_path = report_root.joinpath("leadership_day_summary.csv")
        summary_df = pd.read_csv(summary_path)
    metrics_df = summary_df.loc[:, REPORT_METRICS_COLUMNS].copy()
    metrics_df.to_csv(
        report_root.joinpath("leadership_day_summary_metrics.csv"),
        index=False,
        encoding="utf-8",
    )
    return metrics_df


def _archive_previous_selection(output_root: Path, scenario: str) -> None:
    previous_summary_path = output_root.joinpath("leadership_day_summary.csv")
    previous_plot_path = output_root.joinpath(scenario, "leadership_day_plot.png")
    if not previous_summary_path.exists() or not previous_plot_path.exists():
        return

    previous_df = pd.read_csv(previous_summary_path)
    previous_df = previous_df[previous_df["scenario"] == scenario].copy()
    if previous_df.empty:
        return

    archive_dir = output_root.joinpath("archive", scenario)
    archive_dir.mkdir(parents=True, exist_ok=True)
    previous_df.to_csv(archive_dir.joinpath("previous_selection_summary.csv"), index=False, encoding="utf-8")
    shutil.copy2(previous_plot_path, archive_dir.joinpath("previous_leadership_day_plot.png"))


def run_selection(
    results_root: Path = DEFAULT_RESULTS_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    tail_points: int = DEFAULT_TAIL_POINTS,
    fill_method: str = DEFAULT_FILL_METHOD,
    archive_previous: bool = True,
) -> pd.DataFrame:
    final_dir = output_root.joinpath("final")
    candidates_dir = output_root.joinpath("candidates")
    final_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for scenario in SCENARIOS:
        if archive_previous and scenario in ARCHIVE_SCENARIOS:
            _archive_previous_selection(output_root, scenario)

        candidate_df = build_candidate_pool(results_root, scenario, candidate_top_k, tail_points)
        scenario_candidates_dir = candidates_dir.joinpath(scenario)
        scenario_candidates_dir.mkdir(parents=True, exist_ok=True)
        candidate_df.to_csv(
            scenario_candidates_dir.joinpath("candidate_details.csv"),
            index=False,
            encoding="utf-8",
        )

        for _, row in candidate_df.iterrows():
            day_curve_df = load_day_curve(
                results_root,
                scenario,
                row["selected_model"],
                row["selected_date"],
            )
            candidate_plot_path = scenario_candidates_dir.joinpath(_candidate_plot_name(row))
            _plot_day_curve(candidate_plot_path, row.to_dict(), day_curve_df, fill_method)

        top_metric = candidate_df.sort_values(["MAPE", "MAE", "RMSE", "time_range"]).iloc[0]
        selected_row = sorted(
            [row for _, row in candidate_df.iterrows()],
            key=lambda row: _selection_tuple(scenario, row),
        )[0]
        selection = selected_row.to_dict()
        selection["selection_reason"] = _selection_reason(scenario, selected_row, top_metric)
        selection["note"] = _selection_note(scenario, selected_row)
        selection = {column: selection[column] for column in FINAL_SUMMARY_COLUMNS}

        selected_day_curve = load_day_curve(
            results_root,
            scenario,
            selection["selected_model"],
            selection["selected_date"],
        )
        _plot_day_curve(
            final_dir.joinpath(scenario, "leadership_day_plot.png"),
            selection,
            selected_day_curve,
            fill_method,
        )
        _copy_original_plot(
            results_root,
            scenario,
            selection["selected_model"],
            scenario_candidates_dir.joinpath("original_test_prediction.png"),
        )
        summary_rows.append(selection)

    summary_df = pd.DataFrame(summary_rows, columns=FINAL_SUMMARY_COLUMNS)
    summary_df.to_csv(
        final_dir.joinpath("leadership_day_summary.csv"),
        index=False,
        encoding="utf-8",
    )
    return summary_df


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select and redraw AIDC leadership comparison days.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--selection-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--candidate-top-k", type=int, default=DEFAULT_CANDIDATE_TOP_K)
    parser.add_argument("--tail-points", type=int, default=DEFAULT_TAIL_POINTS)
    parser.add_argument("--fill-method", default=DEFAULT_FILL_METHOD)
    parser.add_argument("--build-report", action="store_true")
    parser.add_argument(
        "--archive-previous",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.build_report:
        build_report_package(
            selection_root=args.selection_root,
            report_root=args.report_root,
            results_root=args.results_root,
            fill_method=args.fill_method,
        )
        return
    run_selection(
        results_root=args.results_root,
        output_root=args.output_root,
        candidate_top_k=args.candidate_top_k,
        tail_points=args.tail_points,
        fill_method=args.fill_method,
        archive_previous=args.archive_previous,
    )


if __name__ == "__main__":
    main()
