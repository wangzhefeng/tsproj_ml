# -*- coding: utf-8 -*-
"""从滑窗宽表生成概率预测长表、窗口指标和 horizon 指标（legacy 宽表工具，仅测试消费）。

生产评估通路（evaluate_marginal_distribution / 指标内核）已迁入 `model_evaluation/`
（2026-08-30 evaluation 模块化）。"""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, cast

import numpy as np
import pandas as pd

from model_forecasting.tensors import PointForecastTensor
from model_evaluation.metrics import crossing_metrics, interval_metrics, pinball_loss
from probabilistic.calibration import (
    CalibrationRecord,
    CalibrationResult,
    calibrate_with_records,
)
from probabilistic.postprocessing import _parse_quantile_columns
from probabilistic.spec import IntervalSpec
from probabilistic.types import MarginalForecastDistribution


def _resolve_interval_columns(
    quantile_columns,
    lower_quantile=None,
    upper_quantile=None,
):
    """按数值 level 选择基础区间；未指定时兼容首尾 quantile。"""
    if len(quantile_columns) < 2:
        raise ValueError("interval evaluation requires at least two quantile columns")
    if lower_quantile is None and upper_quantile is None:
        return quantile_columns[0], quantile_columns[-1]
    if lower_quantile is None or upper_quantile is None:
        raise ValueError("lower_quantile and upper_quantile must be provided together")

    def find(level):
        matches = [
            item
            for item in quantile_columns
            if np.isclose(item[0], float(level), rtol=0.0, atol=1e-12)
        ]
        if len(matches) != 1:
            raise ValueError(f"quantile level={float(level):g} has no unique CSV column")
        return matches[0]

    lower = find(lower_quantile)
    upper = find(upper_quantile)
    if lower[0] >= upper[0]:
        raise ValueError("lower_quantile must be < upper_quantile")
    return lower, upper


def _score_row(
    *,
    window,
    forecast_origin,
    record_kind: str,
    stage: str,
    metric: str,
    value: float,
    n_points: int,
    n_windows: int,
    quantile=None,
    interval_name=None,
    lower_quantile=None,
    upper_quantile=None,
    target_coverage=None,
    ci_lower=None,
    ci_upper=None,
) -> Dict:
    return {
        "window": window,
        "forecast_origin": forecast_origin,
        "scope": "all",
        "record_kind": record_kind,
        "stage": stage,
        "metric": metric,
        "quantile": quantile,
        "interval_name": interval_name,
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "target_coverage": target_coverage,
        "value": float(value),
        "n_points": int(n_points),
        "n_windows": int(n_windows),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def _append_interval_score_rows(
    rows: List[Dict],
    *,
    metrics: Dict[str, float],
    window,
    forecast_origin,
    stage: str,
    interval_name: str,
    lower_quantile: float,
    upper_quantile: float,
    n_windows: int,
) -> None:
    target_coverage = upper_quantile - lower_quantile
    for metric_name in (
        "coverage",
        "width",
        "normalized_width",
        "winkler",
        "coverage_gap",
        "calibration_error",
    ):
        rows.append(
            _score_row(
                window=window,
                forecast_origin=forecast_origin,
                record_kind="interval",
                stage=stage,
                metric=metric_name,
                value=metrics[metric_name],
                n_points=int(metrics["n_points"]),
                n_windows=n_windows,
                interval_name=interval_name,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
                target_coverage=target_coverage,
                ci_lower=(metrics["coverage_ci_lower"] if metric_name == "coverage" else None),
                ci_upper=(metrics["coverage_ci_upper"] if metric_name == "coverage" else None),
            )
        )


def build_probabilistic_artifacts(
    cv_plot_df: pd.DataFrame,
    interval_specs: Sequence[IntervalSpec] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """由含 processed/raw quantile 的 CV 宽表生成三张概率 artifact。"""
    required = {"time", "window", "Y_trues", "Y_preds"}
    missing = required - set(cv_plot_df.columns)
    if missing:
        raise ValueError(f"probabilistic artifacts missing columns: {sorted(missing)}")
    quantile_columns = _parse_quantile_columns(cv_plot_df)
    if not quantile_columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    prediction_rows: List[Dict] = []
    score_rows: List[Dict] = []
    prepared_windows: List[pd.DataFrame] = []
    levels = tuple(level for level, _ in quantile_columns)
    if interval_specs is None:
        interval_specs = (
            IntervalSpec(f"q{levels[0]:g}_q{levels[-1]:g}", levels[0], levels[-1]),
        )
    interval_columns = []
    for interval in interval_specs:
        lower, upper = _resolve_interval_columns(
            quantile_columns,
            interval.lower_quantile,
            interval.upper_quantile,
        )
        interval_columns.append((interval, lower, upper))

    for window, window_frame in cv_plot_df.groupby("window", sort=False):
        ordered = window_frame.sort_values("time").copy()
        ordered["time"] = pd.to_datetime(ordered["time"])
        ordered["horizon_step"] = np.arange(1, len(ordered) + 1, dtype=int)
        forecast_origin = ordered["time"].min()
        ordered["forecast_origin"] = forecast_origin
        prepared_windows.append(ordered)
        y_true = ordered["Y_trues"].to_numpy(dtype=float)

        raw_columns = []
        for _, column in quantile_columns:
            raw_column = f"{column}_raw"
            raw_columns.append(
                ordered[raw_column if raw_column in ordered.columns else column].to_numpy(
                    dtype=float
                )
            )
        raw_matrix = np.column_stack(raw_columns)
        processed_matrix = np.column_stack(
            [ordered[column].to_numpy(dtype=float) for _, column in quantile_columns]
        )
        ordered_rows = ordered.reset_index(drop=True)
        for quantile_index, (level, _) in enumerate(quantile_columns):
            for row_index in range(len(ordered_rows)):
                row = ordered_rows.iloc[row_index]
                prediction_rows.append(
                    {
                        "forecast_origin": forecast_origin,
                        "window": window,
                        "time": row["time"],
                        "horizon_step": int(row["horizon_step"]),
                        "quantile": level,
                        "prediction_raw": raw_matrix[row_index, quantile_index],
                        "prediction_processed": processed_matrix[row_index, quantile_index],
                        "y_true": float(row["Y_trues"]),
                        "point_prediction": float(row["Y_preds"]),
                    }
                )
            for stage, values in (("raw", raw_matrix), ("processed", processed_matrix)):
                loss = pinball_loss(y_true, values[:, quantile_index], level)
                score_rows.append(
                    _score_row(
                        window=window,
                        forecast_origin=forecast_origin,
                        record_kind="quantile",
                        stage=stage,
                        metric="pinball",
                        quantile=level,
                        value=float(np.mean(loss)),
                        n_points=len(loss),
                        n_windows=1,
                    )
                )

        crossing = crossing_metrics(raw_matrix, levels, processed_matrix)
        for metric_name, value in crossing.items():
            score_rows.append(
                _score_row(
                    window=window,
                    forecast_origin=forecast_origin,
                    record_kind="quantile",
                    stage="raw",
                    metric=metric_name,
                    value=value,
                    n_points=len(ordered),
                    n_windows=1,
                )
            )

        # raw quantile 允许 crossing，只做 crossing/pinball 诊断；区间指标只评价
        # q50 锚定修复后的 processed boundaries。
        for interval, (lower_level, lower_column), (upper_level, upper_column) in interval_columns:
            lower_values = ordered[lower_column].to_numpy(dtype=float)
            upper_values = ordered[upper_column].to_numpy(dtype=float)
            if np.all(lower_values <= upper_values):
                metrics = interval_metrics(
                    y_true,
                    lower_values,
                    upper_values,
                    target_coverage=interval.nominal_coverage,
                )
                _append_interval_score_rows(
                    score_rows,
                    metrics=metrics,
                    window=window,
                    forecast_origin=forecast_origin,
                    stage="processed",
                    interval_name=interval.name,
                    lower_quantile=lower_level,
                    upper_quantile=upper_level,
                    n_windows=1,
                )

    prediction_df = pd.DataFrame(prediction_rows)
    score_df = pd.DataFrame(score_rows)
    prepared = pd.concat(prepared_windows, ignore_index=True)
    has_naive = "Y_naive" in prepared.columns
    horizon_rows: List[Dict] = []
    for horizon_step, horizon_frame in prepared.groupby("horizon_step", sort=True):
        horizon_number = int(np.asarray(horizon_step).item())
        y_true = horizon_frame["Y_trues"].to_numpy(dtype=float)
        y_pred = horizon_frame["Y_preds"].to_numpy(dtype=float)
        n_windows = int(horizon_frame["window"].nunique())
        forecast_origin = pd.NaT
        finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        n_valid = int(finite_mask.sum())
        # point 指标：MAE/MAPE 按 horizon 聚合；naive 对照存在时同口径输出
        if n_valid > 0:
            abs_error = np.abs(y_true - y_pred)
            for metric_name, value in (
                ("mae", float(np.mean(abs_error[finite_mask]))),
                (
                    "mape",
                    float(
                        np.mean(
                            abs_error[finite_mask]
                            / np.abs(y_true[finite_mask]).clip(min=1e-12)
                        )
                    ),
                ),
            ):
                row = _score_row(
                    window=None,
                    forecast_origin=forecast_origin,
                    record_kind="point",
                    stage="processed",
                    metric=metric_name,
                    value=value,
                    n_points=n_valid,
                    n_windows=n_windows,
                )
                row["horizon_step"] = horizon_number
                horizon_rows.append(row)
            if has_naive:
                y_naive = horizon_frame["Y_naive"].to_numpy(dtype=float)
                naive_finite = finite_mask & np.isfinite(y_naive)
                n_naive = int(naive_finite.sum())
                if n_naive > 0:
                    model_mae = float(np.mean(abs_error[naive_finite]))
                    naive_mae = float(
                        np.mean(np.abs(y_true[naive_finite] - y_naive[naive_finite]))
                    )
                    for metric_name, value in (
                        ("naive_mae", naive_mae),
                        (
                            "naive_skill",
                            (
                                1.0 - model_mae / naive_mae
                                if naive_mae > 0
                                else float("nan")
                            ),
                        ),
                    ):
                        row = _score_row(
                            window=None,
                            forecast_origin=forecast_origin,
                            record_kind="point",
                            stage="processed",
                            metric=metric_name,
                            value=value,
                            n_points=n_naive,
                            n_windows=n_windows,
                        )
                        row["horizon_step"] = horizon_number
                        horizon_rows.append(row)
        # crossing 指标：按 horizon 对 processed quantile matrix 聚合
        processed_matrix = np.column_stack(
            [horizon_frame[column].to_numpy(dtype=float) for _, column in quantile_columns]
        )
        crossing = crossing_metrics(processed_matrix, levels, processed_matrix)
        for metric_name, value in crossing.items():
            row = _score_row(
                window=None,
                forecast_origin=forecast_origin,
                record_kind="quantile",
                stage="processed",
                metric=metric_name,
                value=value,
                n_points=len(horizon_frame),
                n_windows=n_windows,
            )
            row["horizon_step"] = horizon_number
            horizon_rows.append(row)
        for level, column in quantile_columns:
            loss = pinball_loss(y_true, horizon_frame[column].to_numpy(dtype=float), level)
            row = _score_row(
                window=None,
                forecast_origin=forecast_origin,
                record_kind="quantile",
                stage="processed",
                metric="pinball",
                quantile=level,
                value=float(np.mean(loss)),
                n_points=len(loss),
                n_windows=n_windows,
            )
            row["horizon_step"] = horizon_number
            horizon_rows.append(row)
        for interval, (lower_level, lower_column), (upper_level, upper_column) in interval_columns:
            lower_values = horizon_frame[lower_column].to_numpy(dtype=float)
            upper_values = horizon_frame[upper_column].to_numpy(dtype=float)
            if np.all(lower_values <= upper_values):
                interval_result = interval_metrics(
                    y_true,
                    lower_values,
                    upper_values,
                    target_coverage=interval.nominal_coverage,
                )
                interval_rows: List[Dict] = []
                _append_interval_score_rows(
                    interval_rows,
                    metrics=interval_result,
                    window=None,
                    forecast_origin=forecast_origin,
                    stage="processed",
                    interval_name=interval.name,
                    lower_quantile=lower_level,
                    upper_quantile=upper_level,
                    n_windows=n_windows,
                )
                for row in interval_rows:
                    row["horizon_step"] = horizon_number
                horizon_rows.extend(interval_rows)

    return prediction_df, score_df, pd.DataFrame(horizon_rows)


def evaluate_prequential_cqr(
    cv_plot_df: pd.DataFrame,
    *,
    freq: str,
    calibration_windows: int,
    min_windows: int,
    min_scores: int,
    alpha: float,
    label_availability_delay_steps: int = 0,
    interval_name: str | None = None,
    lower_quantile: float | None = None,
    upper_quantile: float | None = None,
    allow_interval_shrink: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按标签可得时间对每个历史窗口执行因果 prequential CQR。"""
    required = {"time", "window", "Y_trues", "conformal_score"}
    missing = required - set(cv_plot_df.columns)
    if missing:
        raise ValueError(f"prequential CQR missing columns: {sorted(missing)}")
    quantile_columns = _parse_quantile_columns(cv_plot_df)
    if len(quantile_columns) < 2:
        raise ValueError("prequential CQR requires at least two quantile columns")
    delay_steps = int(label_availability_delay_steps)
    if delay_steps < 0:
        raise ValueError("label_availability_delay_steps must be >= 0")
    delay = pd.tseries.frequencies.to_offset(freq) * delay_steps
    (lower_level, lower_column), (upper_level, upper_column) = _resolve_interval_columns(
        quantile_columns,
        lower_quantile,
        upper_quantile,
    )
    interval_name = interval_name or (
        f"q{int(round(lower_level * 100))}_q{int(round(upper_level * 100))}"
    )

    records: List[CalibrationRecord] = []
    prepared_windows = []
    for window, window_frame in cv_plot_df.groupby("window", sort=False):
        window_number = int(np.asarray(window).item())
        ordered = window_frame.sort_values("time").copy()
        ordered["time"] = pd.to_datetime(ordered["time"])
        origin = cast(pd.Timestamp, pd.Timestamp(ordered["time"].iloc[0]))
        ordered["forecast_origin"] = origin
        ordered["horizon_step"] = np.arange(1, len(ordered) + 1, dtype=int)
        prepared_windows.append((origin, window_number, ordered))
        ordered_rows = ordered.reset_index(drop=True)
        for row_index in range(len(ordered_rows)):
            row = ordered_rows.iloc[row_index]
            target_time = pd.Timestamp(row["time"])
            records.append(
                CalibrationRecord(
                    forecast_origin=origin,
                    target_time=target_time,
                    label_available_at=pd.Timestamp(target_time + delay),
                    horizon_step=int(row["horizon_step"]),
                    window=window_number,
                    interval_name=interval_name,
                    lower=float(row[lower_column]),
                    upper=float(row[upper_column]),
                    y_true=float(row["Y_trues"]),
                    score=float(row["conformal_score"]),
                    stage="processed",
                )
            )

    interval_rows: List[Dict] = []
    report_rows: List[Dict] = []
    for origin, window, ordered in sorted(prepared_windows, key=lambda item: item[0]):
        result = calibrate_with_records(
            ordered[lower_column].to_numpy(dtype=float),
            ordered[upper_column].to_numpy(dtype=float),
            records,
            forecast_origin=origin,
            n_windows=calibration_windows,
            min_windows=min_windows,
            min_scores=min_scores,
            alpha=alpha,
            allow_interval_shrink=allow_interval_shrink,
        )
        selected_origins = [
            pd.Timestamp(record.forecast_origin) for record in result.selected_records
        ]
        selected_labels = [
            pd.Timestamp(record.label_available_at) for record in result.selected_records
        ]
        report_rows.append(
            {
                "scope": "prequential",
                "forecast_origin": origin,
                "window": window,
                "interval_name": interval_name,
                "method": "cqr",
                "target_coverage": 1.0 - float(alpha),
                "selected_origin_min": min(selected_origins) if selected_origins else pd.NaT,
                "selected_origin_max": max(selected_origins) if selected_origins else pd.NaT,
                "selected_label_available_max": max(selected_labels) if selected_labels else pd.NaT,
                "selected_windows": result.selected_windows,
                "n_windows": int(calibration_windows),
                "n_scores": result.selected_scores,
                "correction": result.correction,
                "allow_interval_shrink": allow_interval_shrink,
                "status": result.status,
                "reason": result.reason,
            }
        )
        if result.status != "applied" or result.lower is None or result.upper is None:
            continue
        ordered_rows = ordered.reset_index(drop=True)
        for row_index in range(len(ordered_rows)):
            row = ordered_rows.iloc[row_index]
            interval_rows.append(
                {
                    "forecast_origin": origin,
                    "window": window,
                    "time": row["time"],
                    "horizon_step": int(row["horizon_step"]),
                    "interval_name": interval_name,
                    "base_lower_quantile": lower_level,
                    "base_upper_quantile": upper_level,
                    "target_coverage": 1.0 - float(alpha),
                    "lower": float(result.lower[row_index]),
                    "upper": float(result.upper[row_index]),
                    "method": "cqr",
                    "y_true": float(row["Y_trues"]),
                }
            )
    return pd.DataFrame(interval_rows), pd.DataFrame(report_rows)


def calibrate_final_from_cv(
    cv_plot_df: pd.DataFrame,
    *,
    q_low: np.ndarray,
    q_high: np.ndarray,
    forecast_origin: pd.Timestamp,
    freq: str,
    calibration_windows: int,
    min_windows: int,
    min_scores: int,
    alpha: float,
    label_availability_delay_steps: int = 0,
    interval_name: str | None = None,
    lower_quantile: float | None = None,
    upper_quantile: float | None = None,
    allow_interval_shrink: bool = False,
):
    """使用与 prequential evaluator 相同的 as-of 选择器校准 final forecast。"""
    required = {"time", "window", "Y_trues", "conformal_score"}
    missing = required - set(cv_plot_df.columns)
    if missing:
        raise ValueError(f"final CQR missing columns: {sorted(missing)}")
    quantile_columns = _parse_quantile_columns(cv_plot_df)
    if len(quantile_columns) < 2:
        raise ValueError("final CQR requires at least two quantile columns")
    delay_steps = int(label_availability_delay_steps)
    if delay_steps < 0:
        raise ValueError("label_availability_delay_steps must be >= 0")
    delay = pd.tseries.frequencies.to_offset(freq) * delay_steps
    (lower_level, lower_column), (upper_level, upper_column) = _resolve_interval_columns(
        quantile_columns,
        lower_quantile,
        upper_quantile,
    )
    interval_name = interval_name or (
        f"q{int(round(lower_level * 100))}_q{int(round(upper_level * 100))}"
    )
    records: List[CalibrationRecord] = []
    for window, window_frame in cv_plot_df.groupby("window", sort=False):
        window_number = int(np.asarray(window).item())
        ordered = window_frame.sort_values("time").copy()
        ordered["time"] = pd.to_datetime(ordered["time"])
        origin = cast(pd.Timestamp, pd.Timestamp(ordered["time"].iloc[0]))
        for horizon_index, (_, row) in enumerate(ordered.iterrows(), start=1):
            target_time = pd.Timestamp(row["time"])
            records.append(
                CalibrationRecord(
                    forecast_origin=origin,
                    target_time=target_time,
                    label_available_at=pd.Timestamp(target_time + delay),
                    horizon_step=horizon_index,
                    window=window_number,
                    interval_name=interval_name,
                    lower=float(row[lower_column]),
                    upper=float(row[upper_column]),
                    y_true=float(row["Y_trues"]),
                    score=float(row["conformal_score"]),
                    stage="processed",
                )
            )
    return calibrate_with_records(
        q_low,
        q_high,
        records,
        forecast_origin=forecast_origin,
        n_windows=calibration_windows,
        min_windows=min_windows,
        min_scores=min_scores,
        alpha=alpha,
        allow_interval_shrink=allow_interval_shrink,
    )


def append_final_calibration_report(
    report_path: Path,
    *,
    result: CalibrationResult,
    forecast_origin: pd.Timestamp,
    interval_name: str,
    target_coverage: float,
    calibration_windows: int,
    allow_interval_shrink: bool = False,
) -> None:
    """把 final forecast 的 as-of 校准选择追加到共享审计表。"""
    selected_origins = [
        pd.Timestamp(record.forecast_origin) for record in result.selected_records
    ]
    selected_labels = [
        pd.Timestamp(record.label_available_at) for record in result.selected_records
    ]
    row = {
        "scope": "final",
        "forecast_origin": pd.Timestamp(forecast_origin),
        "window": None,
        "interval_name": str(interval_name),
        "method": "cqr",
        "target_coverage": float(target_coverage),
        "selected_origin_min": min(selected_origins) if selected_origins else pd.NaT,
        "selected_origin_max": max(selected_origins) if selected_origins else pd.NaT,
        "selected_label_available_max": max(selected_labels) if selected_labels else pd.NaT,
        "selected_windows": result.selected_windows,
        "n_windows": int(calibration_windows),
        "n_scores": result.selected_scores,
        "correction": result.correction,
        "allow_interval_shrink": allow_interval_shrink,
        "status": result.status,
        "reason": result.reason,
    }
    path = Path(report_path)
    existing_records = []
    if path.exists():
        existing_records = pd.read_csv(path).to_dict(orient="records")
    report = pd.DataFrame(existing_records + [row])
    for column in (
        "forecast_origin",
        "selected_origin_min",
        "selected_origin_max",
        "selected_label_available_max",
    ):
        if column in report.columns:
            report[column] = pd.to_datetime(
                report[column],
                errors="coerce",
                format="mixed",
            )
    report.to_csv(
        path,
        index=False,
        encoding="utf-8",
        date_format="%Y-%m-%dT%H:%M:%S",
    )


def write_probabilistic_artifacts(
    cv_plot_df: pd.DataFrame,
    output_dir: Path,
    *,
    enable_cqr: bool = False,
    freq: str = "1D",
    calibration_windows: int = 5,
    min_windows: int = 1,
    min_scores: int = 30,
    alpha: float = 0.1,
    label_availability_delay_steps: int = 0,
    interval_name: str | None = None,
    lower_quantile: float | None = None,
    upper_quantile: float | None = None,
    allow_interval_shrink: bool = False,
    interval_specs: Sequence[IntervalSpec] | None = None,
) -> pd.DataFrame:
    """写出 Phase 0 概率 artifacts，并返回兼容旧宽表的 DataFrame。"""
    predictions, scores, horizon_scores = build_probabilistic_artifacts(
        cv_plot_df,
        interval_specs=interval_specs,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if not predictions.empty:
        if enable_cqr:
            intervals, report = evaluate_prequential_cqr(
                cv_plot_df,
                freq=freq,
                calibration_windows=calibration_windows,
                min_windows=min_windows,
                min_scores=min_scores,
                alpha=alpha,
                label_availability_delay_steps=label_availability_delay_steps,
                interval_name=interval_name,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
                allow_interval_shrink=allow_interval_shrink,
            )
            report.to_csv(
                output_path.joinpath("calibration_report.csv"),
                index=False,
                encoding="utf-8",
            )
            intervals.to_csv(
                output_path.joinpath("probabilistic_intervals_df.csv"),
                index=False,
                encoding="utf-8",
            )
            if not intervals.empty:
                calibrated_score_rows: List[Dict] = []
                calibrated_horizon_rows: List[Dict] = []
                for window, group in intervals.groupby("window", sort=False):
                    target_coverage = float(group["target_coverage"].iloc[0])
                    result = interval_metrics(
                        group["y_true"].to_numpy(dtype=float),
                        group["lower"].to_numpy(dtype=float),
                        group["upper"].to_numpy(dtype=float),
                        target_coverage=target_coverage,
                    )
                    _append_interval_score_rows(
                        calibrated_score_rows,
                        metrics=result,
                        window=window,
                        forecast_origin=pd.Timestamp(group["forecast_origin"].iloc[0]),
                        stage="cqr",
                        interval_name=str(group["interval_name"].iloc[0]),
                        lower_quantile=float(group["base_lower_quantile"].iloc[0]),
                        upper_quantile=float(group["base_upper_quantile"].iloc[0]),
                        n_windows=1,
                    )
                for horizon_step, group in intervals.groupby("horizon_step", sort=True):
                    target_coverage = float(group["target_coverage"].iloc[0])
                    result = interval_metrics(
                        group["y_true"].to_numpy(dtype=float),
                        group["lower"].to_numpy(dtype=float),
                        group["upper"].to_numpy(dtype=float),
                        target_coverage=target_coverage,
                    )
                    rows: List[Dict] = []
                    _append_interval_score_rows(
                        rows,
                        metrics=result,
                        window=None,
                        forecast_origin=pd.NaT,
                        stage="cqr",
                        interval_name=str(group["interval_name"].iloc[0]),
                        lower_quantile=float(group["base_lower_quantile"].iloc[0]),
                        upper_quantile=float(group["base_upper_quantile"].iloc[0]),
                        n_windows=int(group["window"].nunique()),
                    )
                    for row in rows:
                        row["horizon_step"] = int(horizon_step)
                    calibrated_horizon_rows.extend(rows)
                scores = pd.DataFrame(
                    scores.to_dict(orient="records") + calibrated_score_rows
                )
                horizon_scores = pd.DataFrame(
                    horizon_scores.to_dict(orient="records") + calibrated_horizon_rows
                )
        predictions.to_csv(
            output_path.joinpath("probabilistic_predictions_df.csv"),
            index=False,
            encoding="utf-8",
        )
        scores.to_csv(
            output_path.joinpath("probabilistic_scores_df.csv"),
            index=False,
            encoding="utf-8",
        )
        horizon_scores.to_csv(
            output_path.joinpath("horizon_scores_df.csv"),
            index=False,
            encoding="utf-8",
        )
    raw_columns = [
        column
        for column in cv_plot_df.columns
        if str(column).startswith("predict_q") and str(column).endswith("_raw")
    ]
    return cv_plot_df.drop(columns=raw_columns)
