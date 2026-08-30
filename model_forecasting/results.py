"""Canonical long-format forecast results and legacy read-only conversion."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model_forecasting.tensors import PointForecastTensor, require_matching_point_axes
from probabilistic.types import MarginalForecastDistribution, QuantileGrid


_CANONICAL_KEY_COLUMNS = ["series_id", "time", "target"]
_BACKTEST_KEY_COLUMNS = [*_CANONICAL_KEY_COLUMNS, "window"]


def point_tensor_to_long(tensor: PointForecastTensor) -> pd.DataFrame:
    if not isinstance(tensor, PointForecastTensor):
        raise TypeError("tensor must be a PointForecastTensor")
    rows = []
    for series_index, series_id in enumerate(tensor.series_ids):
        for time_index, timestamp in enumerate(tensor.forecast_times):
            for target_index, target in enumerate(tensor.targets):
                rows.append(
                    {
                        "series_id": series_id,
                        "time": pd.Timestamp(timestamp),
                        "target": target,
                        "predict_value": float(
                            tensor.values[series_index, time_index, target_index]
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.duplicated(_CANONICAL_KEY_COLUMNS).any():
        raise ValueError("canonical point result keys must be unique")
    return frame


def distribution_to_long(distribution: MarginalForecastDistribution) -> pd.DataFrame:
    if not isinstance(distribution, MarginalForecastDistribution):
        raise TypeError("distribution must be a MarginalForecastDistribution")
    frame = point_tensor_to_long(distribution.point)
    grid = QuantileGrid(
        distribution.quantiles.levels,
        point_level=distribution.quantiles.point_level,
    )
    for level_index, level in enumerate(distribution.quantiles.levels):
        frame[grid.column_name(level)] = distribution.quantiles.values[
            ..., level_index
        ].reshape(-1)
    return frame


def _point_forecast(value: PointForecastTensor | MarginalForecastDistribution) -> PointForecastTensor:
    if isinstance(value, PointForecastTensor):
        return value
    if isinstance(value, MarginalForecastDistribution):
        return value.point
    raise TypeError("forecast must be PointForecastTensor or MarginalForecastDistribution")


def backtest_tensors_to_long(
    actual: PointForecastTensor,
    prediction: PointForecastTensor | MarginalForecastDistribution,
    *,
    window: int,
    plot_valid: np.ndarray | None = None,
) -> pd.DataFrame:
    point = _point_forecast(prediction)
    require_matching_point_axes(actual, point)
    actual_frame = point_tensor_to_long(actual).rename(
        columns={"predict_value": "actual_value"}
    )
    prediction_frame = (
        point_tensor_to_long(prediction)
        if isinstance(prediction, PointForecastTensor)
        else distribution_to_long(prediction)
    )
    frame = actual_frame.merge(
        prediction_frame,
        on=_CANONICAL_KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    frame["window"] = int(window)
    if plot_valid is None:
        valid = np.isfinite(frame["actual_value"].to_numpy(dtype=float)) & np.isfinite(
            frame["predict_value"].to_numpy(dtype=float)
        )
    else:
        valid = np.asarray(plot_valid, dtype=bool).reshape(-1)
        if len(valid) != len(frame):
            raise ValueError("plot_valid must match canonical long row count")
    frame["plot_valid"] = valid
    if frame.duplicated(_BACKTEST_KEY_COLUMNS).any():
        raise ValueError("canonical backtest result keys must be unique")
    quantile_columns = [column for column in frame if column.startswith("predict_q")]
    return frame[
        [
            "series_id",
            "time",
            "target",
            "actual_value",
            "predict_value",
            *quantile_columns,
            "window",
            "plot_valid",
        ]
    ]


def write_forecast_results(
    output_dir: str | Path,
    forecast: PointForecastTensor | MarginalForecastDistribution,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frame = (
        point_tensor_to_long(forecast)
        if isinstance(forecast, PointForecastTensor)
        else distribution_to_long(forecast)
    )
    destination = output_path / "prediction.csv"
    frame.to_csv(destination, index=False, encoding="utf_8_sig")
    return destination


def _safe_plot_name(target: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(target)).strip("-.")
    return value or "target"


def _plot_backtest(frame: pd.DataFrame, output_path: Path, target: str) -> None:
    import matplotlib.pyplot as plt

    target_frame = frame[frame["target"] == target].copy()
    target_frame["time"] = pd.to_datetime(target_frame["time"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 5))
    for (window, series_id), group in target_frame.groupby(
        ["window", "series_id"], sort=True
    ):
        group = group.sort_values("time")
        valid = group["plot_valid"].astype(bool)
        axis.plot(
            group["time"],
            group["actual_value"].where(valid),
            label=f"actual w{window} {series_id}",
        )
        axis.plot(
            group["time"],
            group["predict_value"].where(valid),
            linestyle="--",
            label=f"predict w{window} {series_id}",
        )
    axis.set_title(str(target))
    axis.set_xlabel("time")
    axis.set_ylabel(str(target))
    axis.grid(True)
    axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def write_backtest_results(
    output_dir: str | Path,
    cv_plot_df: pd.DataFrame,
    test_scores_df: pd.DataFrame,
    *,
    aggregate_weighting: Mapping[str, float],
    metadata: Mapping[str, Any] | None = None,
    probabilistic_scores_df: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    required = {
        "series_id",
        "time",
        "target",
        "actual_value",
        "predict_value",
        "window",
        "plot_valid",
    }
    missing = required - set(cv_plot_df)
    if missing:
        raise ValueError(f"canonical backtest is missing columns: {sorted(missing)}")
    if cv_plot_df.duplicated(_BACKTEST_KEY_COLUMNS).any():
        raise ValueError("canonical backtest result keys must be unique")
    cv_path = output_path / "cv_plot_df.csv"
    scores_path = output_path / "test_scores_df.csv"
    cv_plot_df.to_csv(cv_path, index=False, encoding="utf_8_sig")
    test_scores_df.to_csv(scores_path, index=False, encoding="utf_8_sig")
    # 概率评估（2026-08-30 接线）：quantile 模式的 pinball/central 区间指标，
    # tidy long schema，仅 quantile 模式产出；点评估文件保持原 schema 不变。
    if probabilistic_scores_df is not None:
        probabilistic_scores_df.to_csv(
            output_path / "test_scores_probabilistic_df.csv",
            index=False,
            encoding="utf_8_sig",
        )
    payload = {
        "result_schema_version": 2,
        "aggregate_weighting": {
            str(target): float(weight)
            for target, weight in aggregate_weighting.items()
        },
        **dict(metadata or {}),
    }
    (output_path / "result_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    targets = tuple(dict.fromkeys(str(target) for target in cv_plot_df["target"]))
    if len(targets) == 1:
        _plot_backtest(cv_plot_df, output_path / "test_prediction.png", targets[0])
    else:
        for target in targets:
            _plot_backtest(
                cv_plot_df,
                output_path / "target_plots" / f"{_safe_plot_name(target)}.png",
                target,
            )
    return cv_path, scores_path


class CanonicalResultReader:
    """Read canonical long-schema results."""

    @staticmethod
    def read_prediction(
        path: str | Path,
        *,
        target: str | None = None,
        series_id: object = "__local__",
    ) -> pd.DataFrame:
        frame = pd.read_csv(path)
        if set(_CANONICAL_KEY_COLUMNS + ["predict_value"]).issubset(frame):
            result = frame.copy()
            result["time"] = pd.to_datetime(result["time"])
            return result
        raise ValueError(f"non-canonical prediction file: {path}")

    @staticmethod
    def read_backtest(
        path: str | Path,
        *,
        target: str | None = None,
        series_id: object = "__local__",
    ) -> pd.DataFrame:
        frame = pd.read_csv(path)
        required = {
            "series_id",
            "time",
            "target",
            "actual_value",
            "predict_value",
            "window",
            "plot_valid",
        }
        if required.issubset(frame):
            result = frame.copy()
            result["time"] = pd.to_datetime(result["time"])
            return result
        raise ValueError(f"non-canonical backtest file: {path}")

    @staticmethod
    def read_scores(path: str | Path) -> pd.DataFrame:
        return pd.read_csv(path)


__all__ = [
    "CanonicalResultReader",
    "backtest_tensors_to_long",
    "distribution_to_long",
    "point_tensor_to_long",
    "write_backtest_results",
    "write_forecast_results",
]
