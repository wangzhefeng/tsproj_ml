"""按序列、目标、预测步独立的绝对残差校准；无隐式跨组池化。"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, cast

import numpy as np
import pandas as pd

from forecasting_core.point_intervals import PointIntervalForecast, ResidualCalibrationSpec, validate_residual_state
from forecasting_core.tensors import PointForecastTensor, require_matching_point_axes


@dataclass(frozen=True, slots=True)
class ResidualRecord:
    origin: pd.Timestamp
    available_at: pd.Timestamp
    window: int
    error: float


class ResidualCalibrationTracker:
    def __init__(self, spec: ResidualCalibrationSpec, *, freq_offset):
        if not isinstance(spec, ResidualCalibrationSpec):
            raise TypeError("residual tracker requires ResidualCalibrationSpec")
        self.spec = spec
        self.offset = freq_offset
        self.records: dict[tuple[Any, str, int], list[ResidualRecord]] = {}
        self.origins: set[pd.Timestamp] = set()

    def collect(self, actual: PointForecastTensor, prediction: PointForecastTensor, *, forecast_origin, window: int) -> None:
        require_matching_point_axes(actual, prediction)
        origin = pd.Timestamp(forecast_origin)
        if pd.isna(origin) or origin in self.origins:
            raise ValueError("residual collection requires a finite, unique forecast origin")
        origin = cast(pd.Timestamp, origin)
        expected = pd.date_range(origin + self.offset, periods=prediction.shape[1], freq=self.offset)
        if not prediction.forecast_times.equals(expected):
            raise ValueError("residual horizons must match the origin frequency grid")
        additions = []
        for i, series_id in enumerate(prediction.series_ids):
            for h, target_time in enumerate(prediction.forecast_times):
                for k, target in enumerate(prediction.targets):
                    value = actual.values[i, h, k]
                    if not np.isfinite(value):
                        continue  # 缺失标签不进校准池；不改预测输入缺失合同。
                    error = abs(float(value) - float(prediction.values[i, h, k]))
                    if not math.isfinite(error):
                        raise ValueError("residual score must be finite")
                    additions.append(((series_id, target, h), ResidualRecord(
                        origin, pd.Timestamp(target_time) + self.spec.label_availability_delay_steps * self.offset,
                        int(window), error,
                    )))
        for key, record in additions:
            self.records.setdefault(key, []).append(record)
        self.origins.add(origin)

    def state(self, point: PointForecastTensor, *, forecast_origin) -> dict:
        origin = pd.Timestamp(forecast_origin)
        if pd.isna(origin):
            raise ValueError("calibration origin must be finite")
        radii, statuses, counts = [], [], []
        for series_id in point.series_ids:
            for h in range(point.shape[1]):
                for target in point.targets:
                    eligible = [record for record in self.records.get((series_id, target, h), ())
                                if record.origin < origin and record.available_at <= origin]
                    selected = sorted(eligible, key=lambda record: record.origin, reverse=True)[:self.spec.calibration_windows]
                    windows = len({(record.window, record.origin) for record in selected})
                    n = len(selected)
                    radius = None
                    if windows < self.spec.min_windows:
                        status = "insufficient_windows"
                    elif n < self.spec.min_scores:
                        status = "insufficient_scores"
                    else:
                        rank = math.ceil((n + 1) * self.spec.target_coverage)
                        if rank > n:
                            status = "insufficient_rank"
                        else:
                            radius = sorted(record.error for record in selected)[rank - 1]
                            status = "applied"
                    radii.append(radius)
                    statuses.append(status)
                    counts.append({"windows": windows, "scores": n})
        return {
            "method": "absolute_residual", "grouping": "series_target_horizon",
            "target_coverage": self.spec.target_coverage, "origin": origin.isoformat(), "freq": self.offset.freqstr,
            "series_ids": list(point.series_ids), "targets": list(point.targets), "shape": list(point.shape),
            "radii": radii, "statuses": statuses, "counts": counts,
            "status": "applied" if all(value is not None for value in radii) else "partial_or_insufficient",
        }

    def apply(self, point: PointForecastTensor, *, forecast_origin) -> PointIntervalForecast:
        return apply_residual_state(point, self.state(point, forecast_origin=forecast_origin))


def apply_residual_state(point: PointForecastTensor, state: dict) -> PointIntervalForecast:
    if not isinstance(state, dict) or "target_coverage" not in state:
        raise ValueError("invalid residual state coverage")
    validate_residual_state(state, series_ids=point.series_ids, targets=point.targets,
                            shape=point.shape, coverage=state["target_coverage"])
    offset = pd.tseries.frequencies.to_offset(state["freq"])
    if point.forecast_times[0] - offset < pd.Timestamp(state["origin"]):
        raise ValueError("cannot apply residual calibration to forecasts at/before its origin")
    delta = point.forecast_times[0] - pd.Timestamp(state["origin"])
    expected = pd.date_range(point.forecast_times[0], periods=point.shape[1], freq=offset)
    if delta.value % offset.nanos != 0 or not point.forecast_times.equals(expected):
        raise ValueError("residual forecast frequency grid does not match calibration state")
    return PointIntervalForecast(point, tuple(state["radii"]), state["target_coverage"], tuple(state["statuses"]))
