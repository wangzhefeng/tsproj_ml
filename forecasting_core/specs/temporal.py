"""预测目标区间和原始训练历史边界；回测与生产共用，不读取数据。"""
from collections.abc import Mapping
from typing import Any

import pandas as pd
from pandas.tseries.frequencies import to_offset


def select_training_origins(
    origins: tuple[pd.Timestamp, ...],
    indices: tuple[int, ...],
    sampling: Mapping[str, Any] | None,
    *,
    freq: str | None = None,
) -> tuple[int, ...]:
    """在原训练窗口内选择监督原点；不改变历史可见性或回测时间几何。"""
    if sampling is None:
        return indices
    selected = indices
    clock = sampling.get("time_of_day")
    if clock is not None:
        selected = tuple(i for i in selected if origins[i].strftime("%H:%M:%S.%f") == clock + ":00.000000")
    stride = sampling.get("stride_steps", 1)
    if "anchor_time" in sampling:
        if freq is None:
            raise ValueError("anchor_time sampling requires the original data frequency")
        period = stride * pd.tseries.frequencies.to_offset(freq).nanos
        anchor = pd.Timestamp(sampling["anchor_time"])
        selected = tuple(i for i in selected if (origins[i] - anchor).value % period == 0)
    else:
        selected = selected[::-stride][::-1]
    maximum = sampling.get("max_origins")
    if maximum is not None:
        selected = selected[-maximum:]
    if len(selected) < 2:
        raise ValueError("origin_sampling must retain at least two supervised origins")
    return selected


def validate_temporal_contract(problem, validation: Mapping) -> None:
    sampling = validation.get("training", {}).get("origin_sampling") or {}
    if "anchor_time" in sampling:
        if "stride_steps" not in sampling or "time_of_day" in sampling:
            raise ValueError("origin_sampling.anchor_time requires stride_steps and forbids time_of_day")
        if not isinstance(sampling["anchor_time"], str) or pd.isna(pd.Timestamp(sampling["anchor_time"])):
            raise ValueError("origin_sampling.anchor_time must be a timestamp string")
        if problem.freq not in {"5min", "15min", "1h", "1D"}:
            raise ValueError("anchor_time sampling requires a fixed frequency")
    for field in ("forecast_window", "training_window"):
        if field in validation and not isinstance(validation[field], Mapping):
            raise ValueError(f"{field} must be a mapping")
    forecast = validation.get("forecast_window")
    training = validation.get("training_window")
    if forecast is not None:
        start = forecast.get("start")
        if start not in {"after_origin", "next_day"}:
            raise ValueError("forecast_window.start must be after_origin or next_day")
        gap = forecast.get("gap_steps", 0)
        if isinstance(gap, bool) or not isinstance(gap, int) or gap < 0:
            raise ValueError("forecast_window.gap_steps must be a non-negative integer")
        if start == "next_day":
            if "gap_steps" in forecast:
                raise ValueError("next_day forbids gap_steps")
            if problem.freq not in {"5min", "15min", "1h", "1D"}:
                raise ValueError("next_day requires a fixed daily/intraday frequency")
            if (problem.horizon * to_offset(problem.freq).nanos) % pd.Timedelta(days=1).value:
                raise ValueError("next_day horizon must cover complete days")
    if training is not None:
        kind = training.get("kind")
        if kind == "rolling":
            length = training.get("history_steps")
            if isinstance(length, bool) or not isinstance(length, int) or length < 2:
                raise ValueError("training_window.history_steps must be an integer >= 2")
            if "start_time" in training:
                raise ValueError("rolling training_window forbids start_time")
        elif kind == "expanding":
            if not isinstance(training.get("start_time"), str) or pd.isna(pd.Timestamp(training["start_time"])):
                raise ValueError("expanding training_window requires start_time")
            if "history_steps" in training:
                raise ValueError("expanding training_window forbids history_steps")
        else:
            raise ValueError("training_window.kind must be rolling or expanding")
    if forecast is not None or training is not None:
        if problem.freq not in {"5min", "15min", "1h", "1D"}:
            raise ValueError("explicit temporal windows require fixed daily/intraday frequency")
        if validation.get("horizon_mode", "fixed_steps") != "fixed_steps":
            raise ValueError("forecast_window/training_window require fixed_steps")
        if validation.get("refit_every", 1) != 1:
            raise ValueError("explicit temporal windows require refit_every=1")


def forecast_times(problem, validation: Mapping, origin: pd.Timestamp) -> pd.DatetimeIndex:
    """origin为最后已知点，H为输出点数；不挪动信息截止时刻。"""
    origin = pd.Timestamp(origin)
    spec = validation.get("forecast_window", {})
    offset = to_offset(problem.freq)
    if spec.get("start") == "next_day":
        start = origin.normalize() + pd.DateOffset(days=1)
    else:
        start = origin + (1 + spec.get("gap_steps", 0)) * offset
    times = pd.date_range(start, periods=problem.horizon, freq=offset)
    if spec.get("start") == "next_day":
        days = problem.horizon * offset.nanos // pd.Timedelta(days=1).value
        if times[-1] + offset != start + pd.DateOffset(days=days):
            raise ValueError("next_day fixed-step horizon cannot cross a variable-length DST day")
    return times


def history_start(validation: Mapping, origin: pd.Timestamp, offset) -> pd.Timestamp | None:
    spec = validation.get("training_window")
    if spec is not None:
        start = (pd.Timestamp(spec["start_time"]) if spec["kind"] == "expanding"
                 else origin - (spec["history_steps"] - 1) * offset)
        if start > origin:
            raise ValueError("training_window starts after forecast origin")
        return start
    length = validation.get("train_history_steps")
    return None if length is None else origin - (length - 1) * offset


def has_bounded_history(validation: Mapping) -> bool:
    return validation.get("training_window") is not None or validation.get("train_history_steps") is not None


def forecast_ends(problem, validation: Mapping, origins: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """向量化标签截止时间，不为每个候选原点分配H个时间戳。"""
    spec = validation.get("forecast_window", {})
    offset = to_offset(problem.freq)
    if spec.get("start") == "next_day":
        return origins.normalize() + pd.DateOffset(days=1) + (problem.horizon - 1) * offset
    return origins + (problem.horizon + spec.get("gap_steps", 0)) * offset
