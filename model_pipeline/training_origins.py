"""在原训练窗口内选择监督原点；不改变历史可见性或回测时间几何。"""
from collections.abc import Mapping
from typing import Any

import pandas as pd


def select_training_origins(
    origins: tuple[pd.Timestamp, ...],
    indices: tuple[int, ...],
    sampling: Mapping[str, Any] | None,
    *,
    freq: str | None = None,
) -> tuple[int, ...]:
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
