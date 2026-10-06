"""完整有限观测序列的可保存统计状态；不处理缺失填补。

EWM 的运算顺序对应 pandas 2.3.3 window/aggregations.pyx 的 adjust=True
mean/covariance 实现；保留运算顺序以匹配现有前缀计算，而非换成近似 EMA。
https://github.com/pandas-dev/pandas/blob/v2.3.3/pandas/_libs/window/aggregations.pyx

BSD 3-Clause License (pandas-derived EWM recurrence)
Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team
All rights reserved.
Copyright (c) 2011-2023, Open source contributors.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
from itertools import product
import math
import hashlib
import json

import numpy as np
import pandas as pd

from feature_engineering.kernels.history import history_statistic
from data_loading.information.information_set import SourceLineage
from forecasting_core.specs.data import AvailabilityPolicy


def _fma(a: float, b: float, c: float) -> float:
    # Python 3.10 无 math.fma；精确有理数只在最终结果做一次 binary64 舍入。
    # 不使用近似 longdouble（部分平台与 double 同宽）。
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def _ewm_step(mean, covariance, weight, squared_weight, factor, value, mode):
    weight *= factor
    squared_weight *= factor * factor
    previous_mean = mean
    if mean != value:
        mean = (_fma(weight, mean, value) if mode[0] else weight * mean + value) / (weight + 1.0)
    old, new = previous_mean - mean, value - mean
    inner = _fma(old, old, covariance) if mode[1] else covariance + old * old
    if mode[2] == 1:
        numerator = _fma(weight, inner, new * new)
    elif mode[2] == 2:
        numerator = _fma(new, new, weight * inner)
    else:
        numerator = weight * inner + new * new
    return mean, numerator / (weight + 1.0), weight + 1.0, squared_weight + 1.0


@lru_cache(maxsize=1)
def _arithmetic_mode() -> tuple[bool, bool, int]:
    """显式探测 pandas 构建的乘加收缩方式；不兼容即 RAISE，不放宽精度。"""
    rng = np.random.default_rng(71)
    probes = []
    for values in (rng.normal(size=70), np.full(70, 3.0), 1e12 + rng.normal(size=70)):
        for halflife in (0.25, 2.0, 5.7):
            reference = pd.Series(values).ewm(halflife=halflife, adjust=True)
            decay = 1 - np.exp(np.log(0.5) / halflife)
            factor = 1.0 - 1.0 / (1.0 + float(1 / decay - 1))
            probes.append((values, factor, reference.mean().to_numpy(), reference.std().to_numpy()))
    for mode in product((False, True), (False, True), range(3)):
        matched = True
        for values, factor, means, stds in probes:
            state = (float(values[0]), 0.0, 1.0, 1.0)
            for index, value in enumerate(values[1:], 1):
                state = _ewm_step(*state, factor, float(value), mode)
                mean, covariance, weight, squared = state
                variance = (weight * weight / (weight * weight - squared)) * covariance
                if mean != means[index] or float(np.sqrt(max(variance, 0.0))) != stds[index]:
                    matched = False
                    break
            if not matched:
                break
        if matched:
            return mode
    raise RuntimeError("installed pandas EWM arithmetic is incompatible with exact streaming state")


@dataclass(slots=True)
class EwmState:
    halflife: float
    count: int = 0
    weighted_mean: float = 0.0
    covariance: float = 0.0
    weight: float = 1.0
    squared_weight: float = 1.0
    factor: float = field(init=False)
    arithmetic_mode: tuple[bool, bool, int] = field(init=False)

    def __post_init__(self):
        if isinstance(self.halflife, bool) or not math.isfinite(self.halflife) or self.halflife <= 0:
            raise ValueError("EWM halflife must be finite and positive")
        # 与 pandas 的 halflife -> com -> alpha 转换保持相同运算顺序。
        decay = 1 - np.exp(np.log(0.5) / self.halflife)
        com = float(1 / decay - 1)
        self.factor = 1.0 - 1.0 / (1.0 + com)
        self.arithmetic_mode = _arithmetic_mode()

    def update(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("EWM update requires finite observations")
        if self.count == 0:
            self.weighted_mean = value
            self.count = 1
            return
        mean, covariance, weight, squared_weight = _ewm_step(
            self.weighted_mean, self.covariance, self.weight, self.squared_weight,
            self.factor, value, self.arithmetic_mode,
        )
        if not math.isfinite(mean) or not math.isfinite(covariance):
            raise ValueError("EWM state overflow")
        self.weighted_mean, self.covariance = mean, covariance
        self.weight, self.squared_weight = weight, squared_weight
        self.count += 1

    def mean(self) -> float:
        if not self.count:
            raise ValueError("EWM mean requires observations")
        return self.weighted_mean

    def std(self) -> float:
        numerator = self.weight * self.weight
        denominator = numerator - self.squared_weight
        if self.count < 2 or denominator <= 0:
            raise ValueError("EWM std has insufficient visible history")
        variance = (numerator / denominator) * self.covariance
        if not math.isfinite(variance):
            raise ValueError("EWM variance overflow")
        return float(np.sqrt(max(variance, 0.0)))


@dataclass(slots=True)
class EventState:
    count: int = 0
    tail: tuple[float, ...] = ()
    last_peak: int = 0
    last_trough: int = 0

    def update(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("event update requires finite observations")
        if len(self.tail) == 2:
            left, middle = self.tail
            if left < middle and middle > value:
                self.last_peak = self.count - 1
            if left > middle and middle < value:
                self.last_trough = self.count - 1
        self.tail = (*self.tail, value)[-2:]
        self.count += 1

    def value(self, event: str) -> float:
        if event not in {"peak", "trough"} or not self.count:
            raise ValueError("time_since requires observations and peak/trough event")
        position = self.last_peak if event == "peak" else self.last_trough
        return float(self.count - 1 - position)


@dataclass(slots=True)
class ExpandingState:
    stats: tuple[str, ...]
    count: int = 0
    minimum: float = math.inf
    maximum: float = -math.inf
    last: float = 0.0
    min_diff: float = math.inf
    max_diff: float = -math.inf
    prefix: list[float] | None = field(init=False)

    def __post_init__(self):
        self.prefix = [] if set(self.stats) - {"min", "max", "min_diff", "max_diff"} else None

    def update(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError("expanding update requires finite observations")
        if self.count:
            difference = value - self.last
            if not math.isfinite(difference):
                raise ValueError("expanding difference overflow")
            self.min_diff, self.max_diff = min(self.min_diff, difference), max(self.max_diff, difference)
        self.minimum, self.maximum = min(self.minimum, value), max(self.maximum, value)
        self.last, self.count = value, self.count + 1
        if self.prefix is not None:
            self.prefix.append(value)

    def value(self, stat: str) -> float:
        if stat not in self.stats or not self.count:
            raise ValueError("unknown or uninitialized expanding statistic")
        if stat in {"min", "max"}:
            return self.minimum if stat == "min" else self.maximum
        if stat in {"min_diff", "max_diff"} and self.count > 1:
            return self.min_diff if stat == "min_diff" else self.max_diff
        return history_statistic(pd.Series(self.prefix if self.prefix is not None else [self.last], dtype=float), stat)


class StreamingStatistics:
    """Local/source_time 的只读统计提供器；更新返回新快照，失败不改旧状态。"""

    def __init__(self, advanced, *, config_fingerprint: str, time_col: str, freq: str):
        self.config_fingerprint, self.time_col, self.freq = config_fingerprint, time_col, freq
        self.ewm = {(column, float(halflife)): EwmState(float(halflife))
                    for column in advanced.get("ewm", {}).get("columns", ())
                    for halflife in advanced.get("ewm", {}).get("halflives", ())}
        self.events = {column: EventState() for column in advanced.get("time_since", {}).get("columns", ())}
        self.expanding = {column: ExpandingState(tuple(advanced["expanding"]["stats"]))
                          for column in advanced.get("expanding", {}).get("columns", ())}
        self.columns = tuple(sorted({key[0] for key in self.ewm} | set(self.events) | set(self.expanding)))
        self.count = 0
        self.origin = None
        self.history_start = None
        self.lineage_digest = hashlib.sha256(config_fingerprint.encode()).hexdigest()

    @property
    def state_policy(self) -> dict[str, str]:
        policy = {"finite_history": "bounded"}
        if self.ewm:
            policy["ewm"] = "bounded"
        if self.events:
            policy["time_since"] = "bounded"
        for column, state in self.expanding.items():
            policy[f"expanding:{column}"] = "growing_exact_prefix" if state.prefix is not None else "bounded"
        return policy

    def updated(self, frame: pd.DataFrame, *, origin) -> StreamingStatistics:
        origin = pd.Timestamp(origin)
        times = pd.DatetimeIndex(pd.to_datetime(frame[self.time_col]))
        offset = pd.tseries.frequencies.to_offset(self.freq)
        if (times.empty or times.hasnans or times[-1] != origin
                or not times.equals(pd.date_range(times[0], periods=len(times), freq=offset))
                or (self.origin is not None and times[0] != self.origin + offset)):
            raise ValueError("statistics update requires continuous forward source times")
        values = {column: frame[column].to_numpy(dtype=float) for column in self.columns}
        if any(not np.isfinite(array).all() for array in values.values()):
            raise ValueError("statistics update requires finite values")
        result = deepcopy(self)
        if result.history_start is None:
            result.history_start = times[0]
        for position, timestamp in enumerate(times):
            for (column, _), state in result.ewm.items():
                state.update(float(values[column][position]))
            for column, state in result.events.items():
                state.update(float(values[column][position]))
            for column, state in result.expanding.items():
                state.update(float(values[column][position]))
            payload = [timestamp.isoformat(), *(float(values[column][position]).hex() for column in self.columns)]
            result.lineage_digest = hashlib.sha256(bytes.fromhex(result.lineage_digest)
                + json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
        result.count += len(frame)
        result.origin = origin
        return result

    def require_binding(self, config_fingerprint: str, origin=None) -> None:
        if self.config_fingerprint != config_fingerprint or (origin is not None and self.origin != pd.Timestamp(origin)):
            raise ValueError("statistics snapshot identity/origin mismatch")
        if any(state.arithmetic_mode != _arithmetic_mode() for state in self.ewm.values()):
            raise ValueError("statistics snapshot pandas arithmetic changed; rebuild required")

    @property
    def source_lineage(self) -> SourceLineage:
        return SourceLineage("streaming_statistics", "online:" + self.lineage_digest, None,
                             AvailabilityPolicy.SOURCE_TIME, False)

    def validate_snapshot(self, expected: StreamingStatistics, origin) -> None:
        """恢复只验证保存状态，不由截断历史重造完整前缀。"""
        self.require_binding(expected.config_fingerprint, origin)
        if (self.time_col != expected.time_col or self.freq != expected.freq
                or self.ewm.keys() != expected.ewm.keys() or self.events.keys() != expected.events.keys()
                or self.expanding.keys() != expected.expanding.keys()
                or any(state.stats != expected.expanding[column].stats for column, state in self.expanding.items())):
            raise ValueError("statistics snapshot configuration mismatch")
        offset = pd.tseries.frequencies.to_offset(self.freq)
        if (type(self.count) is not int or self.count < 1 or self.history_start is None
                or self.history_start + (self.count - 1) * offset != self.origin
                or any(state.count != self.count for state in (*self.ewm.values(), *self.events.values(), *self.expanding.values()))
                or any(state.prefix is not None and len(state.prefix) != self.count for state in self.expanding.values())):
            raise ValueError("statistics snapshot count/time grid mismatch")
        if not isinstance(self.lineage_digest, str) or len(self.lineage_digest) != 64:
            raise ValueError("statistics snapshot lineage digest invalid")
        try:
            bytes.fromhex(self.lineage_digest)
        except ValueError as exc:
            raise ValueError("statistics snapshot lineage digest invalid") from exc
        for key, state in self.ewm.items():
            if (not all(math.isfinite(value) for value in (state.weighted_mean, state.covariance, state.weight, state.squared_weight))
                    or state.covariance < 0 or state.weight <= 0 or state.squared_weight <= 0
                    or state.squared_weight > state.weight * state.weight
                    or state.halflife != expected.ewm[key].halflife or state.factor != expected.ewm[key].factor):
                raise ValueError("statistics snapshot EWM numerical state invalid")
        for state in self.events.values():
            if (len(state.tail) != min(self.count, 2) or not all(math.isfinite(v) for v in state.tail)
                    or any(type(index) is not int or index < 0 or index >= max(1, self.count - 1)
                           for index in (state.last_peak, state.last_trough))):
                raise ValueError("statistics snapshot event state invalid")
        for state in self.expanding.values():
            if (not all(math.isfinite(v) for v in (state.minimum, state.maximum, state.last))
                    or not state.minimum <= state.last <= state.maximum
                    or (self.count > 1 and (not math.isfinite(state.min_diff) or not math.isfinite(state.max_diff)
                                           or state.min_diff > state.max_diff))):
                raise ValueError("statistics snapshot expanding state invalid")
            if state.prefix is not None:
                values = np.asarray(state.prefix, dtype=float)
                if (not np.isfinite(values).all() or values[-1] != state.last
                        or values.min() != state.minimum or values.max() != state.maximum
                        or (self.count > 1 and (np.diff(values).min() != state.min_diff or np.diff(values).max() != state.max_diff))):
                    raise ValueError("statistics snapshot expanding prefix mismatch")

    def value(self, kind: str, column: str, stat: str, *, origin, identity, parameter=None) -> float:
        self.require_binding(self.config_fingerprint, origin)
        if identity != ():
            raise ValueError("streaming statistics currently require Local identity")
        if kind == "ewm":
            if stat not in {"mean", "std"}:
                raise ValueError("unknown streaming EWM statistic")
            state = self.ewm[(column, float(parameter))]
            return state.mean() if stat == "mean" else state.std()
        if kind == "time_since":
            return self.events[column].value(stat)
        if kind == "expanding":
            return self.expanding[column].value(stat)
        raise ValueError("unknown streaming statistic kind")
