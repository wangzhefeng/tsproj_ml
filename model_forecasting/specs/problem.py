"""Forecasting problem definition independent of models and strategies."""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Day, Hour, Minute, MonthBegin, MonthEnd


_INFORMATION_MODES = frozenset({"forecast", "nowcast", "oracle"})
_TRAINING_SCOPES = frozenset({"local", "global"})
Frequency: TypeAlias = Literal["5min", "15min", "1h", "1D", "1ME", "1MS"]
_CANONICAL_FIXED_FREQ_BY_NANOS: dict[int, Frequency] = {
    5 * 60 * 1_000_000_000: "5min",
    15 * 60 * 1_000_000_000: "15min",
    60 * 60 * 1_000_000_000: "1h",
    24 * 60 * 60 * 1_000_000_000: "1D",
}
_SUPPORTED_FIXED_OFFSET_TYPES = (Minute, Hour, Day)
_CANONICAL_MONTH_FREQ_BY_TYPE: dict[type, Frequency] = {
    MonthEnd: "1ME",
    MonthBegin: "1MS",
}
_SUPPORTED_DEPRECATED_FIXED_ALIASES = frozenset(
    {"5T", "15T", "60T", "1440T", "H", "1H", "24H"}
)


def _normalize_names(value: Any, field_name: str, *, allow_empty: bool) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_names = (value,)
    elif isinstance(value, Sequence):
        raw_names = tuple(value)
    else:
        raise TypeError(f"{field_name} must be a string or sequence of strings")

    names = []
    for name in raw_names:
        if not isinstance(name, str):
            raise TypeError(f"{field_name} entries must be strings")
        stripped = name.strip()
        if not stripped:
            raise ValueError(f"{field_name} entries must not be blank")
        if name != stripped:
            raise ValueError(f"{field_name} entries must not contain surrounding whitespace")
        names.append(name)

    if not names and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")
    if len(names) != len(set(names)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return tuple(names)


def _normalize_required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be blank")
    if value != stripped:
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    return value


def _normalize_frequency(value: Any) -> Frequency:
    freq = _normalize_required_string(value, "freq")
    if freq == "m":
        raise ValueError("unsupported or ambiguous frequency: 'm'")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            if freq in _SUPPORTED_DEPRECATED_FIXED_ALIASES and freq.endswith("H"):
                warnings.filterwarnings(
                    "ignore",
                    message=r"'H' is deprecated and will be removed in a future version, please use 'h' instead\.",
                    category=FutureWarning,
                )
            if freq in _SUPPORTED_DEPRECATED_FIXED_ALIASES and freq.endswith("T"):
                warnings.filterwarnings(
                    "ignore",
                    message=r"'T' is deprecated and will be removed in a future version, please use 'min' instead\.",
                    category=FutureWarning,
                )
            offset = to_offset(freq)
    except FutureWarning as exc:
        raise ValueError(f"unsupported frequency: {freq!r}") from exc
    except ValueError as exc:
        raise ValueError(f"invalid frequency: {freq!r}") from exc

    if offset.n <= 0:
        raise ValueError("freq must be positive")

    if isinstance(offset, _SUPPORTED_FIXED_OFFSET_TYPES):
        canonical = _CANONICAL_FIXED_FREQ_BY_NANOS.get(offset.nanos)
    elif isinstance(offset, (MonthEnd, MonthBegin)) and offset.n == 1:
        canonical = _CANONICAL_MONTH_FREQ_BY_TYPE[type(offset)]
    else:
        canonical = None
    if canonical is None:
        raise ValueError(f"unsupported frequency: {freq!r}")
    return canonical


@dataclass(frozen=True, slots=True, init=False)
class ForecastProblemSpec:
    time_col: str
    freq: Frequency
    horizon: int
    targets: tuple[str, ...]
    information_mode: Literal["forecast", "nowcast", "oracle"]
    training_scope: Literal["local", "global"]
    series_id_cols: tuple[str, ...] = ()

    def __init__(
        self,
        time_col: str,
        freq: str,
        horizon: int,
        targets: str | Sequence[str],
        information_mode: Literal["forecast", "nowcast", "oracle"],
        training_scope: Literal["local", "global"],
        series_id_cols: str | Sequence[str] = (),
    ) -> None:
        normalized_time_col = _normalize_required_string(time_col, "time_col")
        normalized_freq = _normalize_frequency(freq)
        normalized_targets = _normalize_names(targets, "targets", allow_empty=False)
        normalized_series_id_cols = _normalize_names(
            series_id_cols,
            "series_id_cols",
            allow_empty=True,
        )

        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise TypeError("horizon must be an integer")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if information_mode not in _INFORMATION_MODES:
            raise ValueError(f"unknown information_mode: {information_mode!r}")
        if training_scope not in _TRAINING_SCOPES:
            raise ValueError(f"unknown training_scope: {training_scope!r}")
        if training_scope == "global" and not normalized_series_id_cols:
            raise ValueError("global training_scope requires at least one series_id_col")

        all_names = (normalized_time_col,) + normalized_targets + normalized_series_id_cols
        if len(all_names) != len(set(all_names)):
            raise ValueError("time_col, targets, and series_id_cols must not overlap")

        object.__setattr__(self, "time_col", normalized_time_col)
        object.__setattr__(self, "freq", normalized_freq)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "targets", normalized_targets)
        object.__setattr__(self, "information_mode", information_mode)
        object.__setattr__(self, "training_scope", training_scope)
        object.__setattr__(self, "series_id_cols", normalized_series_id_cols)

    @property
    def n_targets(self) -> int:
        return len(self.targets)

    @property
    def is_global(self) -> bool:
        return self.training_scope == "global"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "time_col": self.time_col,
            "freq": self.freq,
            "horizon": self.horizon,
            "targets": list(self.targets),
            "series_id_cols": list(self.series_id_cols),
            "information_mode": self.information_mode,
            "training_scope": self.training_scope,
        }
