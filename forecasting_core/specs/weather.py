"""天气生成配方的不可变强类型合同；不读取资产，不执行气象计算。"""
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import re
from zoneinfo import ZoneInfo

import pandas as pd


WEATHER_UNITS = frozenset({"degC", "K", "delta_degC", "delta_K", "%", "m/s", "km/h", "Pa", "hPa", "mm", "W/m2", "J/m2", "MJ/m2", "degree", "delta_degree"})


def _text(value, field):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a nonempty trimmed string")


def _choice(value, choices, field):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"invalid weather {field}: {value!r}")


def _duration(value, field):
    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*(?:s|min|h|D)", value):
        raise ValueError(f"{field} must be a positive unit-bearing duration")
    return pd.Timedelta(value)


def _items(value, field, *, nonempty=True):
    if not isinstance(value, (list, tuple)) or (nonempty and not value):
        raise ValueError(f"{field} must be {'nonempty ' if nonempty else ''}list/tuple")
    return tuple(value)


def _typed(cls, value):
    if isinstance(value, cls):
        return value
    expected = {f.name for f in fields(cls)}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{cls.__name__} requires exactly {sorted(expected)}")
    return cls(**value)


@dataclass(frozen=True, slots=True)
class WeatherInputSpec:
    manifest: str
    sha256: str

    def __post_init__(self):
        _text(self.manifest, "manifest")
        if not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")


@dataclass(frozen=True, slots=True)
class WeatherLocationSpec:
    series_id: tuple[str | int, ...]
    location_id: str

    def __post_init__(self):
        values = _items(self.series_id, "series_id", nonempty=False)
        if any(type(v) not in (str, int) or (isinstance(v, str) and not v) for v in values):
            raise ValueError("weather series_id values must be nonempty strings or integers")
        object.__setattr__(self, "series_id", values)
        _text(self.location_id, "location_id")


@dataclass(frozen=True, slots=True)
class WeatherVariableSpec:
    name: str
    input: str
    unit: str
    aggregation: str

    def __post_init__(self):
        _text(self.name, "variable.name")
        _text(self.input, "variable.input")
        _choice(self.unit, WEATHER_UNITS, "unit")
        _choice(self.aggregation, {"point", "mean", "min", "max", "integral", "sum"}, "aggregation")


@dataclass(frozen=True, slots=True)
class WeatherNativeFeatureSpec:
    name: str
    inputs: tuple[str, ...]
    operation: str
    window: str | None

    def __post_init__(self):
        _text(self.name, "native_feature.name")
        inputs = _items(self.inputs, "native_feature.inputs")
        for value in inputs:
            _text(value, "native_feature.input")
        object.__setattr__(self, "inputs", inputs)
        _choice(self.operation, {"relative_humidity", "rolling_mean", "difference"}, "native operation")
        if self.operation == "relative_humidity":
            if len(inputs) != 2 or self.window is not None:
                raise ValueError("relative_humidity requires temperature/dewpoint and null window")
        elif len(inputs) != 1:
            raise ValueError("rolling/difference requires one input")
        else:
            _duration(self.window, "native_feature.window")


@dataclass(frozen=True, slots=True)
class WeatherTemporalSpec:
    timezone: str
    freq: str
    label: str
    closed: str
    upsampling: str
    max_age: str | None

    def __post_init__(self):
        _text(self.timezone, "timezone")
        try:
            ZoneInfo(self.timezone)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"unknown weather timezone: {self.timezone}") from exc
        _choice(self.freq, {"5min", "15min", "1h", "1D", "1ME", "1MS"}, "freq")
        _choice(self.label, {"left", "right"}, "label")
        _choice(self.closed, {"left", "right"}, "closed")
        _choice(self.upsampling, {"exact", "hold", "linear"}, "upsampling")
        if self.upsampling == "exact":
            if self.max_age is not None:
                raise ValueError("exact upsampling requires null max_age")
        else:
            _duration(self.max_age, "max_age")


@dataclass(frozen=True, slots=True)
class WeatherProxySpec:
    years: int
    leap_day: str
    data_kind: str

    def __post_init__(self):
        if type(self.years) is not int or self.years != 1:
            raise ValueError("prior_year_proxy requires years: 1")
        _choice(self.leap_day, {"reject", "feb28"}, "leap_day")
        _choice(self.data_kind, {"observation", "reanalysis"}, "proxy data_kind")


@dataclass(frozen=True, slots=True)
class WeatherResearchSpec:
    release_delay: str
    rationale: str

    def __post_init__(self):
        _duration(self.release_delay, 'research.release_delay')
        _text(self.rationale, 'research.rationale')


@dataclass(frozen=True, slots=True)
class WeatherGenerationSpec:
    inputs: tuple[WeatherInputSpec, ...]
    location_map: tuple[WeatherLocationSpec, ...]
    variables: tuple[WeatherVariableSpec, ...]
    native_features: tuple[WeatherNativeFeatureSpec, ...]
    temporal: WeatherTemporalSpec
    scenario: str
    vintage_policy: str | None
    proxy: WeatherProxySpec | None
    semantics_version: str
    research: WeatherResearchSpec | None = None

    def __post_init__(self):
        for name, cls in (("inputs", WeatherInputSpec), ("location_map", WeatherLocationSpec), ("variables", WeatherVariableSpec), ("native_features", WeatherNativeFeatureSpec)):
            values = _items(getattr(self, name), name, nonempty=name != "native_features")
            object.__setattr__(self, name, tuple(_typed(cls, v) for v in values))
        object.__setattr__(self, "temporal", _typed(WeatherTemporalSpec, self.temporal))
        if len({v.manifest for v in self.inputs}) != len(self.inputs):
            raise ValueError("duplicate weather manifest")
        if len({v.series_id for v in self.location_map}) != len(self.location_map):
            raise ValueError("duplicate weather series mapping")
        for name in ("variables", "native_features"):
            values = getattr(self, name)
            if len({v.name for v in values}) != len(values):
                raise ValueError(f"duplicate weather {name}")
        _choice(self.scenario, {"forecast", "prior_year_proxy"}, "scenario")
        _choice(self.semantics_version, {"weather_v1", "weather_research_v1"}, "semantics_version")
        if self.semantics_version == 'weather_research_v1':
            object.__setattr__(self, 'research', _typed(WeatherResearchSpec, self.research))
        elif self.research is not None:
            raise ValueError('weather_v1 forbids research assumptions')
        if self.scenario == "forecast":
            if self.vintage_policy != "latest_complete_snapshot" or self.proxy is not None:
                raise ValueError("forecast requires latest_complete_snapshot and null proxy")
        else:
            if self.vintage_policy is not None:
                raise ValueError("proxy forbids vintage_policy")
            object.__setattr__(self, "proxy", _typed(WeatherProxySpec, self.proxy))

    @classmethod
    def from_mapping(cls, value):
        if isinstance(value, Mapping) and 'research' not in value:
            value = {**value, 'research': None}
        return _typed(cls, value)

    def canonical_payload(self):
        # 显式转换 tuple，保证 YAML 往返与语义指纹只含普通 JSON 类型。
        def payload(value):
            if isinstance(value, dict):
                return {key: payload(v) for key, v in value.items()}
            if isinstance(value, tuple):
                return [payload(v) for v in value]
            return value
        result = {key: payload(value) for key, value in asdict(self).items()}
        if self.research is None:
            result.pop('research')
        return result
