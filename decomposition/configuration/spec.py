"""分解配置的唯一解析/校验入口；preset 配方在 presets.py。"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any
from decomposition.configuration.presets import ComponentSpec, PresetParams, preset_components

RESERVED_METHODS = {
    "multiplicative", "log_additive", "boxcox_additive", "oof_component_feature",
    "modal_vmd", "modal_ceemdan", "modal_wavelet", "dlinear", "nbeats", "learned_basis",
    "probabilistic_component", "quantile_component",
}
TREND_FORECAST_MODES = frozenset({"polynomial", "damped"})
DECOMPOSITION_METHODS = frozenset({"none", "linear", "quadratic", "damped", "stl", "mstl"})
_DEFAULTS = {
    "composition": "additive", "periods": (), "robust": True, "trend_degree": 1,
    "trend_forecast": "polynomial", "damping": 0.98, "trend_lookback": 28, "seasonal_cycles": 4,
}


@dataclass(frozen=True)
class DecompositionSpec:
    method: str
    composition: str = "additive"
    extractor: ComponentSpec | None = None
    trend_forecaster: ComponentSpec | None = None
    seasonal_forecaster: ComponentSpec | None = None
    composer: ComponentSpec | None = None
    preset: PresetParams | None = None

    @property
    def enabled(self) -> bool:
        return self.method != "none"

    def expand_preset(self) -> DecompositionSpec:
        if self.method not in {"none", "linear", "stl", "mstl"}:
            raise ValueError(f"decomposition method {self.method!r} is not implemented")
        if self.composition != "additive":
            raise ValueError("decomposition supports only additive composition")
        if self.method == "none":
            return DecompositionSpec(method="none", preset=self.preset)
        if self.preset is None or self.preset.method != self.method:
            raise ValueError("decomposition preset must match its method")
        return DecompositionSpec(method=self.method, composition=self.composition,
                                 preset=self.preset, **preset_components(self.preset))


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"decomposition.{name} must be a positive integer")
    return value


def normalize_decomposition_config(value: Any) -> dict[str, Any]:
    """严格校验并归一化别名；不填入未声明默认值，不改变窗口选择策略。"""
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError("decomposition must be a mapping")
    raw = dict(value)
    method = str(raw.get("method", "none") or "none").lower()
    if method == "custom" or method in RESERVED_METHODS:
        raise ValueError(f"decomposition method {method!r} is reserved but not implemented")
    if method not in DECOMPOSITION_METHODS:
        raise ValueError(f"unsupported decomposition method {method!r}")
    unknown = set(raw) - set(_DEFAULTS) - {"method", "fit_history_steps"}
    if unknown:
        raise ValueError(f"Unknown decomposition config keys: {sorted(unknown)}")
    if method == "quadratic":
        if raw.get("trend_degree", 2) != 2:
            raise ValueError("quadratic decomposition requires trend_degree=2")
        raw.setdefault("trend_degree", 2)
        method = "linear"
    elif method == "damped":
        if str(raw.get("trend_forecast", "damped") or "damped").lower() != "damped":
            raise ValueError("damped decomposition requires trend_forecast='damped'")
        raw["trend_forecast"] = "damped"
        method = "linear"
    raw["method"] = method
    composition = str(raw.get("composition", "additive")).lower()
    if composition != "additive":
        raise ValueError("decomposition composition supports only 'additive'")
    if "composition" in raw:
        raw["composition"] = composition
    periods = raw.get("periods", ())
    if not isinstance(periods, Sequence) or isinstance(periods, (str, bytes)):
        raise TypeError("decomposition.periods must be a sequence of positive integers")
    for period in periods:
        if _positive_int(period, "periods") < 2:
            raise ValueError("decomposition periods must be >= 2")
    if len(set(periods)) != len(periods) or list(periods) != sorted(periods):
        raise ValueError("decomposition periods must be unique and increasing")
    if "periods" in raw:
        raw["periods"] = list(periods)
    if "robust" in raw and not isinstance(raw["robust"], bool):
        raise TypeError("decomposition.robust must be bool")
    for name in ("trend_degree", "trend_lookback", "seasonal_cycles"):
        if name in raw:
            _positive_int(raw[name], name)
    if raw.get("trend_degree", 1) not in {1, 2}:
        raise ValueError("decomposition trend_degree must be 1 or 2")
    mode = str(raw.get("trend_forecast", "polynomial")).lower()
    if mode not in TREND_FORECAST_MODES:
        raise ValueError(f"decomposition trend_forecast must be one of {sorted(TREND_FORECAST_MODES)}")
    if "trend_forecast" in raw:
        raw["trend_forecast"] = mode
    damping = raw.get("damping", 0.98)
    if isinstance(damping, bool) or not isinstance(damping, (int, float)) or not isfinite(damping) or not 0 < damping <= 1:
        raise ValueError("decomposition damping must be finite and in (0, 1]")
    if method == "none":
        disallowed = [key for key in _DEFAULTS if key in raw and raw[key] != _DEFAULTS[key]]
        if disallowed:
            raise ValueError(f"decomposition method 'none' should not set trend/seasonal fields: {disallowed}")
    if method == "linear" and {"periods", "robust", "seasonal_cycles"} & set(raw):
        raise ValueError("decomposition method 'linear' should not set seasonal fields")
    if method == "stl" and len(periods) != 1:
        raise ValueError("decomposition method 'stl' requires exactly 1 period")
    if method == "mstl" and len(periods) < 2:
        raise ValueError("decomposition method 'mstl' requires >= 2 periods")
    if "fit_history_steps" in raw:
        _positive_int(raw["fit_history_steps"], "fit_history_steps")
        if method == "none":
            raise ValueError("decomposition.fit_history_steps requires an enabled decomposition")
    return raw


def resolve_decomposition_spec(cfg: Any) -> DecompositionSpec:
    raw = normalize_decomposition_config(getattr(cfg, "decomposition", None))
    raw.pop("fit_history_steps", None)
    params = {**_DEFAULTS, **raw}
    params["periods"] = tuple(params["periods"])
    preset = PresetParams(**params)
    return DecompositionSpec(method=preset.method, composition=preset.composition, preset=preset)
