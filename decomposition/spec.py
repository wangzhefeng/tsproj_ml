# -*- coding: utf-8 -*-
"""时间序列分解的配置契约层。

唯一配置输入为 ``decomposition`` mapping（canonical YAML
``features.transformations.target.decomposition``），归一化为运行时对象
:class:`DecompositionSpec`。legacy ``preprocessing.decomposition_*`` 散列字段
已随 legacy 配置体系删除，不再解析。

preset（none/linear/stl/mstl）在 resolve 阶段展开为固定组件组合；
``custom`` 为未来组件级编排预留，当前一律 fail-fast。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

__all__ = [
    "ComponentSpec",
    "DecompositionSpec",
    "RESERVED_METHODS",
    "TREND_FORECAST_MODES",
    "PresetParams",
    "resolve_decomposition_spec",
]

# preset 模式下允许的键（不含 method 本身）
_PRESET_ALLOWED_KEYS = {
    "composition",
    "periods",
    "robust",
    "trend_degree",
    "trend_forecast",
    "damping",
    "trend_lookback",
    "seasonal_cycles",
}

# custom 模式允许的键
_CUSTOM_ALLOWED_KEYS = {
    "schema_version",
    "extractor",
    "extractor_params",
    "trend_forecaster",
    "trend_params",
    "seasonal_forecaster",
    "seasonal_params",
    "composer",
    "composer_params",
}

# 预留但未实现的组件类型（registry 层面同样拒绝）
RESERVED_METHODS = {
    "multiplicative",
    "log_additive",
    "boxcox_additive",
    "oof_component_feature",
    "modal_vmd",
    "modal_ceemdan",
    "modal_wavelet",
    "dlinear",
    "nbeats",
    "learned_basis",
    "probabilistic_component",
    "quantile_component",
}

_VALID_PRESET_METHODS = {"none", "linear", "stl", "mstl"}
TREND_FORECAST_MODES = frozenset({"polynomial", "damped"})

# PresetParams 的字段默认值
_PRESET_DEFAULTS = {
    "composition": "additive",
    "periods": (),
    "robust": True,
    "trend_degree": 1,
    "trend_forecast": "polynomial",
    "damping": 0.98,
    "trend_lookback": 28,
    "seasonal_cycles": 4,
}


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """深拷贝并冻结 mapping，运行期不得持有原始 YAML dict。"""
    if value is None:
        return MappingProxyType({})
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class ComponentSpec:
    """组件级配置（custom 模式预留；preset 展开后同样填充）。"""

    type: str
    params: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class PresetParams:
    """preset 模式归一化后的参数集合。"""

    method: str
    composition: str = "additive"
    periods: tuple[int, ...] = ()
    robust: bool = True
    trend_degree: int = 1
    trend_forecast: str = "polynomial"
    damping: float = 0.98
    trend_lookback: int = 28
    seasonal_cycles: int = 4


@dataclass(frozen=True)
class DecompositionSpec:
    """配置归一化后的唯一运行时输入。

    - preset 模式：``method`` 为 none/linear/stl/mstl，组件三元组由
      ``expand_preset()`` 填充；
    - custom 模式：组件三元组直接来自配置（当前未实现，仅保留类型）。
    """

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

    def expand_preset(self) -> "DecompositionSpec":
        """把 preset 方法展开为固定 extractor/forecaster/composer 组件组合。

        仅 preset 方法可调用；custom 已是展开形态。
        """
        if self.method == "custom":
            raise ValueError("DecompositionSpec.expand_preset() is only for preset methods.")
        if self.method == "none":
            return DecompositionSpec(method="none", preset=self.preset)
        if self.preset is None:  # pragma: no cover - 防御性兜底
            raise ValueError(f"DecompositionSpec method={self.method} has no preset params to expand.")
        p = self.preset
        if self.method == "linear":
            return DecompositionSpec(
                method="linear",
                composition=p.composition,
                extractor=ComponentSpec(
                    "poly_trend",
                    {
                        "degree": p.trend_degree,
                        # extractor 需要在 fit 阶段按 mode 决定近期斜率来源
                        # （damped 从观测 y 的 lookback 段估计，Phase 0 修复语义）
                        "mode": p.trend_forecast,
                        "lookback": p.trend_lookback,
                    },
                ),
                trend_forecaster=ComponentSpec(
                    "poly_trend_forecast",
                    {
                        "degree": p.trend_degree,
                        "mode": p.trend_forecast,
                        "damping": p.damping,
                        "lookback": p.trend_lookback,
                    },
                ),
                seasonal_forecaster=None,
                composer=ComponentSpec(p.composition, {}),
                preset=p,
            )
        # stl / mstl：extractor 由 periods/robust 决定
        ext_type = "stl" if self.method == "stl" else "mstl"
        return DecompositionSpec(
            method=self.method,
            composition=p.composition,
            extractor=ComponentSpec(ext_type, {"periods": tuple(p.periods), "robust": p.robust}),
            trend_forecaster=ComponentSpec(
                "poly_trend_forecast",
                {
                    "degree": p.trend_degree,
                    "mode": p.trend_forecast,
                    "damping": p.damping,
                    "lookback": p.trend_lookback,
                },
            ),
            seasonal_forecaster=ComponentSpec(
                "phase_template",
                {"cycles": p.seasonal_cycles, "periods": tuple(p.periods)},
            ),
            composer=ComponentSpec(p.composition, {}),
            preset=p,
        )


def _coerce_preset(raw: Mapping[str, Any], source: str) -> PresetParams:
    """把新写法 mapping 归一化为 PresetParams（含未知键/合法性校验）。"""
    unknown = set(raw) - ({"method"} | _PRESET_ALLOWED_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown decomposition config keys in {source}: {sorted(unknown)}; "
            f"allowed: {sorted({'method'} | _PRESET_ALLOWED_KEYS)}"
        )
    method = str(raw.get("method", "none")).lower()
    if method in RESERVED_METHODS:
        raise ValueError(
            f"decomposition method '{method}' is reserved but not implemented; "
            f"supported presets: {sorted(_VALID_PRESET_METHODS)}"
        )
    if method not in _VALID_PRESET_METHODS:
        raise ValueError(
            f"Unsupported decomposition method '{method}' in {source}; "
            f"supported: {sorted(_VALID_PRESET_METHODS | {'custom'})}"
        )
    composition = str(raw.get("composition", "additive")).lower()
    if composition != "additive":
        raise ValueError(
            f"decomposition composition '{composition}' is not supported; only 'additive'."
        )
    periods_raw = raw.get("periods", ())
    periods = tuple(int(p) for p in periods_raw or ())
    if method == "none":
        disallowed = {"periods", "robust", "trend_degree", "trend_forecast",
                      "damping", "trend_lookback", "seasonal_cycles", "composition"} & set(raw)
        if disallowed and any(raw.get(k, _PRESET_DEFAULTS.get(k)) != _PRESET_DEFAULTS.get(k) for k in disallowed):
            raise ValueError(
                f"decomposition method 'none' should not set trend/seasonal fields: "
                f"{sorted(disallowed)} in {source}"
            )
    if method == "linear":
        disallowed = {"periods", "robust", "seasonal_cycles"} & set(raw)
        if disallowed:
            raise ValueError(
                f"decomposition method 'linear' should not set seasonal fields: "
                f"{sorted(disallowed)} in {source}"
            )
    if method == "stl" and len(periods) != 1:
        raise ValueError(f"decomposition method 'stl' requires exactly 1 period; got {periods}.")
    if method == "mstl" and len(periods) < 2:
        raise ValueError(f"decomposition method 'mstl' requires >= 2 periods; got {periods}.")
    if method in {"stl", "mstl"} and not periods:
        raise ValueError(f"decomposition method '{method}' requires 'periods'.")
    trend_forecast = str(raw.get("trend_forecast", "polynomial")).lower()
    if trend_forecast not in TREND_FORECAST_MODES:
        raise ValueError(
            "decomposition trend_forecast must be one of "
            f"{sorted(TREND_FORECAST_MODES)}; got {trend_forecast!r}."
        )
    return PresetParams(
        method=method,
        composition=composition,
        periods=periods,
        robust=bool(raw.get("robust", True)),
        trend_degree=int(raw.get("trend_degree", 1)),
        trend_forecast=trend_forecast,
        damping=float(raw.get("damping", 0.98)),
        trend_lookback=int(raw.get("trend_lookback", 28)),
        seasonal_cycles=int(raw.get("seasonal_cycles", 4)),
    )


def _spec_from_custom(raw: Mapping[str, Any], source: str) -> DecompositionSpec:
    """custom 模式：组件级编排（未来契约，当前 fail-fast）。"""
    unknown = set(raw) - _CUSTOM_ALLOWED_KEYS - {"method"}
    if unknown:
        raise ValueError(
            f"Unknown decomposition custom keys in {source}: {sorted(unknown)}; "
            f"allowed: {sorted({'method'} | _CUSTOM_ALLOWED_KEYS)}"
        )
    raise ValueError(
        "decomposition method 'custom' is reserved for future component-level "
        "composition and is not implemented yet; use presets "
        f"{sorted(_VALID_PRESET_METHODS)} via the same 'decomposition' group."
    )


def resolve_decomposition_spec(cfg: Any) -> DecompositionSpec:
    """把 ``cfg.decomposition`` mapping 归一化为 DecompositionSpec。

    规则：

    - 未配置或空 mapping → ``method=none``；
    - preset 方法（none/linear/stl/mstl）→ ``_coerce_preset`` 校验并归一化；
    - ``custom`` → 组件级编排预留，fail-fast；
    - RESERVED_METHODS → fail-fast。

    legacy ``preprocessing.decomposition_*`` 散列字段已随 legacy 配置体系删除，
    不再解析；canonical YAML 的未知字段由 ``load_yaml_config()`` 在上游拦截。
    """
    source = "cfg"
    raw: Mapping[str, Any] = getattr(cfg, "decomposition", None) or {}

    if not raw:
        return DecompositionSpec(method="none")

    method = str(raw.get("method", "none")).lower()
    if method == "custom":
        return _spec_from_custom(raw, source)

    merged = _coerce_preset(raw, source)
    return DecompositionSpec(method=merged.method, composition=merged.composition, preset=merged)
