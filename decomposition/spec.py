# -*- coding: utf-8 -*-
"""时间序列分解的配置契约层。

将两类外部输入归一化为唯一的运行时对象 :class:`DecompositionSpec`：

1. 旧写法 ``overrides.preprocessing.decomposition_*`` 散列字段（兼容层）；
2. 新写法 ``overrides.decomposition`` 原始 mapping（规范层）。

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
    "PresetParams",
    "resolve_decomposition_spec",
]

# 兼容旧写法的 preset 字段（preprocessing.decomposition_* 后缀 → spec 字段名）
_PRESET_FIELD_MAP: dict[str, str] = {
    "decomposition_method": "method",
    "decomposition_periods": "periods",
    "decomposition_robust": "robust",
    "decomposition_trend_degree": "trend_degree",
    "decomposition_trend_forecast": "trend_forecast",
    "decomposition_damping": "damping",
    "decomposition_trend_lookback": "trend_lookback",
    "decomposition_seasonal_cycles": "seasonal_cycles",
}

# 新写法 preset 模式下允许的键（不含 method 本身）
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

# PresetParams 的字段默认值（与 PreprocessingConfig 保持一致）
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


def _preset_params_from_legacy(cfg: Any) -> PresetParams | None:
    """从旧 preprocessing.decomposition_* 字段构造 PresetParams；全部默认返回 None。"""
    method = str(getattr(cfg, "decomposition_method", "none") or "none").lower()
    if method == "none" and not any(
        getattr(cfg, key, None) not in (None, "", [], ())
        for key in _PRESET_FIELD_MAP
        if key != "decomposition_method"
    ):
        # method=none 且无其他显式旧字段 → 视为未配置
        return None
    periods_raw = getattr(cfg, "decomposition_periods", None) or []
    return PresetParams(
        method=method,
        composition="additive",
        periods=tuple(int(p) for p in periods_raw),
        robust=bool(getattr(cfg, "decomposition_robust", True)),
        trend_degree=int(getattr(cfg, "decomposition_trend_degree", 1)),
        trend_forecast=str(
            getattr(cfg, "decomposition_trend_forecast", "polynomial") or "polynomial"
        ).lower(),
        damping=float(getattr(cfg, "decomposition_damping", 0.98)),
        trend_lookback=int(getattr(cfg, "decomposition_trend_lookback", 28)),
        seasonal_cycles=int(getattr(cfg, "decomposition_seasonal_cycles", 4)),
    )


def _legacy_explicitly_set(cfg: Any) -> bool:
    """旧字段是否有任何显式偏离非 method 默认值的设置。

    判定口径：decomposition_method != 'none'，或任一参数字段非默认值。
    """
    method = str(getattr(cfg, "decomposition_method", "none") or "none").lower()
    if method != "none":
        return True
    checks = {
        "decomposition_periods": (),
        "decomposition_robust": True,
        "decomposition_trend_degree": 1,
        "decomposition_trend_forecast": "polynomial",
        "decomposition_damping": 0.98,
        "decomposition_trend_lookback": 28,
        "decomposition_seasonal_cycles": 4,
    }
    for key, default in checks.items():
        value = getattr(cfg, key, default)
        if isinstance(default, tuple):
            if tuple(value or ()):
                return True
        elif value != default:
            return True
    return False


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
    return PresetParams(
        method=method,
        composition=composition,
        periods=periods,
        robust=bool(raw.get("robust", True)),
        trend_degree=int(raw.get("trend_degree", 1)),
        trend_forecast=str(raw.get("trend_forecast", "polynomial")).lower(),
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
    """把新旧两类配置输入归一化为 DecompositionSpec。

    合并规则（见设计文档 §5.6）：

    - 只有旧字段 → 旧 preset；
    - 只有新字段 → 新 mapping（custom 则 fail-fast）；
    - 都没有 → method=none；
    - 都有 → 两侧归一化后逐项比较，不一致直接 ValueError；
    - custom 与任何旧字段并存 → ValueError。
    """
    source = "cfg"
    new_raw: Mapping[str, Any] = getattr(cfg, "decomposition", None) or {}
    legacy_set = _legacy_explicitly_set(cfg)
    new_set = bool(new_raw)

    if new_set:
        method = str(new_raw.get("method", "none")).lower()
        if method == "custom":
            if legacy_set:
                raise ValueError(
                    "decomposition method 'custom' cannot be combined with legacy "
                    "'preprocessing.decomposition_*' fields; use one style only."
                )
            return _spec_from_custom(new_raw, source)

    if not new_set and not legacy_set:
        return DecompositionSpec(method="none")

    legacy_params = _preset_params_from_legacy(cfg) if legacy_set else None
    new_params = _coerce_preset(new_raw, source) if new_set else None

    if legacy_params is not None and new_params is not None:
        if legacy_params != new_params:
            raise ValueError(
                "Conflicting decomposition config: legacy "
                "'preprocessing.decomposition_*' and new 'decomposition' group "
                f"resolve to different specs (legacy={legacy_params}, new={new_params})."
            )
        merged = new_params
    else:
        merged = legacy_params or new_params
    assert merged is not None  # 至少一侧存在
    if merged.method in RESERVED_METHODS:  # pragma: no cover - _coerce 已拦截
        raise ValueError(f"decomposition method '{merged.method}' is reserved but not implemented.")
    return DecompositionSpec(method=merged.method, composition=merged.composition, preset=merged)
