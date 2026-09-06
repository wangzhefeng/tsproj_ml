"""已实现 preset 的参数与固定组件配方；不解析 YAML，不构造算法对象。"""
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ComponentSpec:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PresetParams:
    method: str
    composition: str = "additive"
    periods: tuple[int, ...] = ()
    robust: bool = True
    trend_degree: int = 1
    trend_forecast: str = "polynomial"
    damping: float = 0.98
    trend_lookback: int = 28
    seasonal_cycles: int = 4


def preset_components(p: PresetParams) -> dict[str, ComponentSpec | None]:
    seasonal = p.method in {"stl", "mstl"}
    return {
        "extractor": ComponentSpec(p.method if seasonal else "poly_trend",
                                   {"periods": p.periods, "robust": p.robust} if seasonal else {"degree": p.trend_degree}),
        "trend_forecaster": ComponentSpec("poly_trend_forecast", {
            "degree": p.trend_degree, "mode": p.trend_forecast,
            "damping": p.damping, "lookback": p.trend_lookback,
            "source": "trend" if seasonal else "target",
        }),
        "seasonal_forecaster": ComponentSpec("phase_template", {"cycles": p.seasonal_cycles, "periods": p.periods}) if seasonal else None,
        "composer": ComponentSpec(p.composition),
    }
