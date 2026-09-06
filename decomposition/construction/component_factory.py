"""组件实例工厂：只负责 preset 到算法对象的构造，不引用 Pipeline。"""
from dataclasses import dataclass
from decomposition.contracts.interfaces import ComponentExtractor, ComponentForecaster
from decomposition.composition.additive import AdditiveComposer
from decomposition.extraction.polynomial import PolyTrendExtractor
from decomposition.extraction.seasonal import MSTLExtractor, STLExtractor
from decomposition.extrapolation.seasonal import PhaseTemplateForecaster
from decomposition.extrapolation.trend import TrendForecaster
from decomposition.configuration.spec import DecompositionSpec


@dataclass
class Components:
    extractor: ComponentExtractor | None
    trend: ComponentForecaster | None
    seasonal: ComponentForecaster | None
    composer: AdditiveComposer


def build_components(spec: DecompositionSpec) -> Components:
    expanded = spec.expand_preset()
    composer = AdditiveComposer()
    if not expanded.enabled:
        return Components(None, None, None, composer)
    extraction = expanded.extractor
    forecast = expanded.trend_forecaster
    if extraction is None or forecast is None:
        raise ValueError("Enabled decomposition requires extractor and trend specifications.")
    constructors = {"poly_trend": PolyTrendExtractor, "stl": STLExtractor, "mstl": MSTLExtractor}
    if extraction.type not in constructors or forecast.type != "poly_trend_forecast":
        raise ValueError("Unsupported decomposition component specification.")
    extractor = constructors[extraction.type](**extraction.params)
    trend = TrendForecaster(**forecast.params)
    seasonal_spec = expanded.seasonal_forecaster
    seasonal = None
    if seasonal_spec is not None:
        if seasonal_spec.type != "phase_template":
            raise ValueError("Unsupported seasonal forecaster specification.")
        seasonal = PhaseTemplateForecaster(cycles=seasonal_spec.params["cycles"])
    return Components(extractor, trend, seasonal, composer)
