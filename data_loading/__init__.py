"""Canonical forecasting data registry and information-set contracts."""

from data_loading.information_set import (
    InformationSetRequest,
    MaterializedInformationSet,
    SourceLineage,
)
from data_loading.providers import (
    AuxiliaryProvider,
    CompositeProvider,
    EndogenousFutureProvider,
    PersistenceProvider,
    ProvidedScenarioProvider,
    create_endogenous_future_provider,
)
from data_loading.holiday_generator import (
    chinese_holiday_frame,
    chinese_holiday_generator,
    generator_name as chinese_holiday_generator_name,
)
from data_loading.registry import FrameReader, SourceGenerator, SourceRegistry

BUILTIN_GENERATORS: dict[str, SourceGenerator] = {
    chinese_holiday_generator_name(): chinese_holiday_generator,
}

__all__ = [
    "AuxiliaryProvider",
    "BUILTIN_GENERATORS",
    "chinese_holiday_frame",
    "chinese_holiday_generator",
    "CompositeProvider",
    "EndogenousFutureProvider",
    "FrameReader",
    "InformationSetRequest",
    "MaterializedInformationSet",
    "PersistenceProvider",
    "ProvidedScenarioProvider",
    "SourceGenerator",
    "SourceLineage",
    "SourceRegistry",
    "create_endogenous_future_provider",
]
