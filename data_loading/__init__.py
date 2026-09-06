"""Canonical forecasting data registry and information-set contracts."""

from data_loading.information.information_set import (
    InformationSetRequest,
    MaterializedInformationSet,
    SourceLineage,
    TargetAccess,
)
from data_loading.information.providers import (
    AuxiliaryProvider,
    CompositeProvider,
    EndogenousFutureProvider,
    PersistenceProvider,
    ProvidedScenarioProvider,
    create_endogenous_future_provider,
)
from data_loading.calendar_generator import (
    BUILTIN_GENERATORS,
    chinese_holiday_frame,
    chinese_holiday_generator,
    GENERATOR_NAME as CHINESE_HOLIDAY_GENERATOR_NAME,
)
from data_loading.registry import FrameReader, SourceGenerator, SourceRegistry


__all__ = [
    "AuxiliaryProvider",
    "BUILTIN_GENERATORS",
    "CHINESE_HOLIDAY_GENERATOR_NAME",
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
    "TargetAccess",
    "SourceRegistry",
    "create_endogenous_future_provider",
]
