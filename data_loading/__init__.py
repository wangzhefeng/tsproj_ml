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
from data_loading.registry import FrameReader, SourceGenerator, SourceRegistry

__all__ = [
    "AuxiliaryProvider",
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
