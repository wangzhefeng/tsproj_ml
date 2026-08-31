"""Canonical standard multi-step forecasting strategy executors."""

from forecasting_core.specs.strategy import ForecastStrategySpec, StrategyName
from model_training.strategies.base import (
    AdapterPredictor,
    CanonicalStrategyArtifact,
    StrategyModelGroupArtifact,
    StrategyTargetPlan,
    TargetCoordinate,
)
from model_training.strategies.direct import DirectExecutor
from model_training.strategies.dirmo import DirMOExecutor
from model_training.strategies.dirrec import DirRecExecutor
from model_training.strategies.dirrecmo import DirRecMOExecutor
from model_training.strategies.mimo import MIMOExecutor
from model_training.strategies.recmo import RecMOExecutor
from model_training.strategies.recursive import RecursiveExecutor


_EXECUTORS = {
    StrategyName.RECURSIVE: RecursiveExecutor,
    StrategyName.DIRECT: DirectExecutor,
    StrategyName.MIMO: MIMOExecutor,
    StrategyName.RECMO: RecMOExecutor,
    StrategyName.DIRREC: DirRecExecutor,
    StrategyName.DIRMO: DirMOExecutor,
    StrategyName.DIRRECMO: DirRecMOExecutor,
}


def get_standard_executor(spec: ForecastStrategySpec):
    if not isinstance(spec, ForecastStrategySpec):
        raise TypeError("spec must be ForecastStrategySpec")
    return _EXECUTORS[spec.name]

__all__ = [
    "AdapterPredictor",
    "DirectExecutor",
    "DirMOExecutor",
    "DirRecExecutor",
    "DirRecMOExecutor",
    "MIMOExecutor",
    "RecMOExecutor",
    "RecursiveExecutor",
    "CanonicalStrategyArtifact",
    "StrategyModelGroupArtifact",
    "StrategyTargetPlan",
    "TargetCoordinate",
    "get_standard_executor",
]
