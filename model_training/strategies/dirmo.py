"""Canonical DirMO strategy executor."""

from forecasting_core.specs.strategy import StrategyName
from model_training.strategies.base import StandardStrategyExecutor


class DirMOExecutor(StandardStrategyExecutor):
    strategy_name = StrategyName.DIRMO
