"""Canonical DirMO strategy executor."""

from model_forecasting.specs.strategy import StrategyName
from model_training.strategies.base import StandardStrategyExecutor


class DirMOExecutor(StandardStrategyExecutor):
    strategy_name = StrategyName.DIRMO
