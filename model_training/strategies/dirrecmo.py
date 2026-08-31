"""Canonical DirRecMO strategy executor."""

from forecasting_core.specs.strategy import StrategyName
from model_training.strategies.base import StandardStrategyExecutor


class DirRecMOExecutor(StandardStrategyExecutor):
    strategy_name = StrategyName.DIRRECMO
