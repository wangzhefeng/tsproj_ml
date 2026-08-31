"""Canonical DirRec strategy executor."""

from forecasting_core.specs.strategy import StrategyName
from model_training.strategies.base import StandardStrategyExecutor


class DirRecExecutor(StandardStrategyExecutor):
    strategy_name = StrategyName.DIRREC
