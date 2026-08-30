"""Canonical DirRec strategy executor."""

from model_forecasting.specs.strategy import StrategyName
from model_training.strategies.base import StandardStrategyExecutor


class DirRecExecutor(StandardStrategyExecutor):
    strategy_name = StrategyName.DIRREC
