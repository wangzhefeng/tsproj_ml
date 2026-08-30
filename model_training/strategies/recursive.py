"""Canonical Recursive strategy executor."""

from model_forecasting.specs.strategy import StrategyName
from model_training.strategies.base import StandardStrategyExecutor


class RecursiveExecutor(StandardStrategyExecutor):
    strategy_name = StrategyName.RECURSIVE
