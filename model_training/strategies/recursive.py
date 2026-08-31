"""Canonical Recursive strategy executor."""

from forecasting_core.specs.strategy import StrategyName
from model_training.strategies.base import StandardStrategyExecutor


class RecursiveExecutor(StandardStrategyExecutor):
    strategy_name = StrategyName.RECURSIVE
