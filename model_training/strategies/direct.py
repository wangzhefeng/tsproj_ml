"""Canonical Direct strategy executor."""

from model_forecasting.specs.strategy import StrategyName
from model_training.strategies.base import StandardStrategyExecutor


class DirectExecutor(StandardStrategyExecutor):
    strategy_name = StrategyName.DIRECT
