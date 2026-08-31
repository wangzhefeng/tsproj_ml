"""Canonical Direct strategy executor."""

from forecasting_core.specs.strategy import StrategyName
from model_training.strategies.base import StandardStrategyExecutor


class DirectExecutor(StandardStrategyExecutor):
    strategy_name = StrategyName.DIRECT
