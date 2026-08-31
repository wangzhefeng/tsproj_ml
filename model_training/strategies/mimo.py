"""Canonical MIMO strategy executor."""

from forecasting_core.specs.strategy import StrategyName
from model_training.strategies.base import StandardStrategyExecutor


class MIMOExecutor(StandardStrategyExecutor):
    strategy_name = StrategyName.MIMO
