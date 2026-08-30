"""Canonical MIMO strategy executor."""

from model_forecasting.specs.strategy import StrategyName
from model_training.strategies.base import StandardStrategyExecutor


class MIMOExecutor(StandardStrategyExecutor):
    strategy_name = StrategyName.MIMO
