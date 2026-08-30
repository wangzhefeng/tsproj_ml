"""Canonical RecMO strategy executor."""

from model_forecasting.specs.strategy import StrategyName
from model_training.strategies.base import StandardStrategyExecutor


class RecMOExecutor(StandardStrategyExecutor):
    strategy_name = StrategyName.RECMO
