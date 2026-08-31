"""Canonical RecMO strategy executor."""

from forecasting_core.specs.strategy import StrategyName
from model_training.strategies.base import StandardStrategyExecutor


class RecMOExecutor(StandardStrategyExecutor):
    strategy_name = StrategyName.RECMO
