"""Canonical forecasting specifications."""

from model_forecasting.specs.config import ForecastConfigSpec, parse_model_config
from model_forecasting.specs.data import (
    AvailabilityPolicy,
    ColumnRole,
    ColumnSpec,
    DataSourceSpec,
    DataSpec,
)
from model_forecasting.specs.estimator import EstimatorSpec, TargetAdapter
from model_forecasting.specs.feature import FeatureSpec
from model_forecasting.specs.problem import ForecastProblemSpec
from model_forecasting.specs.strategy import ForecastStrategySpec, StrategyName

__all__ = [
    "ForecastProblemSpec",
    "FeatureSpec",
    "TargetAdapter",
    "EstimatorSpec",
    "StrategyName",
    "ForecastStrategySpec",
    "AvailabilityPolicy",
    "ColumnRole",
    "ColumnSpec",
    "DataSourceSpec",
    "DataSpec",
    "ForecastConfigSpec",
    "parse_model_config",
]
