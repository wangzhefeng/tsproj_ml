"""Canonical forecasting specifications."""

from forecasting_core.specs.config import ForecastConfigSpec, parse_model_config
from forecasting_core.specs.data import (
    AvailabilityPolicy,
    ColumnRole,
    ColumnSpec,
    DataSourceSpec,
    DataSpec,
)
from forecasting_core.specs.estimator import EstimatorSpec, TargetAdapter
from forecasting_core.specs.feature import FeatureSpec
from forecasting_core.specs.output import OutputSpec
from forecasting_core.specs.problem import ForecastProblemSpec
from forecasting_core.specs.probabilistic import ProbabilisticConfigSpec
from forecasting_core.specs.strategy import ForecastStrategySpec, StrategyName
from forecasting_core.specs.validation import (
    CalendarMonthBacktestSpec,
    FixedStepBacktestSpec,
    RuntimeValidationSpec,
)

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
    "ProbabilisticConfigSpec",
    "FixedStepBacktestSpec",
    "CalendarMonthBacktestSpec",
    "RuntimeValidationSpec",
    "OutputSpec",
    "ForecastConfigSpec",
    "parse_model_config",
]
