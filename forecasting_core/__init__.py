# -*- coding: utf-8 -*-
"""Stable forecasting contracts with no pipeline/runtime dependencies."""

from forecasting_core.artifacts import (
    ForecastModelBundle,
    MarginalForecastDistribution,
    QuantileGrid,
)
from forecasting_core.probabilistic_spec import ProbabilisticSpec
from forecasting_core.specs import *
from forecasting_core.tensors import (
    MarginalQuantileForecastTensor,
    PointForecastTensor,
    SampleForecastTensor,
    require_matching_point_axes,
)

__all__ = [
    "ForecastModelBundle",
    "MarginalForecastDistribution",
    "ProbabilisticSpec",
    "QuantileGrid",
    "MarginalQuantileForecastTensor",
    "PointForecastTensor",
    "SampleForecastTensor",
    "require_matching_point_axes",
]
