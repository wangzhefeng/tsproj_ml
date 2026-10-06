"""边际预测分布与联合采样能力边界。"""
from __future__ import annotations

from dataclasses import dataclass, field
from forecasting_core.tensors.point import PointForecastTensor
from forecasting_core.tensors.quantile import MarginalQuantileForecastTensor
from typing import Any
import numpy as np


@dataclass
class MarginalForecastDistribution:
    """Canonical per-target marginal quantiles; no joint dependence model."""

    point: PointForecastTensor
    quantiles: MarginalQuantileForecastTensor
    dependence_model: None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.point, PointForecastTensor):
            raise TypeError("point must be a PointForecastTensor")
        if not isinstance(self.quantiles, MarginalQuantileForecastTensor):
            raise TypeError(
                "quantiles must be a MarginalQuantileForecastTensor"
            )
        if self.dependence_model is not None:
            raise ValueError(
                "joint dependence models are unsupported; dependence_model must be None"
            )
        quantile_point = self.quantiles.point()
        if (
            self.point.series_ids != quantile_point.series_ids
            or self.point.targets != quantile_point.targets
            or not self.point.forecast_times.equals(quantile_point.forecast_times)
        ):
            raise ValueError("point and quantiles must have identical axes")
        if not np.allclose(
            self.point.values,
            quantile_point.values,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("point must equal the configured point quantile")
        self.metadata = {
            **dict(self.metadata),
            "distribution_kind": "marginal_quantile",
            "dependence_model": None,
        }

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.quantiles.shape


def generate_joint_samples(
    quantiles: MarginalQuantileForecastTensor,
    *,
    n_samples: int,
):
    if not isinstance(quantiles, MarginalQuantileForecastTensor):
        raise TypeError("quantiles must be a MarginalQuantileForecastTensor")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    raise NotImplementedError(
        "joint sample generation is not implemented; marginal quantiles have "
        "dependence_model=None"
    )
