# -*- coding: utf-8 -*-
"""概率预测强类型对象契约测试。"""

import unittest
from typing import Any

import numpy as np
import pandas as pd

from forecasting_core.artifacts import (
    MarginalForecastDistribution,
    PredictionIntervalForecast,
    QuantileGrid,
)
from forecasting_core.tensors import MarginalQuantileForecastTensor, PointForecastTensor


class QuantileGridTest(unittest.TestCase):
    def test_numeric_grid_is_authoritative_and_column_codec_is_injective(self):
        grid = QuantileGrid((0.101, 0.104, 0.5, 0.9), point_level=0.5)

        self.assertEqual(grid.index_of(0.5), 2)
        self.assertEqual(grid.column_name(0.101), "predict_q10p1")
        self.assertEqual(grid.column_name(0.104), "predict_q10p4")
        self.assertEqual(len({grid.column_name(level) for level in grid.levels}), 4)

    def test_unknown_level_fails_instead_of_using_nearest_quantile(self):
        grid = QuantileGrid((0.1, 0.5, 0.9), point_level=0.5)

        with self.assertRaisesRegex(ValueError, "not present"):
            grid.index_of(0.51)


class MarginalDistributionContractTest(unittest.TestCase):
    @staticmethod
    def _quantiles(**overrides):
        values: dict[str, Any] = dict(
            values=np.array([8., 10., 12., 9., 11., 13.]).reshape(1, 2, 1, 3),
            levels=(0.1, 0.5, 0.9), point_level=0.5,
            series_ids=("A",), targets=("load",),
            forecast_times=pd.date_range("2026-08-01", periods=2, freq="1D"),
        )
        values.update(overrides)
        return MarginalQuantileForecastTensor(**values)

    def test_valid_distribution_keeps_point_bound_to_q50(self):
        quantiles = self._quantiles()
        distribution = MarginalForecastDistribution(point=quantiles.point(), quantiles=quantiles)
        np.testing.assert_array_equal(
            distribution.point.values, quantiles.values[..., 1],
        )
        self.assertEqual(distribution.shape, (1, 2, 1, 3))

    def test_point_must_equal_configured_point_quantile(self):
        with self.assertRaisesRegex(ValueError, "point must equal"):
            quantiles = self._quantiles()
            point = PointForecastTensor(
                values=quantiles.point().values + 1,
                series_ids=quantiles.series_ids, targets=quantiles.targets,
                forecast_times=quantiles.forecast_times,
            )
            MarginalForecastDistribution(point=point, quantiles=quantiles)

    def test_shapes_times_and_interval_bounds_are_strict(self):
        with self.assertRaises(ValueError):
            self._quantiles(values=np.ones((2, 2)))
        with self.assertRaises(ValueError):
            self._quantiles(
                forecast_times=pd.DatetimeIndex(["2026-08-01", "2026-08-01"])
            )
        with self.assertRaisesRegex(ValueError, "lower <= upper"):
            PredictionIntervalForecast(
                name="bad",
                lower=np.array([2.0]),
                upper=np.array([1.0]),
                target_coverage=0.8,
                method="cqr",
                base_quantiles=(0.1, 0.9),
            )

    def test_point_and_quantiles_require_identical_axes(self):
        quantiles = self._quantiles()
        for overrides in ({"series_ids": ("B",)}, {"targets": ("power",)},
                          {"forecast_times": quantiles.forecast_times + pd.Timedelta(days=1)}):
            with self.subTest(overrides=overrides):
                other = self._quantiles(**overrides)
                with self.assertRaisesRegex(ValueError, "identical axes"):
                    MarginalForecastDistribution(point=other.point(), quantiles=quantiles)


if __name__ == "__main__":
    unittest.main()
