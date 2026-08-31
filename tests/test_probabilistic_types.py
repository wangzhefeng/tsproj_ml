# -*- coding: utf-8 -*-
"""概率预测强类型对象契约测试。"""

import unittest

import numpy as np
import pandas as pd

from forecasting_core.artifacts import (
    ForecastDistribution,
    PredictionIntervalForecast,
    QuantileGrid,
)


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


class ForecastDistributionTest(unittest.TestCase):
    @staticmethod
    def _valid_distribution(**overrides):
        grid = QuantileGrid((0.1, 0.5, 0.9), point_level=0.5)
        quantiles = np.array([[8.0, 10.0, 12.0], [9.0, 11.0, 13.0]])
        values = {
            "point": quantiles[:, 1],
            "quantile_grid": grid,
            "quantile_values": quantiles,
            "intervals": {
                "pi90": PredictionIntervalForecast(
                    name="pi90",
                    lower=np.array([7.0, 8.0]),
                    upper=np.array([13.0, 14.0]),
                    target_coverage=0.9,
                    method="cqr",
                    base_quantiles=(0.1, 0.9),
                )
            },
            "space": "target",
            "quantile_stage": "processed",
            "forecast_times": pd.date_range("2026-08-01", periods=2, freq="1D"),
            "metadata": {"recursive_propagation": "median_path"},
        }
        values.update(overrides)
        return ForecastDistribution(**values)

    def test_valid_distribution_keeps_point_bound_to_q50(self):
        distribution = self._valid_distribution()

        np.testing.assert_array_equal(
            distribution.point,
            distribution.quantile_values[:, distribution.quantile_grid.point_index],
        )
        self.assertEqual(distribution.n_steps, 2)

    def test_point_must_equal_configured_point_quantile(self):
        with self.assertRaisesRegex(ValueError, "point must equal"):
            self._valid_distribution(point=np.array([10.0, 99.0]))

    def test_shapes_times_and_interval_bounds_are_strict(self):
        with self.assertRaisesRegex(ValueError, "quantile_values shape"):
            self._valid_distribution(quantile_values=np.ones((2, 2)))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self._valid_distribution(
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

    def test_space_and_stage_are_closed_enums(self):
        with self.assertRaisesRegex(ValueError, "space"):
            self._valid_distribution(space="unknown")
        with self.assertRaisesRegex(ValueError, "quantile_stage"):
            self._valid_distribution(quantile_stage="calibrated")


if __name__ == "__main__":
    unittest.main()
