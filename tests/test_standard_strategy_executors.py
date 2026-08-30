# -*- coding: utf-8 -*-
"""七种 canonical strategy executor 的数值轨迹与依赖契约。"""

import unittest

import numpy as np
import pandas as pd

from model_forecasting.specs import ForecastStrategySpec
from model_training.strategies import (
    DirectExecutor,
    DirMOExecutor,
    DirRecExecutor,
    DirRecMOExecutor,
    MIMOExecutor,
    RecMOExecutor,
    RecursiveExecutor,
    StrategyTargetPlan,
    TargetCoordinate,
)
from model_forecasting.tensors import PointForecastTensor


class CalculableEstimator:
    def __init__(self, offset, output_width):
        self.offset = float(offset)
        self.output_width = output_width
        self.predict_calls = []

    def predict(self, X):
        design = np.asarray(X, dtype=float)
        self.predict_calls.append(design.copy())
        base = design.sum(axis=1, keepdims=True) + self.offset
        return base + np.arange(self.output_width, dtype=float)


class FixedWidthStateProvider:
    def __init__(self, base, state_width):
        self.base = np.asarray(base, dtype=float)
        self.state_width = state_width
        self.calls = []

    def __call__(self, call_index, call_coordinates, dependencies, predicted):
        if dependencies:
            state_coordinates = dependencies[-self.state_width :]
            state = np.column_stack(
                tuple(predicted[coordinate] for coordinate in state_coordinates)
            )
        else:
            state = np.zeros((len(self.base), self.state_width), dtype=float)
        design = np.column_stack((self.base, state))
        self.calls.append((call_index, call_coordinates, dependencies, design.copy()))
        return design


class StrategyTargetPlanTests(unittest.TestCase):
    def test_h4_k2_coordinates_and_calls_are_time_major(self):
        targets = ("load", "temperature")
        expected_coordinates = tuple(
            TargetCoordinate(target, horizon_step)
            for horizon_step in range(1, 5)
            for target in targets
        )

        cases = (
            (ForecastStrategySpec("recursive"), (2, 2, 2, 2), (0, 0, 0, 0)),
            (ForecastStrategySpec("direct"), (2, 2, 2, 2), (0, 1, 2, 3)),
            (ForecastStrategySpec("mimo"), (8,), (0,)),
            (ForecastStrategySpec("recmo", 2), (4, 4), (0, 0)),
            (ForecastStrategySpec("dirrec"), (2, 2, 2, 2), (0, 1, 2, 3)),
            (ForecastStrategySpec("dirmo", 2), (4, 4), (0, 1)),
            (ForecastStrategySpec("dirrecmo", 2), (4, 4), (0, 1)),
        )

        for spec, call_widths, model_indices in cases:
            with self.subTest(strategy=spec.name.value):
                plan = StrategyTargetPlan.from_spec(spec, targets, horizon=4)
                self.assertEqual(plan.coordinates, expected_coordinates)
                self.assertEqual(
                    tuple(len(group) for group in plan.call_coordinates),
                    call_widths,
                )
                self.assertEqual(plan.model_indices, model_indices)
                self.assertEqual(plan.model_count, spec.resolve(4).model_count)

                for call_index, dependencies in enumerate(plan.dependencies):
                    expected = (
                        expected_coordinates[: sum(call_widths[:call_index])]
                        if spec.resolve(4).consumes_previous
                        else ()
                    )
                    self.assertEqual(dependencies, expected)


class StandardStrategyExecutorTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1.0]], dtype=float)
        self.series_ids = ("series-a",)
        self.forecast_times = pd.date_range("2026-08-28", periods=4, freq="1h")
        self.targets = ("load", "temperature")

    def _predict(self, executor_type, spec, estimators):
        target_plan = StrategyTargetPlan.from_spec(spec, self.targets, horizon=4)
        result = executor_type(spec, target_plan, estimators).predict(
            self.X,
            series_ids=self.series_ids,
            forecast_times=self.forecast_times,
        )
        self.assertIsInstance(result, PointForecastTensor)
        self.assertEqual(result.shape, (1, 4, 2))
        self.assertEqual(result.series_ids, self.series_ids)
        self.assertEqual(result.targets, self.targets)
        self.assertTrue(result.forecast_times.equals(self.forecast_times))
        return result.values[0]

    def test_recursive_exact_trajectory_and_previous_dependency(self):
        estimator = CalculableEstimator(10, 2)
        values = self._predict(
            RecursiveExecutor,
            ForecastStrategySpec("recursive"),
            (estimator,),
        )
        np.testing.assert_allclose(
            values,
            [[11, 12], [34, 35], [103, 104], [310, 311]],
        )
        self.assertEqual([call.shape[1] for call in estimator.predict_calls], [1, 3, 5, 7])
        np.testing.assert_allclose(estimator.predict_calls[-1], [[1, 11, 12, 34, 35, 103, 104]])

    def test_direct_exact_trajectory_without_previous_dependency(self):
        estimators = tuple(CalculableEstimator(offset, 2) for offset in (10, 20, 30, 40))
        values = self._predict(
            DirectExecutor,
            ForecastStrategySpec("direct"),
            estimators,
        )
        np.testing.assert_allclose(values, [[11, 12], [21, 22], [31, 32], [41, 42]])
        self.assertEqual([model.predict_calls[0].shape[1] for model in estimators], [1, 1, 1, 1])

    def test_mimo_exact_trajectory_in_one_call(self):
        estimator = CalculableEstimator(10, 8)
        values = self._predict(
            MIMOExecutor,
            ForecastStrategySpec("mimo"),
            (estimator,),
        )
        np.testing.assert_allclose(values, [[11, 12], [13, 14], [15, 16], [17, 18]])
        self.assertEqual(len(estimator.predict_calls), 1)

    def test_recmo_exact_trajectory_and_chunk_dependency(self):
        estimator = CalculableEstimator(10, 4)
        values = self._predict(
            RecMOExecutor,
            ForecastStrategySpec("recmo", 2),
            (estimator,),
        )
        np.testing.assert_allclose(values, [[11, 12], [13, 14], [61, 62], [63, 64]])
        self.assertEqual([call.shape[1] for call in estimator.predict_calls], [1, 5])
        np.testing.assert_allclose(estimator.predict_calls[-1], [[1, 11, 12, 13, 14]])

    def test_dirrec_exact_trajectory_and_previous_dependency(self):
        estimators = tuple(CalculableEstimator(offset, 2) for offset in (10, 20, 30, 40))
        values = self._predict(
            DirRecExecutor,
            ForecastStrategySpec("dirrec"),
            estimators,
        )
        np.testing.assert_allclose(values, [[11, 12], [44, 45], [143, 144], [440, 441]])
        self.assertEqual([model.predict_calls[0].shape[1] for model in estimators], [1, 3, 5, 7])

    def test_dirmo_exact_trajectory_without_previous_dependency(self):
        estimators = (CalculableEstimator(10, 4), CalculableEstimator(20, 4))
        values = self._predict(
            DirMOExecutor,
            ForecastStrategySpec("dirmo", 2),
            estimators,
        )
        np.testing.assert_allclose(values, [[11, 12], [13, 14], [21, 22], [23, 24]])
        self.assertEqual([model.predict_calls[0].shape[1] for model in estimators], [1, 1])

    def test_dirrecmo_exact_trajectory_and_chunk_dependency(self):
        estimators = (CalculableEstimator(10, 4), CalculableEstimator(20, 4))
        values = self._predict(
            DirRecMOExecutor,
            ForecastStrategySpec("dirrecmo", 2),
            estimators,
        )
        np.testing.assert_allclose(values, [[11, 12], [13, 14], [71, 72], [73, 74]])
        self.assertEqual([model.predict_calls[0].shape[1] for model in estimators], [1, 5])

    def test_all_executors_support_single_target(self):
        cases = (
            (RecursiveExecutor, ForecastStrategySpec("recursive"), (CalculableEstimator(1, 1),)),
            (DirectExecutor, ForecastStrategySpec("direct"), tuple(CalculableEstimator(i, 1) for i in range(4))),
            (MIMOExecutor, ForecastStrategySpec("mimo"), (CalculableEstimator(1, 4),)),
            (RecMOExecutor, ForecastStrategySpec("recmo", 2), (CalculableEstimator(1, 2),)),
            (DirRecExecutor, ForecastStrategySpec("dirrec"), tuple(CalculableEstimator(i, 1) for i in range(4))),
            (DirMOExecutor, ForecastStrategySpec("dirmo", 2), tuple(CalculableEstimator(i, 2) for i in range(2))),
            (DirRecMOExecutor, ForecastStrategySpec("dirrecmo", 2), tuple(CalculableEstimator(i, 2) for i in range(2))),
        )
        for executor_type, spec, estimators in cases:
            with self.subTest(strategy=spec.name.value):
                plan = StrategyTargetPlan.from_spec(spec, ("load",), horizon=4)
                result = executor_type(spec, plan, estimators).predict(
                    self.X,
                    series_ids=self.series_ids,
                    forecast_times=self.forecast_times,
                )
                self.assertEqual(result.shape, (1, 4, 1))

    def test_recursive_and_recmo_accept_fixed_schema_state_feature_provider(self):
        cases = (
            (
                RecursiveExecutor,
                ForecastStrategySpec("recursive"),
                (CalculableEstimator(1, 2),),
                2,
            ),
            (
                RecMOExecutor,
                ForecastStrategySpec("recmo", 2),
                (CalculableEstimator(1, 4),),
                4,
            ),
        )
        for executor_type, spec, estimators, state_width in cases:
            with self.subTest(strategy=spec.name.value):
                plan = StrategyTargetPlan.from_spec(spec, self.targets, horizon=4)
                provider = FixedWidthStateProvider(self.X, state_width)
                result = executor_type(spec, plan, estimators).predict(
                    self.X,
                    series_ids=self.series_ids,
                    forecast_times=self.forecast_times,
                    feature_provider=provider,
                )

                self.assertEqual(result.shape, (1, 4, 2))
                self.assertEqual(
                    {call[-1].shape[1] for call in provider.calls},
                    {1 + state_width},
                )
                self.assertEqual(
                    {call.shape[1] for call in estimators[0].predict_calls},
                    {1 + state_width},
                )


if __name__ == "__main__":
    unittest.main()
