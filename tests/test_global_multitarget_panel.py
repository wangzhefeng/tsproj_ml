# -*- coding: utf-8 -*-
"""Canonical Trainer/Forecaster Local and Global multi-target matrix tests."""

import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from model_training.estimators import EstimatorCapabilities
from forecasting_core.specs import (
    ColumnSpec,
    DataSourceSpec,
    DataSpec,
    EstimatorSpec,
    FeatureSpec,
    ForecastConfigSpec,
    ForecastProblemSpec,
    ForecastStrategySpec,
)
from forecasting_core.tensors import PointForecastTensor
from model_forecasting.forecaster import CanonicalForecaster
from model_training.trainer import CanonicalTrainer


STRATEGIES = (
    ("recursive", None),
    ("direct", None),
    ("mimo", None),
    ("recmo", 2),
    ("dirrec", None),
    ("dirmo", 2),
    ("dirrecmo", 2),
)


class CanonicalTrainerForecasterMatrixTest(unittest.TestCase):
    capabilities = EstimatorCapabilities(
        scalar_target=True,
        scalar_quantile=False,
        native_multi_target_point=True,
        native_multi_target_quantile=False,
        sample_weight=True,
        categorical=False,
        nan_support=False,
    )

    @staticmethod
    def build_config(
        strategy,
        chunk,
        targets,
        *,
        global_scope,
        single_model_horizon=False,
    ):
        series_id_cols = ("series_id",) if global_scope else ()
        columns = []
        if global_scope:
            columns.append(ColumnSpec("series_id", "key", categorical=True))
        columns.extend(ColumnSpec(target, "target") for target in targets)
        return ForecastConfigSpec(
            problem=ForecastProblemSpec(
                time_col="time",
                freq="1h",
                horizon=4,
                targets=targets,
                training_scope="global" if global_scope else "local",
                series_id_cols=series_id_cols,
            ),
            data=DataSpec(
                (
                    DataSourceSpec(
                        name="targets",
                        source_type="file",
                        columns=tuple(columns),
                        history_path="unused.csv",
                        time_col="time",
                        series_id_cols=series_id_cols,
                        availability="source_time",
                    ),
                )
            ),
            features=FeatureSpec(
                target_lags={},
                observed_past_lags={},
                datetime_features=(),
                transformations=(
                    {}
                    if not single_model_horizon
                    else {
                        "direct": {
                            "layout": "single_model_horizon",
                            "align_to_target": True,
                            "horizon_feature": {
                                "name": "call_index",
                                "cyclical": False,
                            },
                        }
                    }
                ),
            ),
            strategy=ForecastStrategySpec(strategy, chunk),
            estimator=EstimatorSpec(
                model_type="linear_regression",
                target_adapter="independent",
            ),
            probabilistic={},
            validation={},
            output={},
        )

    @staticmethod
    def target_values(x, n_targets):
        return np.stack(
            [
                np.stack(
                    [2.0 * x + 10.0 * (step + 1) + target for target in range(n_targets)],
                    axis=1,
                )
                for step in range(4)
            ],
            axis=1,
        )

    @staticmethod
    def call_designs(x, n_calls):
        return tuple(
            np.column_stack((x, np.full(len(x), float(call_index))))
            for call_index in range(n_calls)
        )

    def run_case(self, strategy, chunk, targets, *, global_scope):
        config = self.build_config(
            strategy,
            chunk,
            targets,
            global_scope=global_scope,
        )
        resolved = config.strategy.resolve(config.problem.horizon)
        x_train = np.arange(1.0, 21.0)
        y_train = self.target_values(x_train, len(targets))
        trainer = CanonicalTrainer(
            config,
            estimator_factory=LinearRegression,
            capabilities=self.capabilities,
            feature_schema=("x", "call_index"),
        )

        artifact = trainer.train(
            self.call_designs(x_train, resolved.n_calls),
            y_train,
            n_series=2 if global_scope else 1,
        )

        x_predict = np.array([100.0, 200.0]) if global_scope else np.array([100.0])
        call_designs = self.call_designs(x_predict, resolved.n_calls)
        forecaster = CanonicalForecaster(config, artifact)
        result = forecaster.predict(
            call_designs[0],
            series_ids=("A", "B") if global_scope else ("__local__",),
            forecast_times=pd.date_range("2026-09-01", periods=4, freq="1h"),
            feature_provider=lambda call_index, *_: call_designs[call_index],
        )

        self.assertIsInstance(result, PointForecastTensor)
        self.assertEqual(
            result.shape,
            (2 if global_scope else 1, 4, len(targets)),
        )
        np.testing.assert_allclose(
            result.values,
            self.target_values(x_predict, len(targets)),
            atol=1e-8,
        )
        self.assertEqual(artifact.schema_version, 2)
        self.assertEqual(artifact.model_count, resolved.model_count)
        self.assertEqual(artifact.H, 4)
        self.assertEqual(artifact.K, len(targets))
        self.assertEqual(artifact.feature_schema, ("x", "call_index"))
        self.assertEqual(artifact.estimator_coupling, "independent")

    def test_local_k1_and_k2_run_all_seven_strategies(self):
        for strategy, chunk in STRATEGIES:
            for targets in (("load",), ("load", "power")):
                with self.subTest(strategy=strategy, targets=targets):
                    self.run_case(
                        strategy,
                        chunk,
                        targets,
                        global_scope=False,
                    )

    def test_global_n2_k2_runs_all_seven_strategies_without_series_mixing(self):
        for strategy, chunk in STRATEGIES:
            with self.subTest(strategy=strategy):
                self.run_case(
                    strategy,
                    chunk,
                    ("load", "power"),
                    global_scope=True,
                )

    def test_single_model_horizon_shares_one_model_for_local_and_global(self):
        for global_scope, targets in (
            (False, ("load",)),
            (True, ("load", "power")),
        ):
            with self.subTest(global_scope=global_scope, targets=targets):
                config = self.build_config(
                    "direct",
                    None,
                    targets,
                    global_scope=global_scope,
                    single_model_horizon=True,
                )
                x_train = np.arange(1.0, 21.0)
                y_train = self.target_values(x_train, len(targets))
                call_designs = self.call_designs(x_train, 4)
                artifact = CanonicalTrainer(
                    config,
                    estimator_factory=LinearRegression,
                    capabilities=self.capabilities,
                    feature_schema=("x", "call_index"),
                ).train(
                    call_designs,
                    y_train,
                    n_series=2 if global_scope else 1,
                )

                self.assertEqual(artifact.model_count, 1)
                self.assertEqual(artifact.target_plan.model_indices, (0, 0, 0, 0))
                expected_estimators = len(targets)
                self.assertEqual(
                    len(
                        getattr(
                            artifact.model_groups[0].predictor.adapter,
                            "estimators",
                        )
                    ),
                    expected_estimators,
                )

                x_predict = (
                    np.array([100.0, 200.0])
                    if global_scope
                    else np.array([100.0])
                )
                prediction_designs = self.call_designs(x_predict, 4)
                prediction = CanonicalForecaster(config, artifact).predict(
                    prediction_designs[0],
                    series_ids=("A", "B") if global_scope else ("__local__",),
                    forecast_times=pd.date_range(
                        "2026-09-01",
                        periods=4,
                        freq="1h",
                    ),
                    feature_provider=lambda call_index, *_: prediction_designs[
                        call_index
                    ],
                )
                np.testing.assert_allclose(
                    prediction.values,
                    self.target_values(x_predict, len(targets)),
                    atol=1e-8,
                )

    def test_forecaster_rejects_artifact_from_different_direct_layout(self):
        standard = self.build_config(
            "direct",
            None,
            ("load",),
            global_scope=False,
        )
        shared = self.build_config(
            "direct",
            None,
            ("load",),
            global_scope=False,
            single_model_horizon=True,
        )
        x_train = np.arange(1.0, 21.0)
        artifact = CanonicalTrainer(
            standard,
            estimator_factory=LinearRegression,
            capabilities=self.capabilities,
            feature_schema=("x", "call_index"),
        ).train(
            self.call_designs(x_train, 4),
            self.target_values(x_train, 1),
        )

        with self.assertRaisesRegex(ValueError, "target plan"):
            CanonicalForecaster(shared, artifact)


if __name__ == "__main__":
    unittest.main()
