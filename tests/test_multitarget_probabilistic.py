# -*- coding: utf-8 -*-
"""Canonical marginal multi-target probabilistic forecasting tests."""

import unittest

import numpy as np
import pandas as pd

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
from forecasting_core.tensors import (
    MarginalQuantileForecastTensor,
    PointForecastTensor,
)
from model_evaluation.marginal import evaluate_marginal_distribution
from model_forecasting.predictor import (
    CanonicalMarginalQuantileForecaster,
    repair_marginal_quantile_crossing,
)
from model_training.quantile import CanonicalMarginalQuantileTrainer
from forecasting_core.artifacts import (
    MarginalForecastDistribution,
    generate_joint_samples,
)


class MeanQuantileRegressor:
    def __init__(self, level):
        self.level = float(level)

    def fit(self, X, y):
        self.value = float(np.mean(y) + (self.level - 0.5) * 2.0)
        return self

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


class MarginalForecastDistributionTest(unittest.TestCase):
    def tensor(self):
        values = np.array(
            [
                [
                    [[12.0, 10.0, 9.0], [90.0, 100.0, 110.0]],
                    [[8.0, 11.0, 10.0], [120.0, 115.0, 114.0]],
                    [[10.0, 12.0, 14.0], [130.0, 125.0, 140.0]],
                ],
                [
                    [[20.0, 21.0, 22.0], [200.0, 201.0, 202.0]],
                    [[25.0, 24.0, 23.0], [210.0, 211.0, 209.0]],
                    [[30.0, 31.0, 32.0], [220.0, 221.0, 222.0]],
                ],
            ]
        )
        return MarginalQuantileForecastTensor(
            values=values,
            levels=(0.1, 0.5, 0.9),
            point_level=0.5,
            series_ids=("A", "B"),
            forecast_times=pd.date_range("2026-09-01", periods=3, freq="1h"),
            targets=("load", "power"),
        )

    def test_crossing_repair_is_independent_per_series_target_and_keeps_q50(self):
        raw = self.tensor()

        repaired = repair_marginal_quantile_crossing(raw)

        np.testing.assert_allclose(repaired.values[..., 1], raw.values[..., 1])
        self.assertTrue(np.all(np.diff(repaired.values, axis=-1) >= 0.0))
        distribution = MarginalForecastDistribution(
            point=repaired.point(),
            quantiles=repaired,
            dependence_model=None,
            metadata={"quantile_kind": "marginal"},
        )
        self.assertEqual(distribution.shape, (2, 3, 2, 3))
        self.assertIsNone(distribution.dependence_model)

    def test_point_must_equal_q50_and_joint_samples_are_explicitly_unsupported(self):
        repaired = repair_marginal_quantile_crossing(self.tensor())
        invalid_point = PointForecastTensor(
            values=repaired.point().values + 1.0,
            series_ids=repaired.series_ids,
            forecast_times=repaired.forecast_times,
            targets=repaired.targets,
        )
        with self.assertRaisesRegex(ValueError, "point.*point quantile"):
            MarginalForecastDistribution(
                point=invalid_point,
                quantiles=repaired,
                dependence_model=None,
            )
        with self.assertRaises(NotImplementedError):
            generate_joint_samples(repaired, n_samples=100)

    def test_metrics_are_reported_per_target_not_flattened_across_k(self):
        repaired = repair_marginal_quantile_crossing(self.tensor())
        distribution = MarginalForecastDistribution(
            point=repaired.point(),
            quantiles=repaired,
            dependence_model=None,
        )
        actual = PointForecastTensor(
            values=repaired.point().values + np.array([0.0, 10.0])[None, None, :],
            series_ids=repaired.series_ids,
            forecast_times=repaired.forecast_times,
            targets=repaired.targets,
        )

        report = evaluate_marginal_distribution(actual, distribution)

        self.assertEqual(set(report["target"]), {"load", "power", "__aggregate__"})
        target_rows = report[report["scope"] == "target"]
        self.assertEqual(set(target_rows["target"]), {"load", "power"})
        load_mae = report[(report["target"] == "load") & (report["metric"] == "mae")]
        power_mae = report[(report["target"] == "power") & (report["metric"] == "mae")]
        self.assertEqual(load_mae["value"].iloc[0], 0.0)
        self.assertEqual(power_mae["value"].iloc[0], 10.0)
        # aggregate 行为跨 target 池化（proper score：所有有效点等权）
        aggregate_mae = report[
            (report["scope"] == "aggregate") & (report["metric"] == "mae")
        ]
        self.assertEqual(len(aggregate_mae), 1)
        self.assertAlmostEqual(aggregate_mae["value"].iloc[0], 5.0)


class CanonicalMarginalQuantileTrainingTest(unittest.TestCase):
    capabilities = EstimatorCapabilities(
        scalar_target=True,
        scalar_quantile=True,
        native_multi_target_point=False,
        native_multi_target_quantile=False,
        sample_weight=False,
        categorical=False,
        nan_support=False,
    )

    @staticmethod
    def config():
        targets = ("load", "power")
        return ForecastConfigSpec(
            problem=ForecastProblemSpec(
                time_col="time",
                freq="1h",
                horizon=2,
                targets=targets,
                training_scope="local",
                series_id_cols=(),
            ),
            data=DataSpec(
                (
                    DataSourceSpec(
                        name="targets",
                        source_type="file",
                        columns=tuple(ColumnSpec(target, "target") for target in targets),
                        history_path="unused.csv",
                        time_col="time",
                        availability="source_time",
                    ),
                )
            ),
            features=FeatureSpec(
                target_lags={},
                observed_past_lags={},
                datetime_features=(),
                transformations={},
            ),
            strategy=ForecastStrategySpec("direct"),
            estimator=EstimatorSpec(
                model_type="mean_quantile",
                target_adapter="independent",
            ),
            probabilistic={
                "mode": "quantile",
                "quantiles": [0.1, 0.5, 0.9],
                "point_quantile": 0.5,
            },
            validation={},
            output={},
        )

    def test_quantile_training_and_forecasting_preserve_n_h_k_q(self):
        config = self.config()
        X_by_call = (
            np.arange(8.0).reshape(-1, 1),
            np.arange(8.0).reshape(-1, 1),
        )
        Y = np.arange(8.0 * 2 * 2).reshape(8, 2, 2)
        trainer = CanonicalMarginalQuantileTrainer(
            config,
            estimator_factory_for_level=lambda level: lambda: MeanQuantileRegressor(level),
            capabilities=self.capabilities,
            feature_schema=("x",),
        )
        artifact = trainer.train(X_by_call, Y, n_series=1)
        forecaster = CanonicalMarginalQuantileForecaster(config, artifact)
        X_predict = np.array([[1.0]])
        call_designs = (X_predict, X_predict)

        distribution = forecaster.predict(
            X_predict,
            series_ids=("__local__",),
            forecast_times=pd.date_range("2026-09-01", periods=2, freq="1h"),
            feature_provider=lambda call_index, *_: call_designs[call_index],
        )

        self.assertEqual(distribution.shape, (1, 2, 2, 3))
        np.testing.assert_allclose(
            distribution.point.values,
            distribution.quantiles.values[..., 1],
        )
        self.assertEqual(artifact.schema_version, 2)
        self.assertIsNone(artifact.dependence_model)


if __name__ == "__main__":
    unittest.main()
