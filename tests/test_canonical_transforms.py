# -*- coding: utf-8 -*-
"""Canonical Task19 target/feature transformation contracts."""

import pickle
import unittest
from typing import Any, cast

import numpy as np
import pandas as pd

from feature_engineering import FeatureCompiler
from forecasting_core.artifacts import MarginalForecastDistribution
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
from forecasting_core.tensors import MarginalQuantileForecastTensor, PointForecastTensor
from model_pipeline.fold_fit import _fit_runtime_transforms
from feature_engineering.transforms import CanonicalFeatureScaler, CanonicalTargetTransform


def _config(*, target_transform=None, feature_scaling=None):
    transformations = {
        "feature_scaling": feature_scaling
        or {"method": "none", "grouped": False, "encode_categorical": False},
        "target": target_transform
        or {
            "calendar_normalization": {"method": "none"},
            "decomposition": {"method": "none"},
            "scaling": {"method": "none", "inverse": False},
        },
    }
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
            target_lags={"load": (2,), "power": (2,)},
            observed_past_lags={},
            datetime_features=(),
            transformations=transformations,
        ),
        strategy=ForecastStrategySpec("direct"),
        estimator=EstimatorSpec(model_type="ridge", target_adapter="independent"),
        probabilistic={"mode": "point"},
        validation={},
        output={},
    )


class CanonicalTargetTransformTest(unittest.TestCase):
    @staticmethod
    def _history():
        times = pd.date_range("2026-01-01", periods=48, freq="1h")
        step = np.arange(48, dtype=float)
        values = np.empty((2, 48, 2), dtype=float)
        values[0, :, 0] = 10.0 + step * 0.5
        values[0, :, 1] = 1_000.0 + step**2 * 3.0
        values[1, :, 0] = 20_000.0 - step * 7.0
        values[1, :, 1] = 2_000_000.0 + np.sin(step * np.pi / 2.0) * 50_000.0
        return PointForecastTensor(
            values=values,
            series_ids=("A", "B"),
            forecast_times=times,
            targets=("load", "power"),
        )

    def test_identity_transform_reuses_tensor_and_preserves_fitted_state(self):
        history = self._history()
        transform = CanonicalTargetTransform.from_config(_config())

        transformed = transform.fit_transform(history)
        restored = transform.restore_point(transformed)

        self.assertIs(transformed, history)
        self.assertIs(restored, history)
        expected_keys = {
            ("A", "load"),
            ("A", "power"),
            ("B", "load"),
            ("B", "power"),
        }
        self.assertEqual(set(transform.fitted_keys), expected_keys)
        self.assertEqual(
            transform.training_steps,
            {key: () for key in expected_keys},
        )

    def test_identity_transform_reuses_training_array(self):
        history = self._history()
        transform = CanonicalTargetTransform.from_config(_config())
        transform.fit_transform(history)
        values = np.arange(24.0).reshape(3, 4, 2)
        origins = tuple(pd.date_range("2026-03-01", periods=3, freq="1h"))

        transformed = transform.transform_training(
            values,
            origins,
            series_ids=("A", "A", "A"),
        )

        self.assertIs(transformed, values)

    def test_identity_runtime_does_not_materialize_target_history(self):
        class IdentityBuilder:
            feature_schema = ("lag",)
            categorical_schema = ()
            series_ids = ("A",)

            @staticmethod
            def target_history(_cutoff):
                raise AssertionError("identity transform must not load target history")

        config = _config()
        X_by_call = (np.arange(3.0).reshape(3, 1),)
        Y = np.arange(12.0).reshape(3, 2, 2)
        origins = tuple(pd.date_range("2026-03-01", periods=3, freq="1h"))

        _, target_transform, transformed_X, transformed_Y = _fit_runtime_transforms(
            config,
            cast(Any, IdentityBuilder()),
            X_by_call,
            Y,
            origins,
            ("A", "A", "A"),
            cast(pd.Timestamp, pd.Timestamp("2026-03-03")),
        )

        self.assertIs(transformed_X[0], X_by_call[0])
        self.assertIs(transformed_Y, Y)
        self.assertEqual(
            target_transform.fitted_keys,
            (("A", "load"), ("A", "power")),
        )

    def test_identity_restore_values_reuses_recursive_prediction_array(self):
        history = self._history()
        transform = CanonicalTargetTransform.from_config(_config())
        transform.fit_transform(history)
        values = np.array([11.0, 12.0])

        restored = transform.restore_values(
            "A",
            "load",
            values,
            history.forecast_times[:2],
        )

        self.assertIs(restored, values)

    def test_identity_transform_point_reuses_tensor(self):
        history = self._history()
        transform = CanonicalTargetTransform.from_config(_config())
        transform.fit_transform(history)

        transformed = transform.transform_point(history)

        self.assertIs(transformed, history)

    def test_identity_restore_quantiles_reuses_tensor(self):
        history = self._history()
        transform = CanonicalTargetTransform.from_config(_config())
        transform.fit_transform(history)
        quantiles = MarginalQuantileForecastTensor(
            values=np.stack(
                [history.values - 1.0, history.values, history.values + 1.0],
                axis=-1,
            ),
            levels=(0.1, 0.5, 0.9),
            point_level=0.5,
            series_ids=history.series_ids,
            forecast_times=history.forecast_times,
            targets=history.targets,
        )

        restored = transform.restore_quantiles(quantiles)

        self.assertIs(restored, quantiles)

    def test_identity_restore_distribution_reuses_distribution(self):
        history = self._history()
        transform = CanonicalTargetTransform.from_config(_config())
        transform.fit_transform(history)
        quantiles = MarginalQuantileForecastTensor(
            values=np.stack(
                [history.values - 1.0, history.values, history.values + 1.0],
                axis=-1,
            ),
            levels=(0.1, 0.5, 0.9),
            point_level=0.5,
            series_ids=history.series_ids,
            forecast_times=history.forecast_times,
            targets=history.targets,
        )
        distribution = MarginalForecastDistribution(
            point=history,
            quantiles=quantiles,
        )

        restored = transform.restore_distribution(distribution)

        self.assertIs(restored, distribution)

    def test_n2_k2_active_decomposition_and_scaling_methods_roundtrip_exactly(self):
        decomposition_specs = {
            "none": {"method": "none"},
            "linear": {"method": "linear"},
            "quadratic": {"method": "quadratic"},
            "damped": {"method": "damped", "trend_lookback": 12, "damping": 0.95},
            "stl": {"method": "stl", "periods": [4], "seasonal_cycles": 4},
            "mstl": {"method": "mstl", "periods": [4, 12], "seasonal_cycles": 3},
        }
        history = self._history()
        for decomposition_name, decomposition in decomposition_specs.items():
            for scaling in ("none", "minmax", "standard", "robust"):
                with self.subTest(decomposition=decomposition_name, scaling=scaling):
                    config = _config(
                        target_transform={
                            "calendar_normalization": {"method": "none"},
                            "decomposition": decomposition,
                            "scaling": {"method": scaling, "inverse": scaling != "none"},
                        }
                    )
                    transform = CanonicalTargetTransform.from_config(config)

                    transformed = transform.fit_transform(history)
                    restored = transform.restore_point(transformed)

                    np.testing.assert_allclose(restored.values, history.values, atol=1e-7)
                    self.assertEqual(
                        set(transform.fitted_keys),
                        {("A", "load"), ("A", "power"), ("B", "load"), ("B", "power")},
                    )

    def test_point_and_all_quantiles_share_each_series_target_state(self):
        config = _config(
            target_transform={
                "calendar_normalization": {"method": "none"},
                "decomposition": {"method": "linear"},
                "scaling": {"method": "standard", "inverse": True},
            }
        )
        history = self._history()
        transform = CanonicalTargetTransform.from_config(config)
        transformed = transform.fit_transform(history)
        quantiles = MarginalQuantileForecastTensor(
            values=np.stack(
                [transformed.values - 0.25, transformed.values, transformed.values + 0.25],
                axis=-1,
            ),
            levels=(0.1, 0.5, 0.9),
            point_level=0.5,
            series_ids=history.series_ids,
            forecast_times=history.forecast_times,
            targets=history.targets,
        )

        restored = transform.restore_quantiles(quantiles)

        np.testing.assert_allclose(restored.values[..., 1], history.values, atol=1e-8)
        self.assertFalse(np.allclose(restored.values[0, :, 0, 0], restored.values[1, :, 0, 0]))
        self.assertFalse(np.allclose(restored.values[0, :, 0, 0], restored.values[0, :, 1, 0]))

    def test_training_order_and_k1_pickle_roundtrip_match_legacy_semantics(self):
        config = _config(
            target_transform={
                "calendar_normalization": {"method": "per_calendar_day"},
                "decomposition": {"method": "linear"},
                "scaling": {"method": "minmax", "inverse": True},
            }
        )
        full = self._history()
        history = PointForecastTensor(
            values=full.values[:1, :, :1],
            series_ids=("__local__",),
            forecast_times=full.forecast_times,
            targets=("load",),
        )
        transform = CanonicalTargetTransform.from_config(config)

        transformed = transform.fit_transform(history)
        restored_transform = pickle.loads(pickle.dumps(transform))
        restored = restored_transform.restore_point(transformed)

        self.assertEqual(
            restored_transform.training_steps[("__local__", "load")],
            ("calendar_normalization", "decomposition", "target_scaling"),
        )
        np.testing.assert_allclose(restored.values, history.values, atol=1e-8)


class CanonicalFeatureScalerTest(unittest.TestCase):
    def test_local_grouped_scaling_and_categorical_encoding_fit_once(self):
        config = _config(
            feature_scaling={
                "method": "standard",
                "grouped": True,
                "encode_categorical": True,
            }
        )
        scaler = CanonicalFeatureScaler.from_config(
            config,
            feature_names=("load__lag_1", "dt_hour", "site"),
            categorical_names=("site",),
        )
        training = pd.DataFrame(
            {
                "load__lag_1": [10.0, 20.0, 30.0],
                "dt_hour": [0.0, 1.0, 2.0],
                "site": ["A", "B", "A"],
            }
        )

        transformed = scaler.fit_transform(training)
        holdout = scaler.transform(training.iloc[[2]])
        restored_scaler = pickle.loads(pickle.dumps(scaler))

        self.assertEqual(transformed.shape, (3, 3))
        self.assertTrue(np.isfinite(transformed).all())
        np.testing.assert_allclose(holdout, restored_scaler.transform(training.iloc[[2]]))
        self.assertEqual(scaler.feature_groups["categorical"], ("site",))

    def test_no_scaling_numeric_calls_use_ndarray_fast_path(self):
        config = _config(
            feature_scaling={
                "method": "none",
                "grouped": False,
                "encode_categorical": False,
            }
        )
        scaler = CanonicalFeatureScaler.from_config(
            config,
            feature_names=("load__lag_1", "dt_hour"),
        )
        first = np.arange(12.0).reshape(6, 2)
        second = first + 100.0

        transformed = scaler.fit_transform_calls((first, second))
        holdout = scaler.transform(first)

        self.assertIs(transformed[0], first)
        self.assertIs(transformed[1], second)
        self.assertIs(holdout, first)
        self.assertTrue(scaler.is_fitted)

    def test_feature_compiler_validates_but_does_not_consume_runtime_transforms(self):
        config = _config(
            target_transform={
                "calendar_normalization": {"method": "none"},
                "decomposition": {"method": "linear"},
                "scaling": {"method": "robust", "inverse": True},
            },
            feature_scaling={
                "method": "minmax",
                "grouped": False,
                "encode_categorical": False,
            },
        )

        compiler = FeatureCompiler(config)

        self.assertIsInstance(compiler, FeatureCompiler)


if __name__ == "__main__":
    unittest.main()
