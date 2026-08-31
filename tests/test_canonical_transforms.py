# -*- coding: utf-8 -*-
"""Canonical Task19 target/feature transformation contracts."""

import pickle
import unittest

import numpy as np
import pandas as pd

from feature_engineering import FeatureCompiler
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
from model_forecasting.transforms import CanonicalFeatureScaler, CanonicalTargetTransform


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
            information_mode="forecast",
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
