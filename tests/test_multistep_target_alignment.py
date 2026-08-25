#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from features.FeatureEngineering import FeatureEngineer
from models.ModelForecasting import Forecaster
from models.multistep.artifacts import LegacyArtifactAdapter
from models.multistep.executors.direct import DirectExecutor
from probabilistic.spec import ProbabilisticSpec
from probabilistic.types import ProbabilisticModelBundle


class MultistepTargetAlignmentTest(unittest.TestCase):
    def _build_features(self, pred_method: str):
        args = SimpleNamespace(
            pred_method=pred_method,
            enable_lags_features=False,
            lags=[],
            align_direct_features_to_target=False,
        )
        df = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=6, freq="1h"),
                "covariate": [100, 101, 102, 103, 104, 105],
                "y": [10, 11, 12, 13, 14, 15],
            }
        )
        return FeatureEngineer(args, verbose=False).create_endogenous_basic_features(
            df_series=df,
            target_feature="y",
            endogenous_features_with_target=["covariate", "y"],
            horizon=3,
        )

    def test_multivariate_direct_starts_at_next_time_step(self):
        featured, _predictors, targets = self._build_features(
            "multivariate-single-multistep-direct"
        )

        self.assertEqual(targets, ["y_shift_1", "y_shift_2", "y_shift_3"])
        self.assertEqual(featured.iloc[0][targets].tolist(), [11, 12, 13])

    def test_multivariate_dirrec_without_lags_uses_one_step_block(self):
        featured, _predictors, targets = self._build_features(
            "multivariate-single-multistep-direct-recursive"
        )

        self.assertEqual(targets, ["y_shift_1"])
        self.assertEqual(featured.iloc[0][targets].tolist(), [11])

    def test_univariate_and_multivariate_direct_targets_are_symmetric(self):
        method_pairs = (
            (
                "univariate-single-multistep-direct",
                "multivariate-single-multistep-direct",
            ),
            (
                "univariate-single-multistep-direct-recursive",
                "multivariate-single-multistep-direct-recursive",
            ),
        )
        for univariate_method, multivariate_method in method_pairs:
            with self.subTest(
                univariate_method=univariate_method,
                multivariate_method=multivariate_method,
            ):
                _df_u, _predictors_u, targets_u = self._build_features(univariate_method)
                _df_m, _predictors_m, targets_m = self._build_features(multivariate_method)

                self.assertEqual(targets_m, targets_u)

    def test_multivariate_recursive_keeps_current_time_target(self):
        featured, _predictors, targets = self._build_features(
            "multivariate-single-multistep-recursive"
        )

        self.assertEqual(targets, ["y_shift_0"])
        self.assertEqual(featured.iloc[0][targets].tolist(), [10])


class DirectPredictionLengthContractTest(unittest.TestCase):
    class _Model:
        def __init__(self, prediction):
            self.prediction = np.asarray(prediction)

        def predict(self, _X):
            return self.prediction

    @staticmethod
    def _forecaster(model):
        forecaster = Forecaster.__new__(Forecaster)
        forecaster.model = model
        forecaster.df_future = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=3, freq="1h")})
        forecaster.endogenous_features = []
        forecaster._quantile_outputs = None
        forecaster.log_prefix = "[test]"
        forecaster._build_direct_forecast_input = lambda endogenous_features: (
            pd.DataFrame({"x": [1.0]}),
            [],
        )
        forecaster._transform_features = lambda X, categorical_features: X
        return forecaster

    def test_point_direct_rejects_short_output(self):
        forecaster = self._forecaster(self._Model([[1.0, 2.0]]))

        with self.assertRaisesRegex(
            ValueError,
            r"point direct prediction length mismatch: expected 3, got 2",
        ):
            DirectExecutor().execute(forecaster)

    def test_point_direct_rejects_long_output(self):
        forecaster = self._forecaster(self._Model([[1.0, 2.0, 3.0, 4.0]]))

        with self.assertRaisesRegex(
            ValueError,
            r"point direct prediction length mismatch: expected 3, got 4",
        ):
            DirectExecutor().execute(forecaster)

    def test_quantile_direct_rejects_short_output(self):
        bundle = {
            "predict_type": "quantile",
            "quantiles": [0.1, 0.5, 0.9],
            "median_quantile": 0.5,
            "models": {
                0.1: self._Model([[1.0, 2.0]]),
                0.5: self._Model([[1.0, 2.0, 3.0]]),
                0.9: self._Model([[2.0, 3.0, 4.0]]),
            },
        }
        forecaster = self._forecaster(LegacyArtifactAdapter.adapt(bundle))

        with self.assertRaisesRegex(
            ValueError,
            r"quantile q=0.1 direct prediction length mismatch: expected 3, got 2",
        ):
            DirectExecutor().execute(forecaster)

    def test_typed_quantile_bundle_drives_direct_prediction(self):
        spec = ProbabilisticSpec(
            mode="quantile",
            quantiles=(0.1, 0.5, 0.9),
            point_quantile=0.5,
            recursive_propagation="median_path",
            crossing_method="none",
            crossing_report_raw=True,
            intervals=(),
            calibration=None,
        )
        bundle = ProbabilisticModelBundle(
            schema_version=1,
            spec=spec,
            model_type="lightgbm",
            pred_method="univariate-single-multistep-direct",
            models_by_quantile={
                0.1: self._Model([[1.0, 2.0, 3.0]]),
                0.5: self._Model([[2.0, 3.0, 4.0]]),
                0.9: self._Model([[3.0, 4.0, 5.0]]),
            },
            recursive_propagation="median_path",
        )
        forecaster = self._forecaster(bundle)

        prediction = DirectExecutor().execute(forecaster)

        np.testing.assert_array_equal(prediction, [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(
            forecaster.quantile_outputs[0.1],
            [1.0, 2.0, 3.0],
        )


if __name__ == "__main__":
    unittest.main()
