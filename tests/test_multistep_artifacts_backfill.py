# -*- coding: utf-8 -*-

from types import SimpleNamespace
import unittest
from collections import deque

import numpy as np
import pandas as pd

from models.AuxiliaryForecaster import AuxiliaryEndogenousForecaster
from models.multistep.artifacts import MultistepArtifactMetadata
from models.multistep.backfill import (
    AuxiliaryProvider,
    KnownFutureProvider,
    PersistenceProvider,
)
from models.multistep.resolve import resolve_strategy
from models.multistep.state import RecursiveFeatureCache
from models.multistep.weights import BlendWeights


class BlendWeightsContractTest(unittest.TestCase):
    def test_ridge_weights_must_be_resolved_before_training(self):
        args = SimpleNamespace(blend_weight_strategy="ridge_stacking")
        with self.assertRaisesRegex(ValueError, "resolved_blend_weights"):
            BlendWeights.from_args(args)

    def test_combine_requires_two_complete_horizon_outputs(self):
        weights = BlendWeights(0.25, 0.75, strategy="fixed")
        with self.assertRaisesRegex(ValueError, "blend direct prediction length mismatch"):
            weights.combine([1.0], [2.0, 3.0], horizon=2)
        np.testing.assert_allclose(
            weights.combine([4.0, 8.0], [0.0, 4.0], horizon=2),
            [1.0, 5.0],
        )


class EndogenousProviderContractTest(unittest.TestCase):
    def test_known_future_precedes_configured_persistence(self):
        fallback = PersistenceProvider(pd.DataFrame({"x": [1.0, 2.0]}), ["x"])
        provider = KnownFutureProvider(
            pd.DataFrame({"x": [10.0, np.nan]}),
            fallback,
            ["x"],
            horizon=2,
        )
        self.assertEqual(provider.value_at("x", 0).source, "known_future")
        self.assertEqual(provider.value_at("x", 0).value, 10.0)
        self.assertEqual(provider.value_at("x", 1).source, "persistence")
        self.assertEqual(provider.value_at("x", 1).value, 2.0)

    def test_auxiliary_requires_every_feature_and_exact_horizon(self):
        with self.assertRaisesRegex(ValueError, "missing feature 'z'"):
            AuxiliaryProvider({"x": [1.0, 2.0]}, ["x", "z"], horizon=2)
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            AuxiliaryProvider({"x": [1.0]}, ["x"], horizon=2)

    def test_persistence_requires_finite_last_observation(self):
        with self.assertRaisesRegex(ValueError, "finite last value"):
            PersistenceProvider(pd.DataFrame({"x": [np.nan]}), ["x"])

    def test_auxiliary_forecaster_does_not_degrade_without_lags(self):
        forecaster = AuxiliaryEndogenousForecaster(
            SimpleNamespace(lags=[], datetime_features=[]),
            ["x"],
            "y",
        )
        history = pd.DataFrame(
            {"time": pd.date_range("2026-01-01", periods=3, freq="1h"), "x": [1, 2, 3]}
        )
        with self.assertRaisesRegex(ValueError, "requires at least one positive lag"):
            forecaster.fit(history)

    def test_auxiliary_forecaster_requires_a_model_for_every_column(self):
        forecaster = AuxiliaryEndogenousForecaster(
            SimpleNamespace(lags=[1], datetime_features=[]),
            ["x"],
            "y",
        )
        history = pd.DataFrame(
            {"time": pd.date_range("2026-01-01", periods=3, freq="1h"), "x": [1, 2, 3]}
        )
        future = pd.DataFrame(
            {"time": pd.date_range("2026-01-01 03:00", periods=2, freq="1h")}
        )
        with self.assertRaisesRegex(ValueError, "no fitted model"):
            forecaster.predict_horizon(history, future, horizon=2)


class MultistepArtifactMetadataTest(unittest.TestCase):
    def test_metadata_records_effective_dirrec_width(self):
        args = SimpleNamespace(
            pred_method="multivariate-single-multistep-direct-recursive",
            lags=[2, 8],
            block_size=0,
            direct_strategy="multioutput",
        )
        resolved = resolve_strategy(args, horizon=5)
        metadata = MultistepArtifactMetadata.from_strategy(resolved, ["x", "y_lag_2"])
        self.assertEqual(metadata.method_code, "msmdr")
        self.assertEqual(metadata.target_steps, (1, 2))
        self.assertEqual(metadata.model_output_width, 2)
        self.assertEqual(metadata.feature_schema, ("x", "y_lag_2"))


class RecursiveFeatureCacheTest(unittest.TestCase):
    def test_cached_step_matches_direct_feature_construction(self):
        cache = RecursiveFeatureCache(
            df_future_exog=pd.DataFrame({"temperature": [20.0, 21.0]}),
            exogenous_features=("temperature",),
            predictor_features=("temperature", "y_lag_1", "y_lag_2"),
            categorical_features=(),
            target_output_features=("y_shift_0",),
            lag_feature_names=frozenset({"y_lag_1", "y_lag_2"}),
            lag_state={"y": deque([5.0, 6.0], maxlen=2)},
            lags=(1, 2),
        )
        expected = pd.DataFrame(
            [{"temperature": 20.0, "y_lag_1": 6.0, "y_lag_2": 5.0}]
        )
        pd.testing.assert_frame_equal(cache.build_step_input(0), expected)
        self.assertEqual(cache.schema_build_count, 1)

        cache.append_endogenous("y", 7.0)
        expected_next = pd.DataFrame(
            [{"temperature": 21.0, "y_lag_1": 7.0, "y_lag_2": 6.0}]
        )
        pd.testing.assert_frame_equal(cache.build_step_input(1), expected_next)
        self.assertEqual(cache.schema_build_count, 1)

    def test_cache_never_synthesizes_a_missing_predictor(self):
        cache = RecursiveFeatureCache(
            df_future_exog=pd.DataFrame(index=range(1)),
            exogenous_features=(),
            predictor_features=("missing",),
            categorical_features=(),
            target_output_features=("y_shift_0",),
            lag_feature_names=frozenset(),
            lag_state={"y": deque([1.0], maxlen=1)},
            lags=(1,),
        )
        with self.assertRaisesRegex(ValueError, "could not construct features"):
            cache.build_step_input(0)


if __name__ == "__main__":
    unittest.main()
