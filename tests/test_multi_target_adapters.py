import ast
import json
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from model_training.estimators.capabilities import (
    CapabilityRegistry,
    EstimatorCapabilities,
    MODEL_FACTORY_CAPABILITY_REGISTRY,
    ProbeResult,
    make_model_factory,
    probe_native_multioutput,
    resolve_model_capabilities,
)
from model_training.estimators.multi_target import (
    IndependentMultiTargetAdapter,
    NativeMultiTargetAdapter,
    RegressorChainMultiTargetAdapter,
)
from model_forecasting.specs import EstimatorSpec, TargetAdapter
from model_training.strategies import StrategyTargetPlan, TargetCoordinate
from model_forecasting.specs import ForecastStrategySpec
from models.ModelFactory import ModelFactory


class ScalarOnlyRegressor:
    def fit(self, X, y):
        if np.asarray(y).ndim != 1:
            raise ValueError("scalar target required")
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


class WrongShapeRegressor:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


class TrackingNativeRegressor:
    def __init__(self):
        self.was_fit = False

    def fit(self, X, y):
        self.was_fit = True
        return self

    def predict(self, X):
        return np.zeros((len(X), 2), dtype=float)


class RecordingScalarEstimator:
    def __init__(self, prediction_value=0.0):
        self.prediction_value = float(prediction_value)
        self.fit_X = None
        self.fit_y = None
        self.fit_sample_weight = None
        self.predict_X = None

    def fit(self, X, y, **kwargs):
        self.fit_X = np.asarray(X, dtype=float).copy()
        self.fit_y = np.asarray(y, dtype=float).copy()
        self.fit_sample_weight = kwargs.get("sample_weight")
        return self

    def predict(self, X):
        self.predict_X = np.asarray(X, dtype=float).copy()
        return np.full(len(X), self.prediction_value, dtype=float)


class RecordingNativeEstimator:
    def __init__(self):
        self.fit_X = None
        self.fit_y = None
        self.fit_sample_weight = None
        self.predict_X = None

    def fit(self, X, y, **kwargs):
        self.fit_X = np.asarray(X, dtype=float).copy()
        self.fit_y = np.asarray(y, dtype=float).copy()
        self.fit_sample_weight = kwargs.get("sample_weight")
        return self

    def predict(self, X):
        self.predict_X = np.asarray(X, dtype=float).copy()
        return np.tile(self.fit_y[0], (len(X), 1))


class EstimatorSpecTests(unittest.TestCase):
    def test_target_adapter_values_are_exact(self):
        self.assertEqual(
            [adapter.value for adapter in TargetAdapter],
            ["independent", "regressor_chain", "native"],
        )

    def test_model_type_is_exact_nonblank_without_surrounding_whitespace(self):
        for invalid in (None, 1, "", " ", " ridge", "ridge "):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    EstimatorSpec(invalid, "native")

        spec = EstimatorSpec("Ridge", "native")
        self.assertEqual(spec.model_type, "Ridge")

    def test_adapter_is_normalized_and_unknown_values_raise(self):
        self.assertIs(
            EstimatorSpec("ridge", "independent").target_adapter,
            TargetAdapter.INDEPENDENT,
        )
        self.assertIs(
            EstimatorSpec("ridge", TargetAdapter.NATIVE).target_adapter,
            TargetAdapter.NATIVE,
        )
        with self.assertRaises(ValueError):
            EstimatorSpec("ridge", "fallback")
        with self.assertRaises(TypeError):
            EstimatorSpec("ridge", None)

    def test_params_are_deeply_isolated_immutable_and_canonical(self):
        params = {
            "z": [3, {"b": True, "a": None}],
            "a": {"nested": [2.0, "x"]},
        }
        spec = EstimatorSpec("ridge", "native", params)
        expected = {
            "model_type": "ridge",
            "target_adapter": "native",
            "params": {
                "a": {"nested": [2.0, "x"]},
                "z": [3, {"a": None, "b": True}],
            },
        }

        params["z"][1]["b"] = False
        params["a"]["nested"].append("changed")

        first = spec.canonical_payload()
        second = spec.canonical_payload()
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertEqual(json.loads(json.dumps(first)), expected)
        self.assertIsNot(first, second)
        self.assertIsNot(first["params"], second["params"])

        with self.assertRaises(TypeError):
            spec.params["new"] = 1
        with self.assertRaises((AttributeError, TypeError)):
            spec.params["z"].append(4)
        with self.assertRaises(FrozenInstanceError):
            spec.model_type = "other"

    def test_params_reject_non_json_like_values(self):
        invalid_values = (
            {"bad": object()},
            {1: "non-string key"},
            {"bad": {1, 2}},
            {"bad": float("nan")},
            {"bad": float("inf")},
        )
        for params in invalid_values:
            with self.subTest(params=params):
                with self.assertRaises((TypeError, ValueError)):
                    EstimatorSpec("ridge", "native", params)

    def test_spec_contains_no_forecasting_problem_fields(self):
        spec = EstimatorSpec("ridge", "native")
        for forbidden in ("horizon", "strategy", "problem"):
            self.assertFalse(hasattr(spec, forbidden))

    def test_capability_validation_has_no_fallback(self):
        scalar_only = EstimatorCapabilities(
            scalar_target=True,
            scalar_quantile=False,
            native_multi_target_point=False,
            native_multi_target_quantile=False,
            sample_weight=True,
            categorical=False,
            nan_support=False,
        )
        native_point = EstimatorCapabilities(
            scalar_target=False,
            scalar_quantile=False,
            native_multi_target_point=True,
            native_multi_target_quantile=False,
            sample_weight=False,
            categorical=False,
            nan_support=True,
        )
        native_quantile = EstimatorCapabilities(
            scalar_target=False,
            scalar_quantile=False,
            native_multi_target_point=False,
            native_multi_target_quantile=True,
            sample_weight=False,
            categorical=True,
            nan_support=True,
        )

        independent = EstimatorSpec("ridge", "independent")
        chain = EstimatorSpec("ridge", "regressor_chain")
        native = EstimatorSpec("ridge", "native")

        self.assertIs(independent.validate_capabilities(scalar_only), independent)
        self.assertIs(native.validate_capabilities(native_point, "point"), native)
        self.assertIs(native.validate_capabilities(native_quantile, "quantile"), native)

        with self.assertRaises(ValueError):
            independent.validate_capabilities(native_point)
        with self.assertRaises(ValueError):
            chain.validate_capabilities(native_point, "quantile")
        with self.assertRaises(ValueError):
            independent.validate_capabilities(scalar_only, "quantile")
        with self.assertRaises(ValueError):
            native.validate_capabilities(scalar_only, "point")
        with self.assertRaises(ValueError):
            native.validate_capabilities(native_point, "quantile")
        with self.assertRaises(ValueError):
            native.validate_capabilities(native_quantile, "distribution")
        with self.assertRaises(TypeError):
            native.validate_capabilities(object())


class EstimatorCapabilitiesTests(unittest.TestCase):
    def test_capabilities_are_immutable_explicit_and_deterministic(self):
        capabilities = EstimatorCapabilities(
            scalar_target=True,
            scalar_quantile=True,
            native_multi_target_point=False,
            native_multi_target_quantile=True,
            sample_weight=True,
            categorical=False,
            nan_support=True,
        )
        expected = {
            "scalar_target": True,
            "scalar_quantile": True,
            "native_multi_target_point": False,
            "native_multi_target_quantile": True,
            "sample_weight": True,
            "categorical": False,
            "nan_support": True,
        }
        self.assertEqual(capabilities.canonical_payload(), expected)
        self.assertEqual(capabilities.canonical_payload(), expected)
        with self.assertRaises(FrozenInstanceError):
            capabilities.scalar_target = False

        with self.assertRaises(TypeError):
            EstimatorCapabilities(
                scalar_target=1,
                scalar_quantile=False,
                native_multi_target_point=False,
                native_multi_target_quantile=False,
                sample_weight=False,
                categorical=False,
                nan_support=False,
            )


class CapabilityRegistryTests(unittest.TestCase):
    def setUp(self):
        self.scalar = EstimatorCapabilities(
            scalar_target=True,
            scalar_quantile=False,
            native_multi_target_point=False,
            native_multi_target_quantile=False,
            sample_weight=True,
            categorical=False,
            nan_support=False,
        )
        self.native = EstimatorCapabilities(
            scalar_target=True,
            scalar_quantile=False,
            native_multi_target_point=True,
            native_multi_target_quantile=False,
            sample_weight=True,
            categorical=False,
            nan_support=True,
        )

    def test_registry_normalizes_names_and_is_immutable(self):
        source = {" Ridge ": self.native, "LGB": self.scalar}
        registry = CapabilityRegistry(source)
        source["new"] = self.scalar

        self.assertIs(registry.lookup("ridge"), self.native)
        self.assertIs(registry.lookup(" RIDGE "), self.native)
        self.assertIs(registry.lookup("lgb"), self.scalar)
        with self.assertRaises(KeyError):
            registry.lookup("new")
        with self.assertRaises(TypeError):
            registry.capabilities["other"] = self.scalar

    def test_duplicate_normalized_registration_is_forbidden(self):
        with self.assertRaises(ValueError):
            CapabilityRegistry([("Ridge", self.scalar), (" ridge ", self.native)])

    def test_unknown_lookup_raises_and_payload_is_deterministic(self):
        registry = CapabilityRegistry([("z_model", self.scalar), ("a_model", self.native)])
        expected = {
            "a_model": self.native.canonical_payload(),
            "z_model": self.scalar.canonical_payload(),
        }
        self.assertEqual(registry.canonical_payload(), expected)
        self.assertEqual(registry.canonical_payload(), expected)
        with self.assertRaises(KeyError):
            registry.lookup("missing")
        with self.assertRaises((TypeError, ValueError)):
            registry.lookup(" ")

    def test_production_registry_covers_every_model_factory_type(self):
        self.assertEqual(
            set(MODEL_FACTORY_CAPABILITY_REGISTRY.capabilities),
            set(ModelFactory._models),
        )
        ridge = make_model_factory(
            "ridge",
            {"alpha": 1e-6},
            feature_names=("x0", "x1"),
        )()
        ridge.fit(np.arange(12.0).reshape(6, 2), np.arange(6.0))
        self.assertEqual(ridge.predict(np.ones((2, 2))).shape, (2,))

    def test_native_capability_is_enabled_only_after_behavioral_probe(self):
        self.assertFalse(
            MODEL_FACTORY_CAPABILITY_REGISTRY.lookup(
                "ridge"
            ).native_multi_target_point
        )
        self.assertFalse(
            resolve_model_capabilities("ridge", probe_native=True)
            .native_multi_target_point
        )
        self.assertTrue(
            resolve_model_capabilities(
                "randomforest",
                {"n_estimators": 5, "random_state": 0, "n_jobs": 1},
                probe_native=True,
            )
            .native_multi_target_point
        )

    def test_native_capability_probe_is_independent_of_runtime_feature_width(self):
        capabilities = resolve_model_capabilities(
            "randomforest",
            {"n_estimators": 5, "random_state": 0, "n_jobs": 1},
            feature_names=("lag_1", "lag_2", "lag_3", "dt_hour"),
            probe_native=True,
        )

        self.assertTrue(capabilities.native_multi_target_point)

    def test_quantile_factory_rejects_models_without_declared_support(self):
        with self.assertRaisesRegex(ValueError, "scalar quantiles"):
            make_model_factory("ridge", quantile=0.5)
        quantile = make_model_factory(
            "quantileregressor",
            {"alpha": 1e-3},
            feature_names=("x",),
            quantile=0.9,
        )()
        self.assertEqual(quantile.model.params["quantile"], 0.9)


class NativeMultioutputProbeTests(unittest.TestCase):
    def test_ridge_and_random_forest_pass_behavioral_probe(self):
        for factory in (
            Ridge,
            lambda: RandomForestRegressor(n_estimators=5, random_state=0, n_jobs=1),
        ):
            with self.subTest(factory=factory):
                result = probe_native_multioutput(factory)
                self.assertIsInstance(result, ProbeResult)
                self.assertTrue(result.supported, result.reason)
                self.assertIsNone(result.reason)

    def test_scalar_only_fit_failure_is_captured(self):
        result = probe_native_multioutput(ScalarOnlyRegressor)
        self.assertFalse(result.supported)
        self.assertIn("fit failed", result.reason)
        self.assertIn("scalar target required", result.reason)

    def test_wrong_prediction_shape_is_captured(self):
        result = probe_native_multioutput(WrongShapeRegressor)
        self.assertFalse(result.supported)
        self.assertIn("prediction shape", result.reason)
        self.assertIn("(6, 2)", result.reason)

    def test_invalid_factory_raises_but_factory_failures_are_captured(self):
        with self.assertRaises(TypeError):
            probe_native_multioutput(None)

        result = probe_native_multioutput(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        self.assertFalse(result.supported)
        self.assertIn("factory failed", result.reason)
        self.assertIn("boom", result.reason)

    def test_probe_does_not_mutate_factory_estimator(self):
        estimator = TrackingNativeRegressor()
        result = probe_native_multioutput(lambda: estimator)
        self.assertTrue(result.supported, result.reason)
        self.assertFalse(estimator.was_fit)


class MultiTargetAdapterTests(unittest.TestCase):
    def setUp(self):
        self.coordinates = StrategyTargetPlan.from_spec(
            ForecastStrategySpec("mimo"),
            ("load", "temperature"),
            horizon=4,
        ).coordinates
        self.X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        self.Y = np.arange(24, dtype=float).reshape(3, 4, 2)
        self.sample_weight = np.array([1.0, 2.0, 3.0])
        self.scalar_capabilities = EstimatorCapabilities(
            scalar_target=True,
            scalar_quantile=True,
            native_multi_target_point=False,
            native_multi_target_quantile=False,
            sample_weight=True,
            categorical=False,
            nan_support=False,
        )
        self.native_capabilities = EstimatorCapabilities(
            scalar_target=False,
            scalar_quantile=False,
            native_multi_target_point=True,
            native_multi_target_quantile=True,
            sample_weight=True,
            categorical=False,
            nan_support=False,
        )

    def test_independent_uses_time_major_columns_and_restores_n_h_k(self):
        created = []

        def factory():
            estimator = RecordingScalarEstimator(len(created))
            created.append(estimator)
            return estimator

        adapter = IndependentMultiTargetAdapter(
            factory,
            self.scalar_capabilities,
            self.coordinates,
        )
        adapter.fit(self.X, self.Y, sample_weight=self.sample_weight)
        prediction = adapter.predict(self.X[:2])

        flattened = self.Y.reshape(3, 8)
        self.assertEqual(len(created), 8)
        for index, estimator in enumerate(created):
            np.testing.assert_allclose(estimator.fit_y, flattened[:, index])
            np.testing.assert_allclose(estimator.fit_sample_weight, self.sample_weight)
        self.assertEqual(prediction.shape, (2, 4, 2))
        np.testing.assert_allclose(prediction[0], np.arange(8).reshape(4, 2))

    def test_regressor_chain_uses_truth_then_predictions_as_dependencies(self):
        created = []

        def factory():
            estimator = RecordingScalarEstimator(len(created) + 1)
            created.append(estimator)
            return estimator

        adapter = RegressorChainMultiTargetAdapter(
            factory,
            self.scalar_capabilities,
            self.coordinates,
        )
        adapter.fit(self.X, self.Y, sample_weight=self.sample_weight)
        prediction = adapter.predict(self.X[:1])

        flattened = self.Y.reshape(3, 8)
        for index, estimator in enumerate(created):
            self.assertEqual(estimator.fit_X.shape, (3, 2 + index))
            np.testing.assert_allclose(estimator.fit_X[:, 2:], flattened[:, :index])
            np.testing.assert_allclose(estimator.fit_sample_weight, self.sample_weight)
            self.assertEqual(estimator.predict_X.shape, (1, 2 + index))
            if index:
                np.testing.assert_allclose(
                    estimator.predict_X[:, 2:],
                    np.arange(1, index + 1, dtype=float).reshape(1, index),
                )
        np.testing.assert_allclose(prediction[0], np.arange(1, 9).reshape(4, 2))

    def test_native_strictly_flattens_and_restores_time_major_shape(self):
        estimator = RecordingNativeEstimator()
        adapter = NativeMultiTargetAdapter(
            lambda: estimator,
            self.native_capabilities,
            self.coordinates,
        )
        adapter.fit(self.X, self.Y, sample_weight=self.sample_weight)
        prediction = adapter.predict(self.X[:2])

        np.testing.assert_allclose(estimator.fit_y, self.Y.reshape(3, 8))
        np.testing.assert_allclose(estimator.fit_sample_weight, self.sample_weight)
        self.assertEqual(prediction.shape, (2, 4, 2))
        np.testing.assert_allclose(prediction[0], self.Y[0])

    def test_coordinates_are_fixed_time_major_metadata(self):
        adapter = IndependentMultiTargetAdapter(
            lambda: RecordingScalarEstimator(),
            self.scalar_capabilities,
            self.coordinates,
        )
        self.assertEqual(adapter.target_coordinates, self.coordinates)
        self.assertEqual(adapter.metadata["target_coordinates"], self.coordinates)
        with self.assertRaises(TypeError):
            adapter.metadata["target_coordinates"] = ()

        invalid = self.coordinates[:2] + self.coordinates[4:6] + self.coordinates[2:4] + self.coordinates[6:]
        with self.assertRaises(ValueError):
            IndependentMultiTargetAdapter(
                lambda: RecordingScalarEstimator(),
                self.scalar_capabilities,
                invalid,
            )

    def test_native_capability_false_raises_without_fallback(self):
        factory_calls = []
        with self.assertRaises(ValueError):
            NativeMultiTargetAdapter(
                lambda: factory_calls.append(True),
                self.scalar_capabilities,
                self.coordinates,
            )
        self.assertEqual(factory_calls, [])

    def test_quantile_chain_is_forbidden(self):
        with self.assertRaises(ValueError):
            RegressorChainMultiTargetAdapter(
                lambda: RecordingScalarEstimator(),
                self.scalar_capabilities,
                self.coordinates,
                probabilistic_mode="quantile",
            )

    def test_sample_weight_requires_declared_capability(self):
        no_weight = EstimatorCapabilities(
            scalar_target=True,
            scalar_quantile=False,
            native_multi_target_point=False,
            native_multi_target_quantile=False,
            sample_weight=False,
            categorical=False,
            nan_support=False,
        )
        estimator = RecordingScalarEstimator()
        adapter = IndependentMultiTargetAdapter(
            lambda: estimator,
            no_weight,
            self.coordinates,
        )
        with self.assertRaises(ValueError):
            adapter.fit(self.X, self.Y, sample_weight=self.sample_weight)
        self.assertIsNone(estimator.fit_y)

    def test_fit_and_predict_shapes_are_strict(self):
        adapter = IndependentMultiTargetAdapter(
            lambda: RecordingScalarEstimator(),
            self.scalar_capabilities,
            self.coordinates,
        )
        with self.assertRaises(ValueError):
            adapter.fit(self.X, self.Y.reshape(3, 8))
        with self.assertRaises(ValueError):
            adapter.fit(self.X, self.Y[:, :, :1])

        adapter.fit(self.X, self.Y)
        with self.assertRaises(ValueError):
            adapter.predict(self.X[:, 0])

    def test_all_adapters_support_single_target_coordinates(self):
        coordinates = tuple(TargetCoordinate("load", step) for step in range(1, 5))
        Y = np.arange(12, dtype=float).reshape(3, 4, 1)

        independent = IndependentMultiTargetAdapter(
            lambda: RecordingScalarEstimator(),
            self.scalar_capabilities,
            coordinates,
        ).fit(self.X, Y)
        chain = RegressorChainMultiTargetAdapter(
            lambda: RecordingScalarEstimator(),
            self.scalar_capabilities,
            coordinates,
        ).fit(self.X, Y)
        native_estimator = RecordingNativeEstimator()
        native = NativeMultiTargetAdapter(
            lambda: native_estimator,
            self.native_capabilities,
            coordinates,
        ).fit(self.X, Y)

        self.assertEqual(independent.predict(self.X[:2]).shape, (2, 4, 1))
        self.assertEqual(chain.predict(self.X[:2]).shape, (2, 4, 1))
        self.assertEqual(native.predict(self.X[:2]).shape, (2, 4, 1))


class ImportBoundaryTests(unittest.TestCase):
    def test_new_core_does_not_import_legacy_models_package(self):
        for path in (
            Path("model_forecasting/specs/estimator.py"),
            Path("model_training/estimators/capabilities.py"),
            Path("model_training/estimators/multi_target.py"),
            Path("model_training/strategies/base.py"),
        ):
            with self.subTest(path=path):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                imported_modules = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported_modules.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module is not None:
                        imported_modules.append(node.module)
                self.assertFalse(
                    any(
                        module == "models" or module.startswith("models.")
                        for module in imported_modules
                    )
                )


if __name__ == "__main__":
    unittest.main()
