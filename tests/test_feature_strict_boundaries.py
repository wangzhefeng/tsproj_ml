"""特征公共边界：快路径不能省略 schema 合同。"""
import unittest
import tempfile
from pathlib import Path
from dataclasses import replace

from data_loading import InformationSetRequest, SourceRegistry
from feature_engineering import FeatureCompiler
from tests import test_canonical_runtime_smoke as smoke

import numpy as np
import pandas as pd

from feature_engineering.transforms.scaling import CanonicalFeatureScaler


class FeatureBoundaryTest(unittest.TestCase):
    def test_passthrough_dataframe_aligns_names_and_rejects_invalid_schema(self):
        scaler = CanonicalFeatureScaler({}, feature_names=("a", "b"))
        scaler.fit_transform(pd.DataFrame({"a": [1.0, 2.0], "b": [10.0, 20.0]}))
        actual = scaler.transform(pd.DataFrame({"b": [30.0], "a": [3.0]}))
        np.testing.assert_array_equal(actual, [[3.0, 30.0]])
        for frame in (
            pd.DataFrame({"wrong": [3.0], "other": [30.0]}),
            pd.DataFrame([[3.0, 30.0]], columns=pd.Index(["a", "a"])),
        ):
            with self.subTest(columns=list(frame)), self.assertRaisesRegex(ValueError, "schema|unique"):
                scaler.transform(frame)
    def test_batch_respects_each_history_start(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            times = pd.date_range("2026-01-01", periods=60, freq="h")
            pd.DataFrame({"time": times, "load": np.arange(60.)}).to_csv(path, index=False)
            base = smoke.CanonicalRuntimeSmokeTest().build_config(path, mode="point", strategy="direct")
            config = replace(base, features=replace(base.features, transformations={
                "advanced": {"expanding": {"columns": ["load"], "stats": ["mean"]}},
            }))
            requests = [InformationSetRequest(times[p], times[p + 1:p + 3], (), history_start=times[s])
                        for p, s in ((30, 0), (40, 20))]
            registry = SourceRegistry(config.data, directory)
            information = [registry.materialize(request) for request in requests]
            compiler = FeatureCompiler(config)
            actual = compiler.compile_batch(information, requests)
            for compiled, expected in zip(actual, (15., 30.)):
                np.testing.assert_array_equal(compiled.frame.load_expanding_mean, [expected, expected])
    def test_column_availability_uses_single_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            times = pd.date_range("2026-01-01", periods=60, freq="h")
            pd.DataFrame({"time": times, "published": times, "load": np.arange(60.)}).to_csv(path, index=False)
            base = smoke.CanonicalRuntimeSmokeTest().build_config(path, mode="point", strategy="direct")
            source = replace(base.data.sources[0], availability="column", available_at_col="published")
            config = replace(base, data=replace(base.data, sources=(source,)), features=replace(base.features,
                transformations={"advanced": {"rolling": {"columns": ["load"], "windows": [4], "stats": ["mean"]}}}))
            compiler = FeatureCompiler(config)
            request = InformationSetRequest(times[47], times[48:50], ())
            self.assertFalse(compiler.batch_eligibility((request,)).eligible)
            info = SourceRegistry(config.data, directory).materialize(request)
            with self.assertWarnsRegex(RuntimeWarning, "falling back"):
                result = compiler.compile_batch((info,), (request,))[0]
            np.testing.assert_array_equal(result.frame.load_rolling_mean_4, [45.5, 45.5])
    def test_feature_configuration_rejects_silent_noops_and_coercions(self):
        base = smoke.CanonicalRuntimeSmokeTest().build_config("unused.csv", mode="point", strategy="direct")
        cases = (
            ({"advanced": {"rolling": {"columns": ["load"], "windows": [4], "stats": ["mean"], "typo": 1}}}, "unknown"),
            ({"advanced": {"interaction": {"column_pairs": [["load__lag_2", "load__lag_3"]], "operations": ["typo"]}}}, "operation"),
            ({"advanced": {"fourier": {"columns": ["load"], "windows": [16], "top_k": 1.9}}}, "top_k"),
        )
        for transformations, message in cases:
            with self.subTest(transformations=transformations), self.assertRaisesRegex((ValueError, TypeError), message):
                replace(base.features, transformations=transformations)
    def test_named_interaction_cannot_overwrite_lag_or_metadata(self):
        base = smoke.CanonicalRuntimeSmokeTest().build_config("unused.csv", mode="point", strategy="direct")
        for name in ("load__lag_2", "target_time", "horizon_step"):
            config = replace(base, features=replace(base.features, transformations={
                "interactions": {name: ["load__lag_2", "load__lag_3"]},
            }))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "collision|reserved"):
                FeatureCompiler(config)
    def test_division_preserves_nonzero_denominators_and_rejects_zero(self):
        base = smoke.CanonicalRuntimeSmokeTest().build_config("unused.csv", mode="point", strategy="direct")
        compiler = FeatureCompiler(base)
        spec = {"column_pairs": [["a", "b"]], "operations": ["divide"]}
        for denominator in (2., -1e-8):
            row = {"a": 1., "b": denominator}
            compiler._compile_interaction_spec(row, spec)
            self.assertEqual(row["a_divide_b"], 1. / denominator)
        with self.assertRaisesRegex(ValueError, "denominator"):
            compiler._compile_interaction_spec({"a": 1., "b": 0.}, spec)
        with self.assertRaisesRegex(ValueError, "denominator"):
            compiler._compile_interaction_spec({"a": np.ones(2), "b": np.array([1., 0.])}, spec, vectorized=True)
    def test_fourier_nyquist_amplitude_and_one_sided_energy(self):
        from feature_engineering.kernels.spectral import fourier_features
        time = np.arange(16.)
        signal = np.cos(np.pi * time) + np.cos(2 * np.pi * time / 4)
        actual = fourier_features(signal, top_k=2, band_periods=((2, 3), (4, 5)))
        self.assertAlmostEqual(actual["amp_1"], 1.)
        self.assertAlmostEqual(actual["amp_2"], 1.)
        self.assertAlmostEqual(actual["bandenergy_1"], 2. / 3.)
        self.assertAlmostEqual(actual["bandenergy_2"], 1. / 3.)
    def test_removed_scaling_switches_are_rejected(self):
        from feature_engineering.transform_specs import normalize_feature_scaling, normalize_target_transformations
        with self.assertRaisesRegex(ValueError, "grouped"):
            normalize_feature_scaling({"method": "standard", "grouped": False})
        with self.assertRaisesRegex(ValueError, "inverse"):
            normalize_target_transformations({"scaling": {"method": "standard", "inverse": False}})
    def test_rolling_requires_full_window_and_defined_sample_statistics(self):
        from feature_engineering.kernels.history import history_statistic
        with self.assertRaisesRegex(ValueError, "samples|history"):
            history_statistic(pd.Series([1.]), "std")
        base = smoke.CanonicalRuntimeSmokeTest().build_config("unused.csv", mode="point", strategy="direct")
        with self.assertRaisesRegex(ValueError, "samples|window"):
            replace(base.features, transformations={"advanced": {
                "rolling": {"columns": ["load"], "windows": [2], "stats": ["kurt"]},
            }})
    def test_selection_rejects_string_and_unknown_force_keep_in_small_schema(self):
        from feature_engineering.selection import CanonicalFeatureSelector, normalize_feature_selection
        with self.assertRaisesRegex(TypeError, "force_keep"):
            normalize_feature_selection({"enabled": True, "force_keep": "abc"})
        selector = CanonicalFeatureSelector(normalize_feature_selection({"enabled": True, "force_keep": ["missing"]}), ("a", "b"))
        with self.assertRaisesRegex(ValueError, "force_keep"):
            selector.fit(np.ones((3, 2)), np.arange(3.))


if __name__ == "__main__":
    unittest.main()
