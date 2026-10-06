"""融合跨模块合同回归：验证实际数值、身份与失败原因。"""
import tempfile
import unittest
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from model_ensemble.artifacts import OOFPredictionArtifact
from model_ensemble.outputs.cache import load_oof_cache, save_oof_cache
from model_ensemble.configuration.loader import parse_ensemble_document
from model_ensemble.runtime import run_ensemble_config
from model_ensemble.methods.weighted import fit_weighted, combine_weighted
from model_ensemble.training.trainer import _member_oof_scores
from model_ensemble.inference.deployment import predict_ensemble_bundle
from forecasting_core.specs.config import parse_model_config
from pipeline.runner import CanonicalBaseModelRunner
from data_loading import SourceRegistry
from test_ensemble_runtime import EnsembleRuntimeTestBase, RUNTIME_SERVICES, _ensemble_doc, _member_doc


class EnsembleCachePrecisionTest(unittest.TestCase):
    def test_noninteger_float_cache_roundtrip_is_bit_exact(self):
        rng = np.random.default_rng(7)
        values = rng.normal(size=(2, 2, 1))
        artifact = OOFPredictionArtifact(
            {"a": values, "b": values + 1.0}, ("a", "b"), ("load",),
            2, None, "float-roundtrip",
            folds=({"fold": 1}, {"fold": 2}),
        )
        with tempfile.TemporaryDirectory() as directory:
            save_oof_cache(directory, artifact)
            restored = load_oof_cache(directory, artifact.oof_fingerprint)
        for name in artifact.member_order:
            np.testing.assert_array_equal(restored.values_by_member[name], artifact.values_by_member[name])


class EnsembleResolvedIdentityTest(EnsembleRuntimeTestBase):
    def test_member_semantics_change_output_identity_without_changing_ref(self):
        first = self._run("averaging", use_oof_cache=False)
        path = self.root / "member_direct.yaml"
        raw = yaml.safe_load(path.read_text())
        raw["estimator"]["params"]["alpha"] = 100.0
        path.write_text(yaml.safe_dump(raw))
        second = self._run("averaging", use_oof_cache=False)
        self.assertNotEqual(first["model_dir"], second["model_dir"])
        self.assertNotEqual(first["bundle"].config_fingerprint, second["bundle"].config_fingerprint)
        self.assertTrue((first["model_dir"] / "model.pkl").is_file())
        self.assertGreater(float(np.max(np.abs(first["combined_values"] - second["combined_values"]))), 0.0)

    def test_top_level_origin_changes_cache_and_matches_cold_folds(self):
        early = _ensemble_doc("averaging")
        early["validation"]["forecast_origin"] = "2026-01-03T20:00:00"
        first = run_ensemble_config(
            parse_ensemble_document(early), base_dir=self.root,
            output_root=self.root, services=RUNTIME_SERVICES,
        )
        warm = self._run("averaging")
        cold = self._run("averaging", use_oof_cache=False)
        self.assertNotEqual(first["oof_fingerprint"], warm["oof_fingerprint"])
        self.assertFalse(warm["oof_cache_hit"])
        self.assertEqual(warm["oof"].folds, cold["oof"].folds)
        for name in warm["oof"].member_order:
            np.testing.assert_array_equal(warm["oof"].values_by_member[name], cold["oof"].values_by_member[name])


class EnsembleQuantileContractTest(unittest.TestCase):
    def test_weighted_quantiles_pool_errors_and_share_target_weights(self):
        actual = np.arange(16.0).reshape(2, 4, 2)
        a = actual[..., None] + np.array([-1.0, 0.0, 1.0])
        b = actual[..., None] + np.array([-2.0, 0.0, 2.0])
        artifact = fit_weighted({"a": a, "b": b}, actual)
        for weights in artifact.weights_by_target.values():
            np.testing.assert_allclose(weights, (2.0 / 3.0, 1.0 / 3.0))
        np.testing.assert_allclose(combine_weighted(artifact, {"a": a, "b": b}), (2 * a + b) / 3)

    def test_member_pinball_uses_actual_minus_prediction(self):
        actual = np.zeros((1, 1, 1))
        values = np.array([1.0, 2.0, 3.0]).reshape(1, 1, 1, 3)
        oof = OOFPredictionArtifact(
            {"a": values, "b": values}, ("a", "b"), ("load",),
            1, (0.1, 0.5, 0.9), "pinball",
        )
        self.assertAlmostEqual(_member_oof_scores(oof, actual)["a"]["load"]["pinball"], (0.9 + 1.0 + 0.3) / 3)


class EnsembleDeploymentProbabilityTest(EnsembleRuntimeTestBase):
    def test_top_level_point_quantile_survives_bundle_reload(self):
        import pickle
        values = 20.0 + 0.5 * np.arange(72) + 3 * np.sin(np.arange(72))
        pd.DataFrame({"time": pd.date_range("2026-01-01", periods=72, freq="1h"), "load": values}).to_csv(self.root / "data.csv", index=False)
        configs = {}
        for name in ("direct", "recursive"):
            raw = _member_doc("direct", "qr", name)
            raw["probabilistic"] = {"mode": "quantile", "quantiles": [0.1, 0.5, 0.9], "point_quantile": 0.5}
            (self.root / f"member_{name}.yaml").write_text(yaml.safe_dump(raw))
            configs[f"m_{name}"] = parse_model_config(raw, source=name)
        raw = _ensemble_doc("averaging", mode="quantile")
        raw["probabilistic"]["point_quantile"] = 0.1
        result = run_ensemble_config(parse_ensemble_document(raw), base_dir=self.root, output_root=self.root, services=RUNTIME_SERVICES)
        with (result["model_dir"] / "model.pkl").open("rb") as handle:
            bundle = pickle.load(handle)
        designs, providers = {}, {}
        for name, config in configs.items():
            runner = CanonicalBaseModelRunner(config, SourceRegistry(config.data, self.root), pd.Timestamp(raw["validation"]["forecast_origin"]))
            saved = bundle.model["member_bundles"][name]
            matrices, provider = runner.builder.forecast_designs(runner.origin, target_transform=saved.target_transform)
            designs[name], providers[name] = matrices[0], provider
        prediction = predict_ensemble_bundle(bundle, designs, forecast_times=result["forecast_times"], raw_feature_providers=providers)
        self.assertEqual(bundle.probabilistic_spec.point_quantile, 0.1)
        np.testing.assert_allclose(prediction.point.values, result["combined_values"][..., 0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(prediction.quantiles.values, result["combined_values"], rtol=0, atol=1e-12)
