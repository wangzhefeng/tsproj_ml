"""融合持久化：真实可读证据、CQR 状态和中断后的恢复。"""
import json
import pickle
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from model_ensemble.outputs import cache
from model_ensemble.inference.deployment import predict_ensemble_bundle
from model_ensemble.configuration.loader import parse_ensemble_document
from model_ensemble.runtime import run_ensemble_config
from forecasting_core.specs.config import parse_model_config
from pipeline.runner import CanonicalBaseModelRunner
from data_loading import SourceRegistry
from test_ensemble_runtime import EnsembleRuntimeTestBase, RUNTIME_SERVICES, _member_doc, _ensemble_doc
from test_ensemble_oof import _artifact
from model_predicting.artifacts.evidence_collect import collect_model_evidence


class EnsembleCacheTransactionTest(unittest.TestCase):
    def test_interrupted_write_keeps_entry_unpublished_and_retry_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = _artifact()
            original = cache._atomic_write_bytes
            def fail_metadata(path, data):
                if path.name == cache.OOF_METADATA_FILE:
                    raise OSError("injected metadata write failure")
                return original(path, data)
            with patch.object(cache, "_atomic_write_bytes", fail_metadata):
                with self.assertRaisesRegex(OSError, "injected metadata"):
                    cache.save_oof_cache(directory, artifact)
            calls = []
            def factory():
                calls.append(1)
                return artifact
            restored, hit = cache.get_or_create_oof_cache(directory, artifact.oof_fingerprint, factory)
            self.assertFalse(hit)
            self.assertEqual(calls, [1])
            np.testing.assert_array_equal(restored.values_by_member["a"], artifact.values_by_member["a"])
            self.assertTrue(cache.get_or_create_oof_cache(directory, artifact.oof_fingerprint, factory)[1])
            self.assertEqual(calls, [1])


class EnsembleBundleEvidenceTest(EnsembleRuntimeTestBase):
    def test_failed_persistence_marks_failed_and_retry_finishes(self):
        def broken_persist(*args):
            raise OSError("injected bundle write failure")
        with self.assertRaisesRegex(OSError, "injected bundle"):
            self._run("averaging", services=replace(RUNTIME_SERVICES, persist_bundle=broken_persist))
        states = list(self.root.rglob("run_state.json"))
        self.assertEqual(len(states), 1)
        self.assertEqual(json.loads(states[0].read_text())["status"], "failed")
        recovered = self._run("averaging")
        state = json.loads(states[0].read_text())
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["config_fingerprint"], recovered["bundle"].config_fingerprint)
        self.assertTrue(recovered["oof_cache_hit"])
        self.assertTrue(state["artifacts"])

    def test_member_final_bundle_keeps_real_visibility_and_source_lineage(self):
        result = self._run("averaging", use_oof_cache=False)
        for member in result["bundle"].model["member_bundles"].values():
            evidence = collect_model_evidence(member.model)
            self.assertTrue(evidence, "fitted member wrappers must be reachable through slotted adapters")
            self.assertTrue(all(item["fitted"] for item in evidence))
            self.assertTrue(all("native_params" in item for item in evidence))
            self.assertTrue(member.source_lineage)
            self.assertTrue(member.feature_lineage)
            self.assertTrue(member.input_schema["visibility_proof"])
            self.assertTrue(all(item["forecast_origin"] == "2026-01-03T23:00:00" for item in member.input_schema["visibility_proof"]))
        self.assertTrue(result["bundle"].source_lineage)

    def test_fusion_cqr_is_outer_only_persisted_and_reloaded(self):
        document = _ensemble_doc("averaging", mode="quantile")
        document["validation"]["fold_count"] = 3
        document["probabilistic"].update({
            "intervals": [{"name": "central80", "lower_quantile": 0.1, "upper_quantile": 0.9}],
            "calibration": {"method": "cqr", "interval": "central80", "target_coverage": 0.8,
                            "calibration_windows": 3, "min_windows": 1, "min_scores": 1},
        })
        configs = {}
        for name in ("direct", "recursive"):
            raw = _member_doc("direct", "qr", name)
            raw["probabilistic"] = {"mode": "quantile", "quantiles": [0.1, 0.5, 0.9], "point_quantile": 0.5}
            (self.root / f"member_{name}.yaml").write_text(yaml.safe_dump(raw))
            configs[f"m_{name}"] = parse_model_config(raw, source=name)
        result = run_ensemble_config(parse_ensemble_document(document), base_dir=self.root, output_root=self.root, services=RUNTIME_SERVICES)
        with (result["model_dir"] / "model.pkl").open("rb") as handle:
            bundle = pickle.load(handle)
        self.assertEqual(bundle.calibration_state["status"], "applied")
        first_fold = result["backtest"].metadata["folds"][0]
        self.assertEqual(first_fold["calibration"]["status"], "insufficient_windows")
        self.assertEqual(bundle.calibration_state["evaluation_role"], "outer_holdout")
        designs, providers = {}, {}
        for name, config in configs.items():
            origin = pd.Timestamp(document["validation"]["forecast_origin"])
            runner = CanonicalBaseModelRunner(config, SourceRegistry(config.data, self.root), origin)
            matrices, provider = runner.builder.forecast_designs(origin, target_transform=bundle.model["member_bundles"][name].target_transform)
            designs[name], providers[name] = matrices[0], provider
        prediction = predict_ensemble_bundle(bundle, designs, forecast_times=result["forecast_times"], raw_feature_providers=providers)
        interval = prediction.metadata["prediction_intervals"]["central80"]
        saved = pd.read_csv(result["forecast_dir"] / "prediction.csv")
        for key, column in zip(("lower", "upper"), interval["columns"]):
            np.testing.assert_allclose(interval[key].ravel(), saved[column], rtol=0, atol=1e-12)
