"""Direct result labels, nonsemantic identity, and real artifact wiring."""
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from config.config_loader import load_yaml_config
from forecasting_core.specs import ForecastStrategySpec
from model_pipeline.runner import run_canonical_config
from scripts.audit_forecast_configs import _build_single_row
import test_canonical_runtime_smoke as smoke
import test_forecast_config_fingerprint as fingerprints


class DirectResultIdentityTest(unittest.TestCase):
    def test_metadata_and_artifacts_use_effective_method(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "history.csv"
            pd.DataFrame({"time": pd.date_range("2026-01-01", periods=60, freq="h"),
                          "load": np.arange(60, dtype=float) + 10}).to_csv(source, index=False)
            baseline = smoke.CanonicalRuntimeSmokeTest().build_config(source, mode="point", strategy="direct")
            variants = (
                ("independent_models", False, False, "direct"),
                ("single_model_horizon", False, False, "direct-pointwise"),
                ("single_model_horizon", True, False, "direct-pointwise-horizon"),
                ("single_model_horizon", True, True, "direct-pointwise-horizon"),
            )
            for layout, enabled, cyclical, label in variants:
                with self.subTest(label=label):
                    config = replace(baseline, features=replace(baseline.features, transformations={
                        "direct": {"layout": layout, "align_to_target": False,
                                   "horizon_feature": {"enabled": enabled, "cyclical": cyclical}}}),
                        output={"scenario_subpath": "case", "directories": {
                            "checkpoints": str(root / "models"), "tests": str(root / "tests"),
                            "forecast": str(root / "forecast")}})
                    run_canonical_config(config)
                    expected = config.result_method()
                    for parent in ("models", "tests", "forecast"):
                        self.assertTrue((root / parent / "case" / config.result_identity()).is_dir())
                    schema = json.loads((root / "models" / "case" / config.result_identity() / "resolved_model.json").read_text())
                    self.assertEqual(schema["config_fingerprint"], config.fingerprint())
                    self.assertGreater((root / "models" / "case" / config.result_identity() / "model.pkl").stat().st_size, 0)
                    prediction = pd.read_csv(root / "forecast" / "case" / config.result_identity() / "prediction.csv")
                    self.assertTrue(np.isfinite(prediction["predict_value"]).all())
                    metadata = json.loads((root / "tests" / "case" / config.result_identity() / "result_metadata.json").read_text())
                    resolved = json.loads((root / "forecast" / "case" / config.result_identity() / "resolved_config.json").read_text())
                    self.assertEqual(metadata["result_method"], expected)
                    self.assertEqual(resolved["runtime"]["result_method"], expected)
                    row = _build_single_row(root / "alias.yaml", root, config)
                    self.assertEqual(row["result_method"], expected)
                    self.assertEqual(expected["method_label"], label)
                    self.assertIs(metadata["result_method"]["horizon_feature_cyclical"], cyclical)

    def test_fingerprint_is_original_semantic_hash_and_alias_independent(self):
        config = fingerprints.CanonicalConfigFingerprintTest().config()
        payload = config.canonical_payload()
        payload.pop("output")
        payload["validation"] = config.validation.semantic_payload()
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(config.fingerprint(), digest)
        with tempfile.TemporaryDirectory() as directory:
            for name in ("direct.yaml", "unrelated-alias.yaml"):
                path = Path(directory) / name
                path.write_text(json.dumps(config.canonical_payload()))
                self.assertEqual(load_yaml_config(path).result_identity(), config.result_identity())
        for strategy in ("recursive", "mimo"):
            other = replace(config, strategy=ForecastStrategySpec(strategy))
            self.assertEqual(other.result_identity(), f"{strategy}-ridge-local-k1-{other.fingerprint()[:12]}")

    def test_invalid_direct_settings_raise_instead_of_mislabeling(self):
        baseline = fingerprints.CanonicalConfigFingerprintTest().config()
        for direct in ({}, {"layout": "wrong"}, {"layout": "single_model_horizon", "horizon_feature": []},
                       {"layout": "single_model_horizon", "horizon_feature": {"enabled": 1}}):
            with self.subTest(direct=direct):
                config = replace(baseline, features=replace(baseline.features, transformations={"direct": direct}))
                with self.assertRaises((TypeError, ValueError)):
                    config.result_identity()
