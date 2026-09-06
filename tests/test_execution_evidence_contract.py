"""Evidence plumbing tests: synthetic metadata only, no model fit/predict."""
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
from model_ensemble.artifacts import OOFPredictionArtifact
from model_ensemble.cache import save_oof_cache, load_oof_cache
from model_ensemble.contracts import member_execution_evidence

ROOT = Path(__file__).resolve().parents[1]


class ExecutionEvidenceContractTest(unittest.TestCase):
    def test_native_nan_sentinel_and_numpy_parameters_are_json_safe_snapshots(self):
        from model_forecasting.evidence import json_evidence
        source = {"missing": float("nan"), "threads": np.int64(2), "levels": np.array([0.1, 0.9])}
        snapshot = json_evidence(source)
        self.assertEqual(snapshot["missing"], {"nonfinite_float": "nan"})
        self.assertEqual(snapshot["threads"], 2)
        self.assertEqual(snapshot["levels"], [0.1, 0.9])
        source["levels"][0] = 0.2
        self.assertEqual(snapshot["levels"], [0.1, 0.9])
        json.dumps(snapshot, allow_nan=False)

    def test_custom_runner_missing_evidence_is_explicit(self):
        self.assertEqual(member_execution_evidence(object(), None, None)["status"], "unavailable")

    def test_provider_errors_are_not_hidden(self):
        def broken(*args):
            raise ValueError("evidence failed")
        with self.assertRaisesRegex(ValueError, "evidence failed"):
            member_execution_evidence(SimpleNamespace(execution_evidence=broken), None, None)

    def test_metadata_roundtrip_and_old_cache_without_evidence(self):
        evidence = ({"fold": 1, "members": {"a": {"status": "recorded", "fixture": True}}},)
        artifact = OOFPredictionArtifact(
            values_by_member={"a": np.zeros((1, 2, 1))}, member_order=("a",),
            targets=("y",), horizon=2, quantile_levels=None, oof_fingerprint="fixture",
            execution_evidence=evidence,
        )
        with TemporaryDirectory() as directory:
            path = save_oof_cache(directory, artifact)
            restored = load_oof_cache(directory, "fixture")
            self.assertEqual(restored.execution_evidence, evidence)
            metadata_path = path / "oof_metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata.pop("execution_evidence")
            metadata_path.write_text(json.dumps(metadata))
            self.assertEqual(load_oof_cache(directory, "fixture").execution_evidence, ())

    def test_production_wiring_without_executing_lifecycles(self):
        # R5b（2026-09-06）：逐折证据收集移入 model_testing/scoring.py（共用评分体），
        # calendar_month 经 score_holdout_fold 间接接线；点名检查随之更新。
        for path, name in (("model_forecasting/runtime.py", "score_holdout_fold"),
                           ("model_testing/calendar_month.py", "score_holdout_fold"),
                           ("model_testing/scoring.py", "execution_evidence"),
                           ("model_ensemble/oof.py", "member_execution_evidence"),
                           ("model_ensemble/trainer.py", "member_execution_evidence")):
            tree = ast.parse((ROOT / path).read_text())
            names = {getattr(n.func, "id", getattr(n.func, "attr", "")) for n in ast.walk(tree) if isinstance(n, ast.Call)}
            self.assertIn(name, names, path)
        source = (ROOT / "model_ensemble/oof.py").read_text()
        fingerprint_section = source.split("fingerprint_payload =", 1)[1].split("oof_fingerprint =", 1)[0]
        self.assertNotIn("execution_evidence", fingerprint_section)


if __name__ == "__main__":
    unittest.main()
