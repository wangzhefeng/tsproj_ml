"""Completed states must describe verified, nonempty canonical artifacts."""
import json
import tempfile
import unittest
from pathlib import Path

from model_pipeline.batch_runtime import _artifacts_complete, verify_batch_results
from model_pipeline.batch_runtime import run_canonical_batch
from model_pipeline.batch_artifacts import artifact_digests
from tests import test_batch_runtime as batch_fixture


class BatchArtifactGateTest(unittest.TestCase):
    def test_real_manifest_rejects_corruption_and_wrong_identity(self):
        fixture = batch_fixture.CanonicalBatchRuntimeTest()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        report = run_canonical_batch(fixture.paths[:1], output_root=fixture.root / "verified")
        self.assertEqual(report.completed_count, 1)
        state = json.loads(report.state_path.read_text())
        task = next(iter(state["tasks"].values()))
        self.assertTrue(_artifacts_complete(task))
        for key in ("model", "forecast", "backtest", "resolved_config"):
            path = Path(task["artifacts"][key])
            original = path.read_bytes()
            for corrupted in (b"", b"not a valid artifact"):
                with self.subTest(key=key, corruption=corrupted):
                    path.write_bytes(corrupted)
                    self.assertFalse(_artifacts_complete(task))
                    altered = {**task, "artifact_sha256": artifact_digests(task["artifacts"])}
                    self.assertFalse(_artifacts_complete(altered))
            path.write_bytes(original)
        self.assertFalse(_artifacts_complete({**task, "config_fingerprint": "0" * 64}))
        self.assertFalse(_artifacts_complete({**task, "result_identity": "wrong"}))
        self.assertTrue(_artifacts_complete(task))

    def test_empty_manifest_cannot_be_complete(self):
        self.assertFalse(_artifacts_complete({"artifacts": {}}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(json.dumps({
                "config_paths": ["fake.yaml"],
                "tasks": {"fake": {"path": "fake.yaml", "status": "completed", "artifacts": {}}},
            }))
            with self.assertRaises(ValueError):
                verify_batch_results(path)

    def test_partial_or_empty_file_manifest_cannot_be_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"
            path.touch()
            self.assertFalse(_artifacts_complete({"artifacts": {"forecast": str(path)}}))


if __name__ == "__main__":
    unittest.main()
