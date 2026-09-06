"""Lifecycle wiring only: no imports of runtime, fits, predictions or backtests."""
import ast
from pathlib import Path
import unittest
from model_pipeline.run_state import require_completed_state

ROOT = Path(__file__).resolve().parents[1]


class LifecycleStaticContractTest(unittest.TestCase):
    def test_completion_requires_matching_fingerprint_and_completed_state(self):
        require_completed_state({"schema_version": 1, "config_fingerprint": "fixture", "status": "completed"}, "fixture")
        for status, fingerprint in (("running", "fixture"), ("failed", "fixture"), ("completed", "other")):
            with self.assertRaisesRegex(ValueError, "not completed"):
                require_completed_state({"schema_version": 1, "config_fingerprint": fingerprint, "status": status}, "fixture")

    def test_calendar_backtest_precedes_final_fit_and_bundle_persistence(self):
        tree = ast.parse((ROOT / "model_pipeline/lifecycle.py").read_text())
        run = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "execute_lifecycle")
        calls = {}
        for node in ast.walk(run):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                calls.setdefault(name, []).append(node.lineno)
        self.assertIn("run_calendar_month_backtest", calls)
        self.assertLess(max(calls["run_fixed_step_backtest"]), min(calls["fit_final"]))
        self.assertLess(max(calls["run_calendar_month_backtest"]), min(calls["fit_final"]))
        self.assertLess(max(calls["fit_final"]), min(calls["persist_model_bundle"]))

    def test_calendar_helper_does_not_rewrite_final_artifacts(self):
        source = (ROOT / "model_testing/calendar_month.py").read_text()
        self.assertNotIn("persist_model_bundle", source)
        self.assertNotIn("prediction.csv", source)
        self.assertNotIn("resolved_config.json", source)

    def test_no_post_completion_calendar_hook(self):
        for name in ("runner.py", "batch_runtime.py"):
            self.assertNotIn("overwrite_calendar_month_backtest", (ROOT / "model_pipeline" / name).read_text())


if __name__ == "__main__":
    unittest.main()
