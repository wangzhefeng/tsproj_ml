"""退役接口拒绝与保留结果 IO 的回归，不训练模型。"""

import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from data_process.load_event_detection import suppress_boundary_artifacts
from model_forecasting.results import CanonicalResultReader


ROOT = Path(__file__).resolve().parents[1]


class CoreCleanupContractTest(unittest.TestCase):
    def test_retired_definitions_are_not_compatibility_shims(self):
        retired = {
            "forecasting_core/probabilistic_spec.py": {
                "_legacy_spec", "_legacy_fields_are_explicit",
                "resolve_probabilistic_spec", "apply_probabilistic_spec_to_args",
            },
            "model_pipeline/supervised_design.py": {"_labels", "_holdout_training_indices"},
            "model_ensemble/runtime.py": {"_member_origin"},
            "model_ensemble/contracts.py": {"FusionMethod"},
            "model_ensemble/trainer.py": {"MemberAuditScores"},
            "feature_engineering/transforms/pipeline.py": {
                "attach_fitted_target_scaler", "restore_quantile_matrix",
            },
            "data_process/outlier_process.py": {"remove_outliers", "_keep_runs_at_least"},
        }
        for relative, names in retired.items():
            with self.subTest(module=relative):
                tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
                defined = {
                    node.name for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                }
                self.assertEqual(defined & names, set())
        self.assertFalse((ROOT / "utils/frequency.py").exists())

    def test_result_readers_reject_removed_filter_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "canonical.csv"
            pd.DataFrame({
                "series_id": ["A"], "time": ["2026-01-01"], "target": ["load"],
                "predict_value": [2.0], "actual_value": [1.0], "window": [0],
                "plot_valid": [True],
            }).to_csv(path, index=False)
            for reader in (CanonicalResultReader.read_prediction, CanonicalResultReader.read_backtest):
                for keyword in ("target", "series_id"):
                    with self.subTest(reader=reader.__name__, keyword=keyword):
                        with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
                            reader(path, **{keyword: "unused"})

    def test_result_readers_preserve_all_series_targets_and_columns(self):
        expected = pd.DataFrame({
            "series_id": ["A", "B", "A"],
            "time": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"]),
            "target": ["load", "power", "power"],
            "predict_value": [2.0, 4.0, 8.0], "actual_value": [1.0, 3.0, 7.0],
            "window": [0, 0, 1], "plot_valid": [True, False, True],
            "predict_q10": [0.0, 2.0, 6.0],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "canonical.csv"
            expected.to_csv(path, index=False)
            for reader in (CanonicalResultReader.read_prediction, CanonicalResultReader.read_backtest):
                with self.subTest(reader=reader.__name__):
                    pd.testing.assert_frame_equal(reader(path), expected)

    def test_result_readers_still_reject_noncanonical_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wide.csv"
            pd.DataFrame({"time": ["2026-01-01"], "load": [1.0]}).to_csv(path, index=False)
            for reader in (CanonicalResultReader.read_prediction, CanonicalResultReader.read_backtest):
                with self.subTest(reader=reader.__name__):
                    with self.assertRaisesRegex(ValueError, "non-canonical"):
                        reader(path)

    def test_boundary_suppression_rejects_removed_config_argument(self):
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
            suppress_boundary_artifacts([], [], config=None)
        # 不让旧第三位置参数静默变成 zone_days。
        with self.assertRaises(TypeError):
            suppress_boundary_artifacts([], [], None)
        self.assertEqual(suppress_boundary_artifacts([], [], zone_days=2.0, amp_frac=0.5), [])

    def test_pickle_io_import_does_not_initialize_project_logging(self):
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import models.pickle_io; "
            "assert 'utils.log_util' not in sys.modules"
        )
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-B", "-c", code, str(ROOT)],
                cwd=directory, env=environment, capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
