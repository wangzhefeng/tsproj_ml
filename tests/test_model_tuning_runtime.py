"""真实调参生命周期：合成数据，不运行业务场景。"""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from config.config_loader import load_yaml_config
from forecasting_core.specs import EstimatorSpec
from model_tuning.runtime import run_tuning
from model_tuning.specs import TuningSpec
from pipeline.batch_artifacts import validate_artifacts
import test_canonical_runtime_smoke as smoke
from test_model_tuning import search_payload


class ModelTuningRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "load.csv"
        pd.DataFrame({
            "time": pd.date_range("2026-01-01", periods=80, freq="h"),
            "load": 100 + np.arange(80, dtype=float) + np.sin(np.arange(80)),
        }).to_csv(self.source, index=False)
        base = smoke.CanonicalRuntimeSmokeTest().build_config(self.source, mode="point", strategy="direct")
        self.base = replace(base, validation={
            "forecast_origin": "2026-01-02T23:00:00", "history_steps": 30,
            "train_window_steps": 12, "fold_count": 2, "stride_steps": 2,
            "performance": {"total_thread_limit": 1, "multi_output_n_jobs": 1},
        })

    def test_real_search_holdout_final_and_exported_yaml_run_independently(self):
        study_dir = self.root / "study"
        report = run_tuning(self.base, TuningSpec.from_mapping(search_payload()), study_dir=study_dir)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["trial_count"], 2)
        self.assertEqual(len(report["trials"]), 2)
        self.assertTrue(all(trial["status"] == "completed" for trial in report["trials"]))
        winner = min(report["trials"], key=lambda trial: (trial["score"], trial["number"]))
        self.assertEqual(report["best_trial"], winner["number"])
        self.assertEqual(report["selection_metric"], "RMSE")
        self.assertNotIn("holdout_score", report["trials"][report["best_trial"]])
        self.assertGreater(pd.Timestamp(report["holdout_label_start"]), pd.Timestamp(report["search_cutoff"]))
        validate_artifacts(report["final_artifacts"])
        best = load_yaml_config(study_dir / "best.yaml")
        self.assertEqual(best.fingerprint(), report["final_artifacts"]["config_fingerprint"])
        for trial in report["trials"]:
            trial_yaml = Path(trial["config_yaml"])
            self.assertTrue(trial_yaml.is_file())
            self.assertEqual(load_yaml_config(trial_yaml).fingerprint(), trial["config_fingerprint"])
            scores = pd.read_csv(Path(trial["test_dir"]) / "test_scores_df.csv")
            expected = scores.loc[scores["scope"] == "aggregate", "RMSE"].mean()
            self.assertEqual(trial["score"], expected)
        # 独立 CLI 重新执行导出的配置，不依赖 study 或调参模块的内存状态。
        completed = subprocess.run([
            sys.executable, "run.py", "--config-yaml", str(study_dir / "best.yaml"),
        ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=90)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        exported = Path(best.output["directories"]["forecast"])
        self.assertEqual(len(list(exported.rglob("prediction.csv"))), 1)
        self.assertEqual(len(list(Path(best.output["directories"]["checkpoints"]).rglob("model.pkl"))), 1)
        persisted = json.loads((study_dir / "study.json").read_text())
        self.assertEqual(persisted, report)

    def test_holdout_values_do_not_affect_trial_scores_or_selection(self):
        spec = TuningSpec.from_mapping(search_payload())
        first = run_tuning(self.base, spec, study_dir=self.root / "first")
        frame = pd.read_csv(self.source)
        frame.loc[pd.to_datetime(frame["time"]) > pd.Timestamp(self.base.validation["forecast_origin"]), "load"] *= 3.0
        frame.to_csv(self.source, index=False)
        second = run_tuning(self.base, spec, study_dir=self.root / "second")
        self.assertEqual(first["best_trial"], second["best_trial"])
        self.assertEqual([trial["parameters"] for trial in first["trials"]],
                         [trial["parameters"] for trial in second["trials"]])
        self.assertEqual([trial["score"] for trial in first["trials"]],
                         [trial["score"] for trial in second["trials"]])
        self.assertNotEqual(first["holdout_score"], second["holdout_score"])

    def test_overlap_rejected_before_study_creation(self):
        recipe = search_payload()
        recipe["holdout_origin"] = self.base.validation["forecast_origin"]
        with self.assertRaisesRegex(ValueError, "holdout"):
            run_tuning(self.base, TuningSpec.from_mapping(recipe), study_dir=self.root / "bad")
        self.assertFalse((self.root / "bad").exists())

    def test_existing_study_is_never_overwritten(self):
        directory = self.root / "existing"
        directory.mkdir()
        sentinel = directory / "study.json"
        sentinel.write_text("existing result")
        with self.assertRaises(FileExistsError):
            run_tuning(self.base, TuningSpec.from_mapping(search_payload()), study_dir=directory)
        self.assertEqual(sentinel.read_text(), "existing result")

    def test_failed_trials_are_recorded_without_fake_scores(self):
        recipe = search_payload()
        recipe["parameters"] = {"estimator.params.invalid_parameter": {"type": "categorical", "choices": [1]}}
        directory = self.root / "failed"
        with self.assertRaisesRegex(RuntimeError, "no successful trials"):
            run_tuning(self.base, TuningSpec.from_mapping(recipe), study_dir=directory)
        report = json.loads((directory / "study.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["trials"]), 2)
        self.assertTrue(all(trial["status"] == "failed" and trial["score"] is None for trial in report["trials"]))
        self.assertTrue(all("invalid_parameter" in trial["error"] for trial in report["trials"]))
        self.assertFalse((directory / "best.yaml").exists())

    def test_tuning_cli_executes_lightgbm_and_rejects_duplicate_recipe_keys(self):
        base = replace(self.base, estimator=EstimatorSpec(
            model_type="lightgbm", target_adapter="independent",
            params={"n_estimators": 3, "verbosity": -1, "min_child_samples": 2, "random_state": 0},
        ))
        base_path, search_path = self.root / "base.yaml", self.root / "search.yaml"
        base_path.write_text(yaml.safe_dump(base.canonical_payload()))
        recipe = search_payload()
        recipe["parameters"] = {"estimator.params.num_leaves": {"type": "int", "low": 3, "high": 5}}
        search_path.write_text(yaml.safe_dump(recipe))
        command = [sys.executable, "tune.py", "--config-yaml", str(base_path),
                   "--search-yaml", str(search_path), "--study-dir", str(self.root / "lgbm")]
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                   capture_output=True, text=True, timeout=90)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        report = json.loads((self.root / "lgbm/study.json").read_text())
        validate_artifacts(report["final_artifacts"])
        search_path.write_text(search_path.read_text() + "seed: 99\n")
        command[-1] = str(self.root / "duplicate")
        invalid = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                 capture_output=True, text=True, timeout=30)
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("Duplicate YAML key", invalid.stderr)
        self.assertFalse((self.root / "duplicate").exists())


if __name__ == "__main__":
    unittest.main()
