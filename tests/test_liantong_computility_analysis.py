"""算力关联分析：合成关系、滞后方向、缺口和输出保护。"""
import importlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "config/aidc_electricity_computility/electricity/2026-08-31/liantong_IT"
sys.path.insert(0, str(SCRIPT_DIR))


class ComputilityAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("liantong_computility_analysis")

    def fixture(self, size=900):
        random = np.random.default_rng(73)
        x = random.uniform(1, 10, size)
        frame = pd.DataFrame({name: x for name in self.module.feature_names()})
        frame.insert(0, "value", np.r_[np.zeros(6), x[:-6]])
        frame.insert(0, "time", pd.date_range("2026-08-01", periods=size, freq="5min"))
        raw = frame[["time", "value"]].copy()
        raw.loc[100, "value"] = np.nan
        return frame, raw

    def test_correlation_constant_and_ties(self):
        stats = self.module.correlation(pd.Series([1, 1, 2, 3]), pd.Series([2, 2, 4, 6]))
        self.assertAlmostEqual(stats["spearman"], 1)
        stats = self.module.correlation(pd.Series([0, 0, 0]), pd.Series([1, 2, 3]))
        self.assertEqual(stats["status"], "constant")
        self.assertTrue(np.isnan(stats["pearson"]))

    def test_lag_direction_and_difference_gap(self):
        frame, raw = self.fixture()
        tables = self.module.analyze(frame, raw.value.notna())
        feature = self.module.feature_names()[0]
        lag = tables["lag_correlations"].query("feature == @feature and lag_steps == 6").iloc[0]
        self.assertAlmostEqual(lag.pearson, 1)
        self.assertEqual(lag.n, len(frame) - 6 - 1)
        diff = tables["feature_summary"].set_index("feature").loc[feature]
        self.assertEqual(diff.diff_n, len(frame) - 3)  # 差分两端均不得是补值标签
        self.assertEqual(len(tables["feature_summary"]), 54)
        self.assertEqual(len(tables["lag_correlations"]), 54 * len(self.module.LAGS))
        self.assertFalse(tables["daily_correlations"].duplicated(["feature", "date", "lag_steps"]).any())
        self.assertEqual(set(tables["daily_correlations"].lag_steps), {0, 288, 576})
        day = tables["lag_correlations"].query("feature == @feature and lag_steps == 288").iloc[0]
        self.assertGreater(day.valid_days, 0)

    def test_load_rejects_invalid_data(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            frame, raw = self.fixture()
            source, power = root / "merged.csv", root / "raw.csv"
            raw.to_csv(power, index=False)
            variants = []
            bad = frame.copy(); bad.loc[1, "time"] = bad.loc[0, "time"]; variants.append(bad)
            bad = frame.copy(); bad.loc[1, "value"] += 1; variants.append(bad)
            bad = frame.copy(); bad.iloc[1, 2] = np.inf; variants.append(bad)
            bad = frame.copy(); bad["training_unknown"] = 1; variants.append(bad)
            for bad in variants:
                bad.to_csv(source, index=False)
                with self.assertRaises(ValueError):
                    self.module.load_inputs(source, power)

    def test_cli_outputs_repeat_and_protection(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            frame, raw = self.fixture()
            source, power = root / "merged.csv", root / "raw.csv"
            frame.to_csv(source, index=False); raw.to_csv(power, index=False)
            before = [source.read_bytes(), power.read_bytes()]
            output = root / "analysis"
            command = [sys.executable, str(SCRIPT_DIR / "liantong_computility_analysis.py"),
                       "--input-csv", str(source), "--raw-power-csv", str(power), "--output-dir", str(output)]
            result = subprocess.run(command, cwd=temp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            audit = json.loads((output / "analysis_audit.json").read_text())
            self.assertEqual(audit["features"], 54)
            self.assertEqual(audit["excluded_target_rows"], 1)
            self.assertFalse(audit["forecast_improvement_verified"])
            snapshot = {p.name: p.read_bytes() for p in output.iterdir()}
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(subprocess.run(command + ["--overwrite"], capture_output=True).returncode, 0)
            self.assertEqual(snapshot, {p.name: p.read_bytes() for p in output.iterdir()})
            self.assertEqual(before, [source.read_bytes(), power.read_bytes()])
            self.assertIn("因果", (output / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
