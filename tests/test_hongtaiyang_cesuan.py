"""红太阳年度回测的边界、数据与真实模型测试。"""
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "config/hongtaiyang_cesuan"))

from prepare import validate_frame, aggregate_daily
from generate_configs import model_document, STRATEGIES
from annual_backtest import forecast_window, schedule, assemble_year
from annual_reporting import write_annual_results
from forecasting_core.specs.config import parse_model_config


class HongtaiyangTest(unittest.TestCase):
    def test_standalone_entrypoint(self):
        completed = subprocess.run([str(ROOT / ".venv/bin/python"),
                                    str(ROOT / "config/hongtaiyang_cesuan/annual_backtest.py"), "--help"],
                                   cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertIn("--config-yaml", completed.stdout)

    def test_real_matrix_and_daily_aggregation(self):
        paths = sorted((ROOT / "config/hongtaiyang_cesuan").glob("*/*/freq_*/lgbm_*.yaml"))
        self.assertEqual(len(paths), 20)
        for site in ("xinnengyuan", "guangdianchang"):
            raw = pd.read_csv(ROOT / "dataset/hongtaiyang_cesuan" / site / "demand_load.csv")
            daily = aggregate_daily(raw)
            np.testing.assert_allclose(daily.value, raw.value.to_numpy().reshape(-1, 96).mean(axis=1))
            self.assertEqual(len(daily), 365)

    def test_full_schedule(self):
        daily = schedule("1D")
        intraday = schedule("15min")
        self.assertEqual(len(daily), 11)
        self.assertEqual(len(intraday), 334)
        self.assertEqual(len(daily[0][2]), 28)
        self.assertEqual(len(daily[1][2]), 31)
        self.assertEqual(daily[0][0], pd.Timestamp("2025-01-02"))
        self.assertEqual(daily[1][0], pd.Timestamp("2025-01-01"))
        self.assertEqual(intraday[0][0], pd.Timestamp("2025-01-02"))
        self.assertEqual(intraday[-1][2][-1], pd.Timestamp("2025-12-31 23:45"))

    def test_bad_data_rejected(self):
        frame = pd.DataFrame({"time": pd.date_range("2025-01-01", "2026-01-01", freq="15min", inclusive="left"), "value": 1.0})
        validate_frame(frame, "15min")
        with self.assertRaises(ValueError):
            validate_frame(frame.iloc[:-1], "15min")
        frame.loc[1, "value"] = np.nan
        with self.assertRaises(ValueError):
            validate_frame(frame, "15min")

    def test_all_configs(self):
        for daily in (False, True):
            identities = []
            for strategy in STRATEGIES:
                cfg = parse_model_config(model_document("xinnengyuan", "demand_load", daily, strategy), source="test")
                self.assertEqual(cfg.strategy.name.value, "direct" if strategy.startswith("direct") else strategy)
                self.assertEqual(cfg.problem.freq, "1D" if daily else "15min")
                identities.append(cfg.fingerprint())
            self.assertEqual(len(set(identities)), 4)

    def test_february_short_sample_and_no_future_leakage(self):
        times = pd.date_range("2025-01-01", "2026-01-01", freq="1D", inclusive="left")
        frame = pd.DataFrame({"time": times, "value": 100 + np.arange(len(times)) % 7 * 5.0})
        doc = model_document("xinnengyuan", "demand_load", True, "direct")
        doc["estimator"]["params"]["n_estimators"] = 3
        cfg = parse_model_config(doc, source="test")
        window = schedule("1D")[0]
        prediction, audit = forecast_window(cfg, frame, window)
        changed = frame.copy()
        changed.loc[changed.time >= "2025-02-01", "value"] = 1e9
        other, _ = forecast_window(cfg, changed, window)
        np.testing.assert_array_equal(prediction.y_pred, other.y_pred)
        self.assertEqual(len(prediction), 28)
        self.assertTrue(np.isfinite(prediction.y_pred).all())
        self.assertEqual(audit["effective_strategy"], "calendar_baseline")
        self.assertEqual(audit["model_count"], 0)
        changed.loc[changed.time < "2025-01-02", "value"] = 2e9
        for strategy in ("mimo", "direct-pointwise", "direct-pointwise-horizon"):
            doc = model_document("xinnengyuan", "demand_load", True, strategy)
            doc["estimator"]["params"]["n_estimators"] = 3
            other, _ = forecast_window(parse_model_config(doc, source="test"), changed, window)
            np.testing.assert_array_equal(prediction.y_pred, other.y_pred)

    def test_march_three_strategies(self):
        times = pd.date_range("2025-01-01", "2026-01-01", freq="1D", inclusive="left")
        frame = pd.DataFrame({"time": times, "value": 100 + np.arange(len(times)) % 7 * 5.0})
        for strategy in STRATEGIES:
            doc = model_document("xinnengyuan", "demand_load", True, strategy)
            doc["estimator"]["params"]["n_estimators"] = 2
            prediction, audit = forecast_window(parse_model_config(doc, source="test"), frame, schedule("1D")[1])
            self.assertEqual(len(prediction), 31)
            self.assertEqual(audit["effective_strategy"], "direct" if strategy.startswith("direct") else strategy)
            if strategy.startswith("direct-pointwise"):
                self.assertEqual(audit["model_count"], 1)
                self.assertEqual("forecast_horizon_idx" in audit["feature_schema"], strategy.endswith("-horizon"))
            self.assertTrue(np.isfinite(prediction.y_pred).all())

    def test_canonical_annual_reporting(self):
        times = pd.date_range("2025-01-01", "2026-01-01", freq="1D", inclusive="left")
        annual = pd.DataFrame({"time": times, "y_true": 10., "y_pred": 12.})
        annual.loc[annual.time < "2025-02-01", "y_pred"] = 10.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            scores = write_annual_results(output, annual, "1D", {})
            self.assertEqual(scores["rows"], len(annual))
            self.assertAlmostEqual(scores["MAE"], np.abs(annual.y_true - annual.y_pred).mean())
            self.assertEqual(len(list(output.glob("**/*.png"))), 13)
            saved = pd.read_csv(output / "cv_plot_df.csv")
            self.assertEqual(len(saved), len(annual))
            self.assertEqual(set(saved.window), set(range(12)))

    def test_annual_join_rejects_missing_or_duplicate(self):
        times = pd.date_range("2025-01-01", "2026-01-01", freq="1D", inclusive="left")
        actual = pd.DataFrame({"time": times, "value": np.arange(len(times), dtype=float)})
        predicted = pd.DataFrame({"time": times[times.month > 1], "y_pred": 7.0})
        annual = assemble_year(actual, [predicted], "1D")
        np.testing.assert_array_equal(annual.y_pred.iloc[:31], annual.y_true.iloc[:31])
        with self.assertRaises(ValueError):
            assemble_year(actual, [predicted.iloc[:-1]], "1D")
        with self.assertRaises(ValueError):
            assemble_year(actual, [predicted, predicted], "1D")


if __name__ == "__main__":
    unittest.main()
