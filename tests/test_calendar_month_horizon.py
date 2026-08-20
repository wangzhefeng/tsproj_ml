# -*- coding: utf-8 -*-

import tempfile
import unittest
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from config.config_loader import load_yaml_config
from main import Model
from models.ModelTesting import Tester


ROOT = Path(__file__).resolve().parent.parent


class CalendarMonthHorizonTest(unittest.TestCase):
    def _calendar_month_config(self, temp_dir: Path):
        cfg = load_yaml_config(
            ROOT
            / "config/aidc_power_month/route_A/freq_1day/add_decomposition/"
            / "lgbm_usmd_mean_prob_horizon_decomp_linear.yaml"
        )
        cfg.horizon_mode = "calendar_month"
        cfg.train_window_length = 120
        cfg.checkpoints_dir = temp_dir / "models"
        cfg.test_results_dir = temp_dir / "test"
        cfg.pred_results_dir = temp_dir / "forecast"
        return cfg

    def test_model_calendar_month_resolves_complete_august_horizon(self):
        with tempfile.TemporaryDirectory() as td:
            model: Any = Model(self._calendar_month_config(Path(td)))

        self.assertEqual(model.horizon_mode, "calendar_month")
        self.assertEqual(model.horizon, 31)
        self.assertEqual(model.train_window_len, 120)
        self.assertEqual(model.forecast_start_time, pd.Timestamp("2026-08-01"))
        self.assertEqual(model.forecast_end_time, pd.Timestamp("2026-09-01"))
        self.assertEqual(model.n_windows, 6)
        self.assertTrue(model.setting.endswith("-calendar-month"))

    def test_calendar_month_folds_are_month_aligned_with_fixed_training_rows(self):
        df_history = pd.DataFrame(
            {
                "time": pd.date_range("2025-10-01", "2026-07-31", freq="1D"),
                "y": range(304),
            }
        )

        fold_builder = getattr(Tester, "_build_calendar_month_folds")
        folds = fold_builder(
            df_history,
            train_window_len=120,
        )

        self.assertEqual(len(folds), 6)
        self.assertEqual([fold["horizon"] for fold in folds], [31, 30, 31, 30, 31, 28])
        self.assertEqual(
            [(fold["test_start_time"], fold["test_end_time"]) for fold in folds],
            [
                (pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-01")),
                (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-07-01")),
                (pd.Timestamp("2026-05-01"), pd.Timestamp("2026-06-01")),
                (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-05-01")),
                (pd.Timestamp("2026-03-01"), pd.Timestamp("2026-04-01")),
                (pd.Timestamp("2026-02-01"), pd.Timestamp("2026-03-01")),
            ],
        )
        self.assertTrue(all(fold["train_end"] - fold["train_start"] == 120 for fold in folds))
        self.assertTrue(all(fold["test_end"] - fold["test_start"] == fold["horizon"] for fold in folds))

    def test_calendar_month_score_records_month_length_and_total_energy_error(self):
        y_test = np.full(28, 10.0)
        y_pred = np.full(28, 9.0)
        df_history_test = pd.DataFrame(
            {"time": pd.date_range("2026-02-01", "2026-02-28", freq="1D")}
        )

        evaluate_score: Any = Tester._evaluate_score
        scores = evaluate_score(
            y_test,
            y_pred,
            window=1,
            df_history_test=df_history_test,
            log_prefix="[test]",
            horizon_mode="calendar_month",
        )

        self.assertEqual(scores.loc[1, "Calendar Month"], "2026-02")
        self.assertEqual(scores.loc[1, "Forecast Steps"], len(y_test))
        self.assertEqual(scores.loc[1, "Monthly Actual Total"], float(y_test.sum()))
        self.assertEqual(scores.loc[1, "Monthly Predicted Total"], float(y_pred.sum()))
        self.assertAlmostEqual(
            scores.loc[1, "Monthly Total MAPE"],
            abs(float(y_test.sum() - y_pred.sum())) / abs(float(y_test.sum())),
        )


if __name__ == "__main__":
    unittest.main()
