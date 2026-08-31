# -*- coding: utf-8 -*-

import unittest
from pathlib import Path

import pandas as pd

from config.config_loader import load_yaml_config
from model_testing.validation import calendar_month_folds


ROOT = Path(__file__).resolve().parent.parent


class CalendarMonthHorizonTest(unittest.TestCase):
    def _calendar_month_config(self):
        return load_yaml_config(
            ROOT
            / "config/aidc_power_month/route_A/freq_1day/add_decomposition/"
            / "lgbm_usmd_mean_prob_horizon_decomp_linear.yaml"
        )

    def test_calendar_month_config_declares_complete_august_horizon(self):
        cfg = self._calendar_month_config()

        self.assertEqual(cfg.validation["horizon_mode"], "calendar_month")
        self.assertEqual(cfg.problem.horizon, 31)
        self.assertEqual(cfg.validation["train_window_days"], 120)
        forecast_start = pd.Timestamp(cfg.validation["forecast_origin"]).floor("1D") + pd.Timedelta(days=1)
        forecast_end = forecast_start + pd.offsets.MonthBegin(1)
        self.assertEqual(forecast_start, pd.Timestamp("2026-08-01"))
        self.assertEqual(forecast_end, pd.Timestamp("2026-09-01"))
        self.assertEqual(cfg.output["setting_suffix"], "-horizon-decomp-linear")

    def test_calendar_month_folds_are_month_aligned_with_fixed_training_rows(self):
        df_history = pd.DataFrame(
            {
                "time": pd.date_range("2025-10-01", "2026-07-31", freq="1D"),
                "y": range(304),
            }
        )

        folds = calendar_month_folds(
            df_history["time"],
            train_window_days=120,
            fold_count=6,
            stride_months=1,
        )

        self.assertEqual(len(folds), 6)
        self.assertEqual([fold.horizon for fold in folds], [28, 31, 30, 31, 30, 31])
        self.assertTrue(
            all(len(fold.train_indices) == 120 for fold in folds)
        )
        self.assertTrue(
            all(len(fold.forecast_times) == fold.horizon for fold in folds)
        )


if __name__ == "__main__":
    unittest.main()
