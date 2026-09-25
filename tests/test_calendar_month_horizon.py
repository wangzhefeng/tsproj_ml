# -*- coding: utf-8 -*-

import unittest

import pandas as pd

from model_testing.geometry import calendar_month_folds


class CalendarMonthHorizonTest(unittest.TestCase):
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
