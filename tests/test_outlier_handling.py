# -*- coding: utf-8 -*-

import unittest
from types import SimpleNamespace

import pandas as pd

from data_process.outlier_handling import (
    TRAIN_OUTLIER_REPORT_COLUMNS,
    handle_train_outliers,
)


class TrainOutlierHandlingTest(unittest.TestCase):
    def _base_frame(self, values):
        return pd.DataFrame(
            {
                "time": pd.date_range("2026-06-01 00:00:00", periods=len(values), freq="5min"),
                "y": values,
            }
        )

    def test_default_disabled_keeps_training_data_unchanged(self):
        df = self._base_frame([1000.0, 20000.0, 1000.0])
        args = SimpleNamespace(enable_train_outlier_handling=False)

        cleaned, report = handle_train_outliers(
            args=args,
            df_history_train=df,
            target_feature="y",
            window=1,
            log_prefix="[test]",
        )

        pd.testing.assert_frame_equal(cleaned, df)
        self.assertTrue(report.empty)
        self.assertEqual(list(report.columns), TRAIN_OUTLIER_REPORT_COLUMNS)

    def test_high_short_run_is_interpolated_and_reported(self):
        df = self._base_frame([1000.0, 1100.0, 20000.0, 21000.0, 1200.0, 1300.0])
        args = SimpleNamespace(
            enable_train_outlier_handling=True,
            train_outlier_method="local_interpolate",
            high_outlier_threshold=15000.0,
            high_outlier_max_run_points=4,
            drop_outlier_max_run_points=0,
            drop_rebound_min_abs_diff=900.0,
        )

        cleaned, report = handle_train_outliers(
            args=args,
            df_history_train=df,
            target_feature="y",
            window=3,
            log_prefix="[test]",
        )

        self.assertEqual(report["outlier_type"].tolist(), ["high_short_run", "high_short_run"])
        self.assertEqual(report["window"].tolist(), [3, 3])
        self.assertEqual(report["raw_y"].tolist(), [20000.0, 21000.0])
        self.assertLess(cleaned.loc[2, "y"], 1200.0)
        self.assertGreater(cleaned.loc[3, "y"], 1100.0)
        self.assertNotEqual(cleaned.loc[2, "y"], df.loc[2, "y"])
        self.assertNotEqual(cleaned.loc[3, "y"], df.loc[3, "y"])

    def test_drop_rebound_is_interpolated_and_reported_without_time_column(self):
        df = pd.DataFrame({"y": [5000.0, 5100.0, 3500.0, 3600.0, 5200.0, 5300.0]})
        args = SimpleNamespace(
            enable_train_outlier_handling=True,
            train_outlier_method="local_interpolate",
            high_outlier_threshold=15000.0,
            high_outlier_max_run_points=0,
            drop_outlier_max_run_points=2,
            drop_rebound_min_abs_diff=900.0,
        )

        cleaned, report = handle_train_outliers(
            args=args,
            df_history_train=df,
            target_feature="y",
            window=5,
            log_prefix="[test]",
        )

        self.assertEqual(report["outlier_type"].tolist(), ["drop_rebound", "drop_rebound"])
        self.assertEqual(report["time"].tolist(), [2, 3])
        self.assertEqual(report["raw_y"].tolist(), [3500.0, 3600.0])
        self.assertGreater(cleaned.loc[2, "y"], df.loc[2, "y"])
        self.assertGreater(cleaned.loc[3, "y"], df.loc[3, "y"])

    def test_low_short_run_is_interpolated_and_reported(self):
        df = self._base_frame([1000.0, 0.0, 0.0, 1100.0])
        args = SimpleNamespace(
            enable_train_outlier_handling=True,
            train_outlier_method="local_interpolate",
            high_outlier_threshold=15000.0,
            high_outlier_max_run_points=0,
            drop_outlier_max_run_points=0,
            drop_rebound_min_abs_diff=900.0,
            low_outlier_threshold=100.0,
            low_outlier_max_run_points=4,
        )

        cleaned, report = handle_train_outliers(
            args=args,
            df_history_train=df,
            target_feature="y",
            window=7,
            log_prefix="[test]",
        )

        self.assertEqual(report["outlier_type"].tolist(), ["low_short_run", "low_short_run"])
        self.assertEqual(report["raw_y"].tolist(), [0.0, 0.0])
        self.assertGreater(cleaned.loc[1, "y"], df.loc[1, "y"])
        self.assertGreater(cleaned.loc[2, "y"], df.loc[2, "y"])

    def test_low_rule_disabled_when_threshold_none(self):
        # low_outlier_threshold 未传 -> 走默认 None -> 低值规则不触发，数据保持不变
        df = self._base_frame([1000.0, 0.0, 0.0, 1100.0])
        args = SimpleNamespace(
            enable_train_outlier_handling=True,
            train_outlier_method="local_interpolate",
            high_outlier_threshold=15000.0,
            high_outlier_max_run_points=0,
            drop_outlier_max_run_points=0,
            drop_rebound_min_abs_diff=900.0,
        )

        cleaned, report = handle_train_outliers(
            args=args,
            df_history_train=df,
            target_feature="y",
            window=8,
            log_prefix="[test]",
        )

        self.assertTrue(report.empty)
        pd.testing.assert_frame_equal(cleaned, df)

    def test_rise_rebound_is_interpolated_and_reported(self):
        df = self._base_frame([5000.0, 5100.0, 6500.0, 6600.0, 5200.0, 5300.0])
        args = SimpleNamespace(
            enable_train_outlier_handling=True,
            train_outlier_method="local_interpolate",
            high_outlier_threshold=15000.0,
            high_outlier_max_run_points=0,
            drop_outlier_max_run_points=0,
            drop_rebound_min_abs_diff=900.0,
            rise_outlier_max_run_points=2,
            rise_rebound_min_abs_diff=900.0,
        )

        cleaned, report = handle_train_outliers(
            args=args,
            df_history_train=df,
            target_feature="y",
            window=9,
            log_prefix="[test]",
        )

        self.assertEqual(report["outlier_type"].tolist(), ["rise_rebound", "rise_rebound"])
        self.assertEqual(report["raw_y"].tolist(), [6500.0, 6600.0])
        self.assertLess(cleaned.loc[2, "y"], df.loc[2, "y"])
        self.assertLess(cleaned.loc[3, "y"], df.loc[3, "y"])


if __name__ == "__main__":
    unittest.main()
