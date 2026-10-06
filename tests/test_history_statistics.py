"""独立数值内核合同：解析期校验由编译器负责，内核不读取信息集。"""
import unittest

import numpy as np
import pandas as pd

from feature_engineering.kernels.history import history_statistic, rolling_statistics, time_since_event


class HistoryStatisticsTest(unittest.TestCase):
    def test_scalar_statistics_match_independent_numeric_definitions(self):
        values = np.asarray([1.0, 2.0, 4.0, 8.0])
        probabilities = values / values.sum()
        expected = {
            "mean": np.mean(values), "std": np.std(values, ddof=1),
            "min": np.min(values), "max": np.max(values),
            "median": np.median(values),
            "entropy": -np.sum(probabilities * np.log2(probabilities)),
            "max_diff": np.max(np.diff(values)), "min_diff": np.min(np.diff(values)),
        }
        for stat, value in expected.items():
            with self.subTest(stat=stat):
                self.assertAlmostEqual(history_statistic(pd.Series(values), stat), value)

    def test_rolling_uses_only_trailing_values_and_preserves_index(self):
        series = pd.Series([1.0, 2.0, 4.0, 8.0], index=pd.date_range("2026-01-01", periods=4, freq="h"))
        result = rolling_statistics(series, 2, ("mean", "std", "max_diff", "min_diff"))
        for stat, actual in result.items():
            expected = []
            for position in range(len(series)):
                window = series.to_numpy()[max(0, position - 1):position + 1]
                if stat == "mean":
                    value = np.mean(window)
                elif stat == "std":
                    value = np.std(window, ddof=1) if len(window) > 1 else 0.0
                else:
                    value = np.diff(window)[0] if len(window) > 1 else 0.0
                expected.append(value)
            pd.testing.assert_index_equal(actual.index, series.index)
            np.testing.assert_allclose(actual, expected, rtol=1e-14)

    def test_event_requires_a_confirmed_turning_point(self):
        # 最后一项不是已确认峰值，须有右侧观测才能确认。
        values = pd.Series([1.0, 4.0, 2.0, 8.0])
        self.assertEqual(time_since_event(values, "peak"), float(len(values) - 1 - 1))
        self.assertEqual(time_since_event(values, "trough"), float(len(values) - 1 - 2))
        monotonic = pd.Series([1.0, 2.0, 3.0])
        self.assertEqual(time_since_event(monotonic, "peak"), float(len(monotonic) - 1))

    def test_invalid_stat_and_event_raise_and_small_windows_warn(self):
        series = pd.Series([1.0])
        with self.assertRaisesRegex(ValueError, "unsupported history statistic"):
            history_statistic(series, "unknown")
        with self.assertRaisesRegex(ValueError, "unsupported time-since event"):
            time_since_event(series, "unknown")
        for stat in ("std", "max_diff", "min_diff"):
            with self.subTest(stat=stat), self.assertRaisesRegex(ValueError, "samples"):
                history_statistic(series, stat)


if __name__ == "__main__":
    unittest.main()
