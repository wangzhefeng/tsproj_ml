# -*- coding: utf-8 -*-

import unittest

import numpy as np
import pandas as pd

from decomposition.periods import detect_periodicity


class PeriodicityAnalysisTest(unittest.TestCase):

    @staticmethod
    def _frame(n, y):
        return pd.DataFrame(
            {"time": pd.date_range("2026-01-01", periods=n, freq="1h"), "y": y}
        )

    def test_exact_two_cycles_is_accepted(self):
        period = 12
        n = 2 * period  # 恰好两个完整周期
        x = np.arange(n, dtype=float)
        y = 50.0 + 8.0 * np.sin(2.0 * np.pi * x / period)
        report = detect_periodicity(
            self._frame(n, y), time_col="time", target_col="y", seasonal_period=period
        )

        self.assertTrue(report["stl_has_seasonal_component"])
        self.assertEqual(report["stl_seasonal_period_used"], period)

    def test_standard_strength_metrics_output(self):
        period = 12
        n = period * 6
        x = np.arange(n, dtype=float)
        y = 50.0 + 8.0 * np.sin(2.0 * np.pi * x / period) + 0.05 * x
        report = detect_periodicity(
            self._frame(n, y), time_col="time", target_col="y", seasonal_period=period
        )

        # 标准季节/趋势强度按 FPP 公式：max(0, 1 - Var(R)/Var(S+R)) / max(0, 1 - Var(R)/Var(T+R))
        self.assertIn("stl_seasonal_strength", report)
        self.assertIn("stl_trend_strength", report)
        # 强季节信号：季节强度应显著大于 0
        self.assertGreater(report["stl_seasonal_strength"], 0.5)
        # legacy 指标保留
        self.assertIn("stl_seasonal_ratio", report)

    def test_no_seasonality_metrics_are_zero_bounded(self):
        rng = np.random.default_rng(42)
        period = 12
        n = period * 6
        y = 50.0 + rng.normal(0.0, 1.0, n)
        report = detect_periodicity(
            self._frame(n, y), time_col="time", target_col="y", seasonal_period=period
        )

        # 白噪声：标准强度公式带 max(0, ...) 下界
        self.assertGreaterEqual(report["stl_seasonal_strength"], 0.0)
        self.assertLess(report["stl_seasonal_strength"], 0.5)


if __name__ == "__main__":
    unittest.main()
