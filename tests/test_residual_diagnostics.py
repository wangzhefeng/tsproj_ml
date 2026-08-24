# -*- coding: utf-8 -*-
"""B1 残差频谱诊断测试。"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from decomposition.residual_diagnostics import (
    diagnose_window_residual,
    summarize_window_residuals,
    write_residual_diagnostics,
)


def _make_pure_noise(n=200, seed=0):
    rng = np.random.RandomState(seed)
    return rng.normal(0.0, 1.0, n)


def _make_periodic_residual(n=200, period=12, amp=3.0, seed=1):
    rng = np.random.RandomState(seed)
    x = np.arange(n, dtype=float)
    return amp * np.sin(2.0 * np.pi * x / period) + rng.normal(0.0, 0.2, n)


class DiagnoseWindowTest(unittest.TestCase):
    def test_pure_noise_no_stable_period(self):
        residual = _make_pure_noise()
        row = diagnose_window_residual(residual, window_idx=0)
        self.assertIsNotNone(row.fft_dominant_period_samples)
        # 白噪声的 FFT 主导周期不应是强信号（amplitude 相对小）
        self.assertLess(row.fft_dominant_amplitude, 100.0)

    def test_periodic_residual_detects_period(self):
        residual = _make_periodic_residual(period=12)
        row = diagnose_window_residual(residual, window_idx=0)
        # FFT 主导周期应接近 12
        self.assertIsNotNone(row.fft_dominant_period_samples)
        self.assertAlmostEqual(row.fft_dominant_period_samples, 12.0, delta=1.0)
        # ACF 应在 lag 12 附近有显著峰
        acf_periods = [lag for lag, _ in row.acf_top_periods]
        self.assertIn(12, acf_periods)

    def test_short_series_graceful(self):
        row = diagnose_window_residual(np.array([1.0, 2.0, 1.0]), window_idx=0)
        self.assertEqual(row.n_obs, 3)
        self.assertIsNone(row.fft_dominant_period_samples)


class SummarizeTest(unittest.TestCase):
    def test_stable_band_detected(self):
        # 所有窗口都有 period=12 的强周期 → stable_band_detected=True
        rows = [
            diagnose_window_residual(_make_periodic_residual(period=12, seed=i), window_idx=i)
            for i in range(5)
        ]
        summary = summarize_window_residuals(rows)
        self.assertTrue(summary.stable_band_detected)
        self.assertAlmostEqual(summary.fft_period_median, 12.0, delta=1.0)
        self.assertLess(summary.fft_period_cv, 0.3)

    def test_unstable_band_not_detected(self):
        # 窗口间周期强漂移（period 12 vs 30）→ CV 高 → not stable
        rows = [
            diagnose_window_residual(_make_periodic_residual(period=p, seed=i), window_idx=i)
            for i, p in enumerate([12, 30, 12, 30, 12])
        ]
        summary = summarize_window_residuals(rows)
        # ACF 显著但 CV 高 → stable_band_detected=False
        self.assertFalse(summary.stable_band_detected)

    def test_pure_noise_not_detected(self):
        rows = [
            diagnose_window_residual(_make_pure_noise(seed=i), window_idx=i)
            for i in range(5)
        ]
        summary = summarize_window_residuals(rows)
        # 白噪声 ACF 弱 → not stable
        self.assertFalse(summary.stable_band_detected)


class WriteCsvTest(unittest.TestCase):
    def test_csv_output_structure(self):
        rows = [
            diagnose_window_residual(_make_periodic_residual(period=12), window_idx=0),
            diagnose_window_residual(_make_periodic_residual(period=12, seed=2), window_idx=1),
        ]
        summary = summarize_window_residuals(rows)
        with tempfile.TemporaryDirectory() as tmp:
            out = write_residual_diagnostics(summary, Path(tmp) / "residual_diagnostics.csv")
            self.assertTrue(out.exists())
            import pandas as pd

            df = pd.read_csv(out)
            self.assertEqual(len(df), 3)  # 2 windows + 1 summary
            self.assertIn("window_idx", df.columns)
            self.assertIn("fft_dominant_period_samples", df.columns)


if __name__ == "__main__":
    unittest.main()
