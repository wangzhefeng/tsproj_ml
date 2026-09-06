"""通用周期分析不依赖目标分解运行时；诊断表不借列存汇总值。"""
import subprocess
import sys
import unittest
import numpy as np
from decomposition.diagnostics.residuals import diagnose_window_residual, summarize_window_residuals


class AnalysisBoundaryTest(unittest.TestCase):
    def test_period_analysis_import_does_not_load_decomposition(self):
        result = subprocess.run([sys.executable, "-c", "import sys; from timeseries_analysis.periods import fft_dominant_period; assert 'decomposition' not in sys.modules"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_residual_summary_has_explicit_columns(self):
        from decomposition.diagnostics.residuals import residual_diagnostics_frame
        residual = np.sin(2 * np.pi * np.arange(240) / 24)
        summary = summarize_window_residuals([diagnose_window_residual(residual, 0), diagnose_window_residual(residual, 1)])
        table = residual_diagnostics_frame(summary)
        self.assertEqual(table["record_type"].tolist(), ["window", "window", "summary"])
        self.assertTrue(summary.stable_band_detected)
        self.assertEqual(table.iloc[-1]["fft_period_cv"], summary.fft_period_cv)
        self.assertTrue(np.isnan(table.iloc[-1]["n_obs"]))
        self.assertTrue(np.isnan(table.iloc[-1]["fft_dominant_amplitude"]))


if __name__ == "__main__":
    unittest.main()
