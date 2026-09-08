"""原始数据绘图不得聚合、遗漏月份或修改输入。"""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import matplotlib.image as mpimg
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "config/hongtaiyang_cesuan"))

from prepare import SOURCES
from plot_raw_data import build_raw_figures, plot_raw_data


class HongtaiyangVisualizationTest(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2025-01-01", "2026-01-01", freq="15min", inclusive="left")
        self.frame = pd.DataFrame({"time": times, "value": np.arange(len(times), dtype=float) % 96})

    def test_every_original_point_is_plotted(self):
        before = self.frame.copy(deep=True)
        figures = build_raw_figures(self.frame, title="Test load", color="#2563eb")
        try:
            self.assertEqual(len(figures[0].axes), 1)
            self.assertEqual(len(figures[1].axes), 12)
            annual = figures[0].axes[0].lines[0]
            np.testing.assert_array_equal(annual.get_ydata(), self.frame.value)
            np.testing.assert_array_equal(annual.get_xdata(), self.frame.time.to_numpy())
            monthly_values = np.concatenate([axis.lines[0].get_ydata() for axis in figures[1].axes])
            monthly_times = np.concatenate([axis.lines[0].get_xdata() for axis in figures[1].axes])
            np.testing.assert_array_equal(monthly_values, self.frame.value)
            np.testing.assert_array_equal(monthly_times, self.frame.time.to_numpy())
            for month, axis in enumerate(figures[1].axes, start=1):
                self.assertTrue(all(mdates.num2date(tick).month == month for tick in axis.get_xticks()))
            pd.testing.assert_frame_equal(self.frame, before)
        finally:
            for figure in figures:
                figure.clear()

    def test_three_sources_six_readable_pngs_and_unchanged_csvs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = {}
            for site, target in SOURCES:
                path = root / site / f"{target}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                self.frame.to_csv(path, index=False)
                hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
            outputs = plot_raw_data(root)
            self.assertEqual(len(set(outputs)), 6)
            for path in outputs:
                self.assertEqual(path.parent.name, "visualization")
                self.assertEqual(path.suffix, ".png")
                image = mpimg.imread(path)
                self.assertGreater(image.shape[0], 100)
                self.assertGreater(float(np.ptp(image[:, :, :3])), 0)
            for path, digest in hashes.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_invalid_source_fails_before_any_plot_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for site, target in SOURCES:
                path = root / site / f"{target}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                frame = self.frame.iloc[:-1] if target == "pv_load" else self.frame
                frame.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                plot_raw_data(root)
            self.assertEqual(list(root.rglob("*.png")), [])

    def test_standalone_cli(self):
        result = subprocess.run([str(ROOT / ".venv/bin/python"),
                                 str(ROOT / "config/hongtaiyang_cesuan/plot_raw_data.py"), "--help"],
                                cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertIn("--data-root", result.stdout)


if __name__ == "__main__":
    unittest.main()
