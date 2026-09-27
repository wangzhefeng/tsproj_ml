"""sliding_window / expanding_window 回测形态：spec 合同、几何、窗口分派、产物与端到端 smoke。"""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_core.specs import (
    ExpandingWindowBacktestSpec,
    SlidingWindowBacktestSpec,
)
from forecasting_core.specs.validation import RuntimeValidationSpec
from model_testing.artifacts.reporting import write_backtest_results
from model_testing.contracts import geometry as backtest_geometry
from model_testing.contracts.windows import rolling_backtest_windows
from pipeline.runner import CanonicalBaseModelRunner, run_canonical_config
from data_loading import SourceRegistry
from tests import test_canonical_runtime_smoke as smoke


def _parse(validation):
    return RuntimeValidationSpec.from_mapping(
        validation, source="test", require_geometry=True
    ).backtest


class BacktestModeSpecContractTest(unittest.TestCase):
    """horizon_mode=sliding_window/expanding_window 的配置合同。"""

    def test_sliding_window_spec_accepted(self):
        spec = _parse({
            "horizon_mode": "sliding_window",
            "history_steps": 100, "train_window_steps": 50,
            "fold_count": 5, "stride_steps": 1,
        })
        self.assertIsInstance(spec, SlidingWindowBacktestSpec)

    def test_expanding_window_spec_accepted(self):
        spec = _parse({
            "horizon_mode": "expanding_window",
            "history_steps": 100, "fold_count": 5, "stride_steps": 2,
        })
        self.assertIsInstance(spec, ExpandingWindowBacktestSpec)

    def test_expanding_forbids_train_window_and_raw_history(self):
        for extra in ({"train_window_steps": 50}, {"train_history_steps": 20}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                _parse({"horizon_mode": "expanding_window", "history_steps": 100,
                        "fold_count": 5, "stride_steps": 2, **extra})

    def test_sliding_forbids_raw_history_and_calendar_fields(self):
        for extra in ({"train_history_steps": 20}, {"train_window_days": 30}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                _parse({"horizon_mode": "sliding_window", "history_steps": 100,
                        "train_window_steps": 50, "fold_count": 5, "stride_steps": 1, **extra})

    def test_missing_geometry_fields_raise(self):
        with self.assertRaises(ValueError):
            _parse({"horizon_mode": "expanding_window", "history_steps": 100, "stride_steps": 2})
        with self.assertRaises(ValueError):
            _parse({"horizon_mode": "sliding_window", "history_steps": 100,
                    "fold_count": 5, "stride_steps": 1})

    def test_unknown_horizon_mode_raises(self):
        with self.assertRaises(ValueError):
            _parse({"horizon_mode": "weekly", "history_steps": 10,
                    "train_window_steps": 5, "fold_count": 2, "stride_steps": 1})

    def test_spec_dataclass_validation(self):
        with self.assertRaises(ValueError):
            SlidingWindowBacktestSpec(history_steps=10, train_window_steps=10,
                                      fold_count=2, stride_steps=1)
        with self.assertRaises(ValueError):
            ExpandingWindowBacktestSpec(history_steps=10, fold_count=0, stride_steps=1)


class RollingModeGeometryTest(unittest.TestCase):
    """rolling_origin_folds 的 expanding/sliding 几何。"""

    def setUp(self):
        self.origins = tuple(pd.date_range("2026-01-01", periods=24, freq="1h"))
        self.geometry = backtest_geometry.TimeGeometry(
            offset=pd.tseries.frequencies.to_offset("1h"), horizon=2,
        )

    def test_expanding_folds_grow_training_set(self):
        folds = backtest_geometry.rolling_origin_folds(
            self.origins, self.geometry,
            history_steps=None, train_window_steps=None, fold_count=3, stride_steps=2,
        )
        sizes = [len(fold.train_indices) for fold in folds]
        self.assertEqual(sizes, sorted(sizes))
        self.assertLess(sizes[0], sizes[-1])

    def test_fixed_truncates_but_expanding_does_not(self):
        fixed = backtest_geometry.rolling_origin_folds(
            self.origins, self.geometry, history_steps=None,
            train_window_steps=5, fold_count=3, stride_steps=2,
        )
        expanding = backtest_geometry.rolling_origin_folds(
            self.origins, self.geometry, history_steps=None,
            train_window_steps=None, fold_count=3, stride_steps=2,
        )
        self.assertTrue(all(len(fold.train_indices) == 5 for fold in fixed))
        self.assertGreater(len(expanding[-1].train_indices), 5)

    def test_sliding_folds_overlap(self):
        folds = backtest_geometry.rolling_origin_folds(
            self.origins, self.geometry, history_steps=None,
            train_window_steps=5, fold_count=3, stride_steps=1,
        )
        for prev, nxt in zip(folds, folds[1:]):
            self.assertGreaterEqual(
                self.geometry.label_end(prev.origin),
                self.geometry.label_start(nxt.origin),
            )

    def test_invalid_train_window_steps_raises(self):
        with self.assertRaises(ValueError):
            backtest_geometry.rolling_origin_folds(
                self.origins, self.geometry, history_steps=None,
                train_window_steps=0, fold_count=2, stride_steps=1,
            )


class RollingWindowDispatchTest(unittest.TestCase):
    """contracts/windows.py 的 spec 分派与 sliding 重叠校验。"""

    def windows(self, spec, horizon=2):
        origins = tuple(pd.date_range("2026-01-01", periods=24, freq="1h"))
        return rolling_backtest_windows(
            origins,
            offset=pd.tseries.frequencies.to_offset("1h"),
            horizon=horizon,
            backtest=spec,
        )

    def test_sliding_requires_overlap_stride(self):
        with self.assertRaisesRegex(ValueError, "stride_steps < horizon"):
            self.windows(SlidingWindowBacktestSpec(
                history_steps=20, train_window_steps=5, fold_count=2, stride_steps=2,
            ), horizon=2)

    def test_expanding_windows_have_growing_train_sets(self):
        folds = self.windows(ExpandingWindowBacktestSpec(
            history_steps=20, fold_count=3, stride_steps=2,
        ))
        sizes = [len(fold.train_indices) for fold in folds]
        self.assertLess(sizes[0], sizes[-1])


class StitchOverviewContractTest(unittest.TestCase):
    """stitch_overview=False（sliding 重叠折）跳过拼接总图。"""

    def frame(self, *, overlap: bool):
        rows = []
        times = (
            [pd.date_range("2026-01-01", periods=2, freq="1h")] * 2
            if overlap
            else [pd.date_range("2026-01-01", periods=2, freq="1h"),
                  pd.date_range("2026-01-01 02:00", periods=2, freq="1h")]
        )
        for window, window_times in ((1, times[0]), (2, times[1])):
            for i, t in enumerate(window_times):
                rows.append({"series_id": "s", "time": t, "target": "load",
                             "actual_value": float(i), "predict_value": float(i),
                             "window": window, "plot_valid": True})
        return pd.DataFrame(rows)

    def scores(self):
        return pd.DataFrame({"window": [1, 2], "scope": ["target", "target"],
                             "metric": ["mae", "mae"], "value": [0.0, 0.0]})

    def test_stitch_disabled_skips_overview_plot(self):
        with tempfile.TemporaryDirectory() as d:
            write_backtest_results(d, self.frame(overlap=True), self.scores(),
                                   aggregate_weighting={"load": 1.0}, stitch_overview=False)
            root = Path(d)
            self.assertFalse((root / "test_prediction.png").exists())
            self.assertTrue((root / "cv_plot_df.csv").exists())
            self.assertTrue((root / "windows_results" / "window_01.png").exists())

    def test_stitch_enabled_keeps_overview_plot(self):
        with tempfile.TemporaryDirectory() as d:
            write_backtest_results(d, self.frame(overlap=False), self.scores(),
                                   aggregate_weighting={"load": 1.0})
            self.assertTrue((Path(d) / "test_prediction.png").exists())


class WindowModeSmokeTest(unittest.TestCase):
    """合成数据端到端：sliding 重叠折不拼总图，expanding 训练集逐折扩大。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "target.csv"
        times = pd.date_range("2026-01-01", periods=52, freq="1h")
        pd.DataFrame({"time": times, "load": 100 + np.arange(52, dtype=float)}).to_csv(
            self.path, index=False
        )

    def make_config(self, validation):
        base = smoke.CanonicalRuntimeSmokeTest().build_config(self.path, mode="point")
        return replace(base, validation=validation)

    def test_sliding_window_end_to_end(self):
        config = self.make_config({
            "forecast_origin": "2026-01-02T23:00:00",
            "horizon_mode": "sliding_window",
            "history_steps": 40, "train_window_steps": 10,
            "fold_count": 3, "stride_steps": 1, "seasonal_naive_lag": 2,
        })
        result = run_canonical_config(config, output_root=self.root / "sliding", backtest_only=True)
        self.assertFalse((result.test_dir / "test_prediction.png").exists())
        self.assertTrue((result.test_dir / "windows_results" / "window_01.png").exists())
        frame = pd.read_csv(result.test_dir / "cv_plot_df.csv")
        self.assertEqual(sorted(frame["window"].unique()), [1, 2, 3])
        metadata = json.loads((result.test_dir / "result_metadata.json").read_text())
        self.assertEqual(metadata["backtest"]["mode"], "sliding_window")
        self.assertEqual(metadata["backtest"]["train_window_steps"], 10)

    def test_expanding_window_end_to_end(self):
        config = self.make_config({
            "forecast_origin": "2026-01-02T23:00:00",
            "horizon_mode": "expanding_window",
            "history_steps": 40, "fold_count": 3, "stride_steps": 2,
            "seasonal_naive_lag": 2,
        })
        result = run_canonical_config(config, output_root=self.root / "expanding", backtest_only=True)
        self.assertTrue((result.test_dir / "test_prediction.png").exists())
        metadata = json.loads((result.test_dir / "result_metadata.json").read_text())
        self.assertEqual(metadata["backtest"]["mode"], "expanding_window")
        self.assertNotIn("train_window_steps", metadata["backtest"])
        sizes = [w["training_sample_count"] for w in metadata["backtest"]["windows"]]
        self.assertEqual(sizes, sorted(sizes))
        self.assertLess(sizes[0], sizes[-1])

    def test_expanding_final_fit_rejected(self):
        config = self.make_config({
            "forecast_origin": "2026-01-02T23:00:00",
            "horizon_mode": "expanding_window",
            "history_steps": 40, "fold_count": 3, "stride_steps": 2,
            "seasonal_naive_lag": 2,
        })
        runner = CanonicalBaseModelRunner(
            config, SourceRegistry(config.data, self.root), pd.Timestamp("2026-01-02T23:00:00"),
        )
        with self.assertRaisesRegex(ValueError, "backtest-only"):
            runner.final_bundle_inputs()


if __name__ == "__main__":
    unittest.main()
