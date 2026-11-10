# -*- coding: utf-8 -*-
"""calendar_month origin_day（月中发报原点）合同与运行时行为。

origin_day 将每折原点从「目标月开始前一天」移动到「上一个月的 origin_day 日」：
runner 需先穿越 lead-1 个不评分的 gap 步，评分切片只保留目标自然月。
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_core.specs import CalendarMonthBacktestSpec
from forecasting_core.specs.config import parse_model_config
from model_pipeline.runner import run_canonical_config
from model_testing.geometry import calendar_month_folds

ROOT = Path(__file__).resolve().parents[1]
TIMES = pd.date_range("2025-10-01", "2026-07-31", freq="1D")


def _payload(history_path: str, **validation_extra):
    return {
        "schema_version": 2,
        "problem": {
            "time_col": "time",
            "freq": "1D",
            "horizon": 37,
            "targets": ["load"],
            "training_scope": "local",
            "series_id_cols": [],
        },
        "data": {
            "sources": [
                {
                    "name": "target_history",
                    "source_type": "file",
                    "columns": [
                        {"name": "load", "role": "target", "categorical": False}
                    ],
                    "history_path": history_path,
                    "time_col": "time",
                    "series_id_cols": [],
                    "availability": "source_time",
                }
            ]
        },
        "features": {
            "target_lags": {"load": [1, 7]},
            "observed_past_lags": {},
            "datetime_features": [],
            "transformations": {},
        },
        "strategy": {"name": "recursive"},
        "estimator": {
            "model_type": "ridge",
            "target_adapter": "independent",
            "params": {"alpha": 1e-8},
        },
        "probabilistic": {"mode": "point"},
        "validation": {
            "forecast_origin": "2026-07-31T00:00:00",
            "schedule_mode": "daily",
            "horizon_mode": "calendar_month",
            "train_window_days": 120,
            "fold_count": 5,
            "stride_months": 1,
            **validation_extra,
        },
        "output": {"scenario_subpath": "origin-day-test"},
    }


class CalendarMonthOriginDayGeometryTest(unittest.TestCase):
    def test_origin_day_moves_origin_and_keeps_month_scoring(self):
        folds = calendar_month_folds(
            TIMES,
            train_window_days=120,
            fold_count=5,
            stride_months=1,
            origin_day=25,
        )
        self.assertEqual(
            [fold.origin.date().isoformat() for fold in folds],
            ["2026-02-25", "2026-03-25", "2026-04-25", "2026-05-25", "2026-06-25"],
        )
        self.assertEqual([fold.horizon for fold in folds], [31, 30, 31, 30, 31])
        # lead = 上月天数 - origin_day + 1；runner 跨度 = lead - 1 + 当月天数
        self.assertEqual([fold.lead for fold in folds], [4, 7, 6, 7, 6])
        self.assertEqual(
            [fold.lead - 1 + fold.horizon for fold in folds], [34, 36, 36, 36, 36]
        )
        for fold in folds:
            self.assertEqual(len(fold.train_indices), 120)
            self.assertEqual(fold.train_indices[-1], fold.origin_index)
            self.assertEqual(fold.train_indices[0], fold.origin_index + 1 - 120)
            self.assertEqual(fold.forecast_times[0].day, 1)
            self.assertEqual(fold.metadata["lead_steps"], fold.lead)

    def test_default_origin_day_keeps_month_end_origin(self):
        folds = calendar_month_folds(
            TIMES, train_window_days=120, fold_count=6, stride_months=1
        )
        self.assertEqual([fold.lead for fold in folds], [1] * 6)
        self.assertEqual(
            [fold.origin.date().isoformat() for fold in folds],
            [
                "2026-01-31",
                "2026-02-28",
                "2026-03-31",
                "2026-04-30",
                "2026-05-31",
                "2026-06-30",
            ],
        )
        for fold in folds:
            self.assertEqual(fold.train_indices[-1], fold.origin_index)
            self.assertNotIn("origin_day", fold.metadata)

    def test_origin_day_rejects_invalid_values_and_incomplete_folds(self):
        for invalid in (0, 29, True, "25"):
            with self.assertRaises(ValueError):
                calendar_month_folds(
                    TIMES,
                    train_window_days=120,
                    fold_count=1,
                    stride_months=1,
                    origin_day=invalid,
                )
        # 2 月折原点 2026-01-25 前仅 117 天，不足 120：完整折从 3 月起，只有 5 折。
        folds = calendar_month_folds(
            TIMES,
            train_window_days=120,
            fold_count=6,
            stride_months=1,
            origin_day=25,
        )
        self.assertEqual(len(folds), 5)


class CalendarMonthOriginDaySpecTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        data_path = Path(self._tmp.name) / "daily.csv"
        pd.DataFrame({"time": TIMES, "load": 100.0 + np.arange(len(TIMES))}).to_csv(
            data_path, index=False
        )
        self.history_path = str(data_path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_origin_day_roundtrip_and_validation(self):
        spec = parse_model_config(
            _payload(self.history_path, origin_day=25), source="<origin-day-spec>"
        )
        backtest = spec.validation.backtest
        self.assertIsInstance(backtest, CalendarMonthBacktestSpec)
        self.assertEqual(backtest.origin_day, 25)
        self.assertEqual(backtest.train_window_days, 120)
        self.assertEqual(backtest.stride_months, 1)
        self.assertEqual(spec.validation["origin_day"], 25)

    def test_origin_day_bounds_and_fingerprint(self):
        for invalid in (0, 29, True):
            with self.assertRaises(ValueError):
                parse_model_config(
                    _payload(self.history_path, origin_day=invalid),
                    source="<origin-day-spec>",
                )
        base = parse_model_config(_payload(self.history_path), source="<origin-day-spec>")
        shifted = parse_model_config(
            _payload(self.history_path, origin_day=25), source="<origin-day-spec>"
        )
        self.assertNotEqual(base.fingerprint(), shifted.fingerprint())

    def test_fixed_steps_forbids_origin_day(self):
        payload = _payload(self.history_path)
        payload["validation"] = {
            "forecast_origin": "2026-07-31T00:00:00",
            "schedule_mode": "daily",
            "horizon_mode": "fixed_steps",
            "fold_count": 5,
            "stride_steps": 37,
            "origin_day": 25,
            "training_window": {"kind": "rolling", "history_steps": 120},
        }
        with self.assertRaises(ValueError):
            parse_model_config(payload, source="<origin-day-spec>")


class CalendarMonthOriginDayRuntimeTest(unittest.TestCase):
    def test_origin_day_backtest_scores_only_target_months(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_path = root / "daily.csv"
            pd.DataFrame(
                {"time": TIMES, "load": 100.0 + np.arange(len(TIMES))}
            ).to_csv(data_path, index=False)
            config = parse_model_config(
                _payload(str(data_path), origin_day=25), source="<origin-day-runtime>"
            )
            result = run_canonical_config(config, output_root=root / "results")

            cv = pd.read_csv(result.test_dir / "cv_plot_df.csv")
            self.assertEqual(
                cv.groupby("window")["time"].nunique().tolist(),
                [31, 30, 31, 30, 31],
            )
            for _, frame in cv.groupby("window"):
                times = pd.to_datetime(frame["time"])
                self.assertEqual(times.dt.to_period("M").nunique(), 1)
                self.assertEqual(times.min().day, 1)
                self.assertEqual(times.max().day, times.max().days_in_month)

            metadata = json.loads(
                (result.test_dir / "result_metadata.json").read_text(encoding="utf-8")
            )
            windows = metadata["backtest"]["windows"]
            self.assertEqual(
                [window["origin"][:10] for window in windows],
                ["2026-02-25", "2026-03-25", "2026-04-25", "2026-05-25", "2026-06-25"],
            )
            self.assertEqual(
                [window["lead_steps"] for window in windows], [4, 7, 6, 7, 6]
            )
            for window in windows:
                self.assertLessEqual(
                    window["training_label_end_max"][:10], window["origin"][:10]
                )

            prediction = pd.read_csv(result.forecast_dir / "prediction.csv")
            forecast_times = pd.to_datetime(prediction["time"])
            # 最终预测保持 origin+1..origin+H 全跨度（37 步），月内交付按当月筛选。
            self.assertEqual(len(forecast_times), 37)
            self.assertEqual(forecast_times.min(), pd.Timestamp("2026-08-01"))
            self.assertEqual(forecast_times.max(), pd.Timestamp("2026-09-06"))


if __name__ == "__main__":
    unittest.main()
