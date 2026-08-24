# -*- coding: utf-8 -*-
"""因果 prequential CQR 校准样本选择测试。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
import pandas as pd

from probabilistic.calibration import (
    CalibrationRecord,
    calibrate_with_records,
    select_calibration_records,
)
from probabilistic.evaluation import (
    append_final_calibration_report,
    calibrate_final_from_cv,
    evaluate_prequential_cqr,
)


class CalibrationRecordSelectionTest(unittest.TestCase):
    @staticmethod
    def _record(window, origin, target, available, score):
        return CalibrationRecord(
            forecast_origin=pd.Timestamp(origin),
            target_time=pd.Timestamp(target),
            label_available_at=pd.Timestamp(available),
            horizon_step=1,
            window=window,
            interval_name="q10_q90",
            lower=0.0,
            upper=1.0,
            y_true=0.5,
            score=score,
            stage="processed",
        )

    def test_selector_uses_only_latest_complete_prior_windows_and_deduplicates_targets(self):
        records = [
            self._record(3, "2026-01-01", "2026-01-02", "2026-01-02", 1.0),
            self._record(3, "2026-01-01", "2026-01-03", "2026-01-03", 9.0),
            self._record(2, "2026-01-03", "2026-01-03", "2026-01-03", 2.0),
            self._record(2, "2026-01-03", "2026-01-04", "2026-01-04", 3.0),
            # 当前窗口即使首个标签已到达，也因整个窗口尚未完成而不能自校准。
            self._record(1, "2026-01-05", "2026-01-05", "2026-01-05", 4.0),
            self._record(1, "2026-01-05", "2026-01-06", "2026-01-06", 5.0),
        ]

        selected = select_calibration_records(
            records,
            forecast_origin=pd.Timestamp("2026-01-05"),
            n_windows=2,
        )

        self.assertEqual({record.window for record in selected}, {2, 3})
        self.assertEqual(
            [record.target_time for record in selected],
            [pd.Timestamp("2026-01-03"), pd.Timestamp("2026-01-04"), pd.Timestamp("2026-01-02")],
        )
        # 重复 target_time=2026-01-03 保留较新 origin 的 window=2 记录。
        self.assertEqual(selected[0].score, 2.0)

    def test_calibration_requires_both_minimum_windows_and_scores(self):
        records = [
            self._record(3, "2026-01-01", "2026-01-02", "2026-01-02", 1.0),
            self._record(3, "2026-01-01", "2026-01-03", "2026-01-03", 1.0),
            self._record(2, "2026-01-03", "2026-01-04", "2026-01-04", 1.0),
            self._record(2, "2026-01-03", "2026-01-05", "2026-01-05", 1.0),
        ]
        lower = np.array([10.0])
        upper = np.array([20.0])

        insufficient_windows = calibrate_with_records(
            lower,
            upper,
            records,
            forecast_origin=pd.Timestamp("2026-01-03"),
            n_windows=2,
            min_windows=2,
            min_scores=2,
            alpha=0.1,
        )
        self.assertEqual(insufficient_windows.status, "insufficient_windows")

        insufficient_scores = calibrate_with_records(
            lower,
            upper,
            records,
            forecast_origin=pd.Timestamp("2026-01-06"),
            n_windows=2,
            min_windows=2,
            min_scores=5,
            alpha=0.1,
        )
        self.assertEqual(insufficient_scores.status, "insufficient_scores")

        applied = calibrate_with_records(
            lower,
            upper,
            records,
            forecast_origin=pd.Timestamp("2026-01-06"),
            n_windows=2,
            min_windows=2,
            min_scores=4,
            alpha=0.1,
        )
        self.assertEqual(applied.status, "applied")
        self.assertEqual(applied.selected_windows, 2)
        self.assertEqual(applied.selected_scores, 4)
        self.assertEqual(applied.correction, 1.0)
        np.testing.assert_array_equal(applied.lower, [9.0])
        np.testing.assert_array_equal(applied.upper, [21.0])

    def test_prequential_evaluator_never_uses_current_or_future_scores(self):
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(
                    [
                        "2026-01-05", "2026-01-06",
                        "2026-01-03", "2026-01-04",
                        "2026-01-01", "2026-01-02",
                    ]
                ),
                "window": [1, 1, 2, 2, 3, 3],
                "Y_trues": [15.0, 15.0, 15.0, 15.0, 2.0, 2.0],
                "predict_q10": [10.0, 10.0, 10.0, 10.0, 0.0, 0.0],
                "predict_q50": [15.0, 15.0, 15.0, 15.0, 0.5, 0.5],
                "predict_q90": [20.0, 20.0, 20.0, 20.0, 1.0, 1.0],
                "conformal_score": [-5.0, -5.0, -5.0, -5.0, 1.0, 1.0],
            }
        )

        intervals, report = evaluate_prequential_cqr(
            frame,
            freq="1D",
            calibration_windows=1,
            min_windows=1,
            min_scores=2,
            alpha=0.1,
            label_availability_delay_steps=0,
        )

        status_by_window = report.set_index("window")["status"].to_dict()
        correction_by_window = report.set_index("window")["correction"].to_dict()
        self.assertEqual(status_by_window[3], "insufficient_windows")
        self.assertEqual(status_by_window[2], "applied")
        self.assertEqual(status_by_window[1], "applied")
        self.assertEqual(correction_by_window[2], 1.0)
        # window 1 只能用 window 2 的负 score；no-shrink 后 correction=0。
        self.assertEqual(correction_by_window[1], 0.0)
        self.assertEqual(set(intervals["window"]), {1, 2})
        self.assertEqual(len(intervals), 4)

    def test_final_calibration_selects_latest_origin_not_largest_window_id(self):
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(
                    [
                        "2026-01-05", "2026-01-06",
                        "2026-01-03", "2026-01-04",
                        "2026-01-01", "2026-01-02",
                    ]
                ),
                "window": [1, 1, 2, 2, 3, 3],
                "Y_trues": [0.5] * 6,
                "predict_q10": [0.0] * 6,
                "predict_q50": [0.5] * 6,
                "predict_q90": [1.0] * 6,
                # 最新 origin 的 window=1 为负 score，最旧 window=3 为正 score。
                "conformal_score": [-2.0, -2.0, 3.0, 3.0, 9.0, 9.0],
            }
        )

        result = calibrate_final_from_cv(
            frame,
            q_low=np.array([10.0]),
            q_high=np.array([20.0]),
            forecast_origin=pd.Timestamp("2026-01-07"),
            freq="1D",
            calibration_windows=1,
            min_windows=1,
            min_scores=2,
            alpha=0.1,
            label_availability_delay_steps=0,
        )

        self.assertEqual(result.status, "applied")
        self.assertEqual({record.window for record in result.selected_records}, {1})
        self.assertEqual(result.correction, 0.0)

    def test_final_calibration_uses_explicit_base_interval_not_grid_extremes(self):
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "window": [1, 1],
                "Y_trues": [23.0, 23.0],
                "predict_q05": [0.0, 0.0],
                "predict_q10": [10.0, 10.0],
                "predict_q50": [15.0, 15.0],
                "predict_q90": [20.0, 20.0],
                "predict_q95": [30.0, 30.0],
                "conformal_score": [3.0, 3.0],
            }
        )

        result = calibrate_final_from_cv(
            frame,
            q_low=np.array([100.0]),
            q_high=np.array([200.0]),
            forecast_origin=pd.Timestamp("2026-01-03"),
            freq="1D",
            calibration_windows=1,
            min_windows=1,
            min_scores=2,
            alpha=0.2,
            interval_name="central80",
            lower_quantile=0.1,
            upper_quantile=0.9,
        )

        self.assertEqual(result.correction, 3.0)
        self.assertTrue(
            all(record.interval_name == "central80" for record in result.selected_records)
        )
        self.assertTrue(all(record.lower == 10.0 for record in result.selected_records))
        self.assertTrue(all(record.upper == 20.0 for record in result.selected_records))

    def test_prequential_calibration_uses_explicit_base_interval(self):
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(
                    ["2026-01-03", "2026-01-04", "2026-01-01", "2026-01-02"]
                ),
                "window": [1, 1, 2, 2],
                "Y_trues": [23.0] * 4,
                "predict_q05": [0.0] * 4,
                "predict_q10": [10.0] * 4,
                "predict_q50": [15.0] * 4,
                "predict_q90": [20.0] * 4,
                "predict_q95": [30.0] * 4,
                "conformal_score": [3.0] * 4,
            }
        )

        intervals, report = evaluate_prequential_cqr(
            frame,
            freq="1D",
            calibration_windows=1,
            min_windows=1,
            min_scores=2,
            alpha=0.2,
            interval_name="central80",
            lower_quantile=0.1,
            upper_quantile=0.9,
        )

        self.assertEqual(set(report["interval_name"]), {"central80"})
        self.assertEqual(set(intervals["base_lower_quantile"]), {0.1})
        self.assertEqual(set(intervals["base_upper_quantile"]), {0.9})

    def test_final_calibration_allows_explicit_interval_shrink(self):
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "window": [1, 1],
                "Y_trues": [15.0, 15.0],
                "predict_q10": [10.0, 10.0],
                "predict_q50": [15.0, 15.0],
                "predict_q90": [20.0, 20.0],
                "conformal_score": [-2.0, -2.0],
            }
        )

        result = calibrate_final_from_cv(
            frame,
            q_low=np.array([10.0]),
            q_high=np.array([20.0]),
            forecast_origin=pd.Timestamp("2026-01-03"),
            freq="1D",
            calibration_windows=1,
            min_windows=1,
            min_scores=2,
            alpha=0.1,
            allow_interval_shrink=True,
        )

        self.assertEqual(result.correction, -2.0)
        np.testing.assert_array_equal(result.lower, [12.0])
        np.testing.assert_array_equal(result.upper, [18.0])

    def test_prequential_calibration_propagates_allow_shrink_to_report(self):
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(
                    ["2026-01-03", "2026-01-04", "2026-01-01", "2026-01-02"]
                ),
                "window": [1, 1, 2, 2],
                "Y_trues": [15.0] * 4,
                "predict_q10": [10.0] * 4,
                "predict_q50": [15.0] * 4,
                "predict_q90": [20.0] * 4,
                "conformal_score": [-2.0] * 4,
            }
        )

        _, report = evaluate_prequential_cqr(
            frame,
            freq="1D",
            calibration_windows=1,
            min_windows=1,
            min_scores=2,
            alpha=0.1,
            allow_interval_shrink=True,
        )

        applied = report.loc[report["status"] == "applied"]
        self.assertTrue(bool(applied["allow_interval_shrink"].all()))
        self.assertEqual(set(applied["correction"]), {-2.0})

    def test_final_calibration_is_appended_to_shared_audit_report(self):
        records = [
            self._record(1, "2026-01-01", "2026-01-02", "2026-01-02", 1.0),
            self._record(1, "2026-01-01", "2026-01-03", "2026-01-03", 1.0),
        ]
        result = calibrate_with_records(
            np.array([10.0]),
            np.array([20.0]),
            records,
            forecast_origin=cast(pd.Timestamp, pd.Timestamp("2026-01-04")),
            n_windows=5,
            min_windows=1,
            min_scores=2,
            alpha=0.1,
        )

        with TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir, "calibration_report.csv")
            pd.DataFrame(
                [{"scope": "prequential", "forecast_origin": "2026-01-03"}]
            ).to_csv(report_path, index=False)
            append_final_calibration_report(
                report_path,
                result=result,
                forecast_origin=pd.Timestamp("2026-01-04"),
                interval_name="q10_q90",
                target_coverage=0.9,
                calibration_windows=5,
                allow_interval_shrink=True,
            )

            report = pd.read_csv(report_path)
            final_row = report[report["scope"] == "final"].iloc[0]
            self.assertEqual(final_row["selected_windows"], 1)
            self.assertEqual(final_row["n_windows"], 5)
            self.assertEqual(final_row["n_scores"], 2)
            self.assertEqual(final_row["status"], "applied")
            self.assertTrue(bool(final_row["allow_interval_shrink"]))
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("2026-01-03T00:00:00", report_text)
            self.assertIn("2026-01-04T00:00:00", report_text)
            self.assertNotIn("2026-01-04 00:00:00", report_text)


if __name__ == "__main__":
    unittest.main()
