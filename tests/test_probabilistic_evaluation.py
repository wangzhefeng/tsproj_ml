# -*- coding: utf-8 -*-
"""概率评估长表与 horizon 聚合测试。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import pandas as pd

from probabilistic.evaluation import (
    build_probabilistic_artifacts,
    write_probabilistic_artifacts,
)
from probabilistic.spec import IntervalSpec


class ProbabilisticArtifactTest(unittest.TestCase):
    @staticmethod
    def _frame():
        return pd.DataFrame(
            {
                "time": pd.to_datetime(
                    ["2026-01-03", "2026-01-04", "2026-01-01", "2026-01-02"]
                ),
                "window": [1, 1, 2, 2],
                "Y_trues": [0.0, 2.0, 1.0, 3.0],
                "Y_preds": [0.0, 2.0, 1.0, 3.0],
                "predict_q10_raw": [1.0, 1.0, 1.0, 2.0],
                "predict_q10": [1.0, 1.0, 1.0, 2.0],
                "predict_q50_raw": [0.0, 2.0, 1.0, 3.0],
                "predict_q50": [0.0, 2.0, 1.0, 3.0],
                "predict_q90_raw": [3.0, 3.0, 3.0, 4.0],
                "predict_q90": [3.0, 3.0, 3.0, 4.0],
                "conformal_score": [-5.0, -5.0, 1.0, 1.0],
            }
        )

    def test_artifacts_include_horizon_step_and_window_counts(self):
        frame = self._frame()

        predictions, scores, horizon_scores = build_probabilistic_artifacts(frame)

        for _, group in predictions.groupby("window"):
            self.assertEqual(sorted(group["horizon_step"].unique().tolist()), [1, 2])
        window_q10 = scores[
            (scores["record_kind"] == "quantile")
            & (scores["stage"] == "processed")
            & (scores["metric"] == "pinball")
            & (scores["quantile"] == 0.1)
            & (scores["window"] == 1)
        ].iloc[0]
        self.assertAlmostEqual(window_q10["value"], 0.5)
        horizon_q10 = horizon_scores[
            (horizon_scores["record_kind"] == "quantile")
            & (horizon_scores["metric"] == "pinball")
            & (horizon_scores["quantile"] == 0.1)
            & (horizon_scores["horizon_step"] == 1)
        ].iloc[0]
        self.assertAlmostEqual(horizon_q10["value"], 0.45)
        self.assertEqual(horizon_q10["n_points"], 2)
        self.assertEqual(horizon_q10["n_windows"], 2)

    def test_horizon_aggregation_covers_point_mae_mape_crossing_and_naive_skill(self):
        frame = self._frame()
        # 昨日同时刻 naive：比 Y_trues 恒差 1.0，skill 可手算
        frame["Y_naive"] = [1.0, 3.0, 2.0, 4.0]

        _, _, horizon_scores = build_probabilistic_artifacts(frame)

        point_rows = cast(
            pd.DataFrame, horizon_scores[horizon_scores["record_kind"] == "point"]
        )
        metrics = set(point_rows["metric"].unique())
        self.assertIn("mae", metrics)
        self.assertIn("mape", metrics)
        crossing_metrics = {
            metric
            for metric in horizon_scores["metric"].unique()
            if "crossing" in str(metric)
        }
        self.assertTrue(crossing_metrics)

        # horizon 1: Y_trues=[0,1], Y_preds=[0,1], Y_naive=[1,2] → MAE=0, MAPE=0
        h1_mae = point_rows[
            (point_rows["metric"] == "mae") & (point_rows["horizon_step"] == 1)
        ].iloc[0]
        self.assertAlmostEqual(h1_mae["value"], 0.0)
        # naive MAE=1.0 → skill = 1 - 0/1 = 1.0
        h1_skill = point_rows[
            (point_rows["metric"] == "naive_skill") & (point_rows["horizon_step"] == 1)
        ].iloc[0]
        self.assertAlmostEqual(h1_skill["value"], 1.0)
        h1_naive_mae = point_rows[
            (point_rows["metric"] == "naive_mae") & (point_rows["horizon_step"] == 1)
        ].iloc[0]
        self.assertAlmostEqual(h1_naive_mae["value"], 1.0)
        # horizon 2: MAE=0 同理；两步都要有点指标
        self.assertEqual(
            set(point_rows["horizon_step"].unique()), {1, 2}
        )

    def test_horizon_point_metrics_absent_without_naive_column(self):
        frame = self._frame()
        self.assertNotIn("Y_naive", frame.columns)

        _, _, horizon_scores = build_probabilistic_artifacts(frame)

        point_rows = horizon_scores[horizon_scores["record_kind"] == "point"]
        self.assertIn("mae", set(point_rows["metric"].unique()))
        self.assertNotIn("naive_skill", set(point_rows["metric"].unique()))

    def test_writer_creates_probability_artifacts_and_hides_raw_columns_from_legacy_frame(self):
        with TemporaryDirectory() as tmpdir:
            legacy_frame = write_probabilistic_artifacts(self._frame(), Path(tmpdir))

            self.assertTrue(Path(tmpdir, "probabilistic_predictions_df.csv").exists())
            self.assertTrue(Path(tmpdir, "probabilistic_scores_df.csv").exists())
            self.assertTrue(Path(tmpdir, "horizon_scores_df.csv").exists())
            self.assertNotIn("predict_q10_raw", legacy_frame.columns)
            self.assertIn("predict_q10", legacy_frame.columns)

    def test_raw_crossing_is_diagnosed_without_being_scored_as_an_interval(self):
        frame = self._frame()
        frame.loc[0, "predict_q10_raw"] = 5.0
        frame.loc[0, "predict_q90_raw"] = 3.0

        _, scores, _ = build_probabilistic_artifacts(frame)

        raw_interval_rows = scores[
            (scores["record_kind"] == "interval")
            & (scores["stage"] == "raw")
        ]
        processed_interval_rows = scores[
            (scores["record_kind"] == "interval")
            & (scores["stage"] == "processed")
        ]
        self.assertTrue(raw_interval_rows.empty)
        self.assertFalse(processed_interval_rows.empty)

    def test_base_metrics_cover_every_declared_interval_not_only_grid_extremes(self):
        frame = self._frame()
        frame["predict_q05"] = frame["predict_q10"] - 1.0
        frame["predict_q95"] = frame["predict_q90"] + 1.0

        _, scores, horizon_scores = build_probabilistic_artifacts(
            frame,
            interval_specs=(
                IntervalSpec("central80", 0.1, 0.9),
                IntervalSpec("central90", 0.05, 0.95),
            ),
        )

        interval_names = set(
            scores.loc[scores["record_kind"] == "interval", "interval_name"]
        )
        horizon_interval_names = set(
            horizon_scores.loc[
                horizon_scores["record_kind"] == "interval", "interval_name"
            ]
        )
        self.assertEqual(interval_names, {"central80", "central90"})
        self.assertEqual(horizon_interval_names, {"central80", "central90"})

    def test_unrepaired_processed_crossing_skips_interval_metrics_without_crashing(self):
        frame = self._frame()
        frame.loc[0, "predict_q10"] = 5.0
        frame.loc[0, "predict_q90"] = 3.0

        _, scores, horizon_scores = build_probabilistic_artifacts(frame)

        window_one_interval_rows = scores[
            (scores["record_kind"] == "interval")
            & (scores["window"] == 1)
        ]
        horizon_one_interval_rows = horizon_scores[
            (horizon_scores["record_kind"] == "interval")
            & (horizon_scores["horizon_step"] == 1)
        ]
        self.assertTrue(window_one_interval_rows.empty)
        self.assertTrue(horizon_one_interval_rows.empty)

    def test_writer_adds_prequential_intervals_report_and_calibrated_metrics(self):
        with TemporaryDirectory() as tmpdir:
            write_probabilistic_artifacts(
                self._frame(),
                Path(tmpdir),
                enable_cqr=True,
                freq="1D",
                calibration_windows=1,
                min_windows=1,
                min_scores=2,
                alpha=0.1,
                label_availability_delay_steps=0,
            )

            intervals = pd.read_csv(Path(tmpdir, "probabilistic_intervals_df.csv"))
            report = pd.read_csv(Path(tmpdir, "calibration_report.csv"))
            scores = pd.read_csv(Path(tmpdir, "probabilistic_scores_df.csv"))

            self.assertEqual(set(intervals["window"]), {1})
            self.assertEqual(report.set_index("window").loc[2, "status"], "insufficient_windows")
            self.assertEqual(report.set_index("window").loc[1, "status"], "applied")
            self.assertIn("cqr", set(scores["stage"]))

    def test_writer_propagates_explicit_interval_and_shrink_policy(self):
        frame = self._frame()
        frame["predict_q05"] = frame["predict_q10"] - 10.0
        frame["predict_q95"] = frame["predict_q90"] + 10.0
        with TemporaryDirectory() as tmpdir:
            write_probabilistic_artifacts(
                frame,
                Path(tmpdir),
                enable_cqr=True,
                freq="1D",
                calibration_windows=1,
                min_windows=1,
                min_scores=2,
                alpha=0.1,
                interval_name="central80",
                lower_quantile=0.1,
                upper_quantile=0.9,
                allow_interval_shrink=True,
                interval_specs=(IntervalSpec("central80", 0.1, 0.9),),
            )

            intervals = pd.read_csv(Path(tmpdir, "probabilistic_intervals_df.csv"))
            report = pd.read_csv(Path(tmpdir, "calibration_report.csv"))
            scores = pd.read_csv(Path(tmpdir, "probabilistic_scores_df.csv"))

        self.assertEqual(set(report["interval_name"]), {"central80"})
        self.assertTrue(bool(report["allow_interval_shrink"].all()))
        self.assertEqual(set(intervals["base_lower_quantile"]), {0.1})
        self.assertEqual(set(intervals["base_upper_quantile"]), {0.9})
        base_interval_names = set(
            scores.loc[
                (scores["record_kind"] == "interval")
                & (scores["stage"] == "processed"),
                "interval_name",
            ]
        )
        self.assertEqual(base_interval_names, {"central80"})


if __name__ == "__main__":
    unittest.main()
