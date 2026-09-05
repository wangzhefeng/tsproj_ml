# -*- coding: utf-8 -*-

import math
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd

# AIDC 专项脚本目录（路径含日期段，不是合法 Python 包，按脚本自身约定做 sys.path 引导）
_AIDC_SCRIPTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "config" / "aidc_electricity_computility" / "electricity" / "2026-06-11" / "scripts"
)
if str(_AIDC_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AIDC_SCRIPTS_DIR))

from select_aidc_leadership_days import (  # noqa: E402
    ARCHIVE_SCENARIOS,
    DEFAULT_RESULTS_ROOT,
    FINAL_SUMMARY_COLUMNS,
    REPORT_PLOT_META,
    REPORT_METRICS_COLUMNS,
    SCENARIOS,
    _read_daily_scores,
    build_report_package,
    build_candidate_pool,
    export_report_metrics_summary,
    fill_display_series,
    load_day_curve,
    resolve_report_font,
    run_selection,
)


class SelectAidcLeadershipDaysTest(unittest.TestCase):
    def _make_day_curve(
        self,
        day,
        base_value,
        pred_shift=0.0,
        tail_shift=0.0,
        inject_nan=False,
    ):
        times = pd.date_range(f"{day} 00:00:00", periods=288, freq="5min")
        true_values = []
        pred_values = []
        for i, _ in enumerate(times):
            wave = 25.0 * math.sin(i / 18.0)
            trend = i * 0.2
            y_true = base_value + wave + trend
            y_pred = y_true + pred_shift
            if i >= 240:
                y_pred += tail_shift
            true_values.append(y_true)
            pred_values.append(y_pred)

        if inject_nan:
            true_values[30] = float("nan")
            pred_values[31] = float("nan")

        return [
            {
                "series_id": "__local__",
                "time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "target": "power",
                "actual_value": yt,
                "predict_value": yp,
                "window": 1,
                "plot_valid": True,
            }
            for ts, yt, yp in zip(times, true_values, pred_values)
        ]

    def _write_model_result(self, root, scenario, model_name, score_rows, curve_rows):
        model_dir = Path(root).joinpath(scenario, model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(score_rows).to_csv(model_dir.joinpath("test_scores_df.csv"), index=False)
        pd.DataFrame(curve_rows).to_csv(model_dir.joinpath("cv_plot_df.csv"), index=False)
        model_dir.joinpath("test_prediction.png").write_bytes(b"fake-png")

    def _write_full_fixture(self, root):
        score_defaults = {
            "R2": 0.5,
            "MSE": 100.0,
            "RMSE": 10.0,
            "MAE": 8.0,
            "MAPE": 0.01,
            "MAPE Accuracy": 0.99,
            "MAPE Threshold": 100.0,
            "MAPE Valid Points": 288,
            "MAPE Excluded Points": 0,
            "MAPE Excluded Ratio": 0.0,
        }

        for scenario in SCENARIOS:
            self._write_model_result(
                root,
                scenario,
                "lightgbm-df_power-usmd-15",
                [
                    {
                        "time_range": "2026-05-01 00:00:00~2026-05-01 23:55:00",
                        **score_defaults,
                    },
                    {
                        "time_range": "中位数",
                        **score_defaults,
                    },
                ],
                self._make_day_curve("2026-05-01", 1000.0, pred_shift=3.0),
            )
            self._write_model_result(
                root,
                scenario,
                "lightgbm-df_power-usmdp-2",
                [
                    {
                        "time_range": "2026-05-02 00:00:00~2026-05-02 23:55:00",
                        **score_defaults,
                        "MAPE": 0.011,
                        "MAE": 7.5,
                    },
                    {
                        "time_range": "中位数",
                        **score_defaults,
                    },
                ],
                self._make_day_curve("2026-05-02", 1000.0, pred_shift=1.5, inject_nan=True),
            )

    def _write_report_fixture(self, selection_root, results_root):
        final_dir = Path(selection_root).joinpath("final")
        final_dir.mkdir(parents=True, exist_ok=True)
        candidates_dir = Path(selection_root).joinpath("candidates")

        final_rows = [
            {
                "scenario": "A1_IT",
                "selected_model": "lightgbm-df_power-usmd-15",
                "selected_date": "2026-05-16",
                "time_range": "2026-05-16 00:00:00~2026-05-16 23:55:00",
                "R2": 0.4,
                "MSE": 10.0,
                "RMSE": 3.0,
                "MAE": 2.0,
                "MAPE": 0.01,
                "MAPE Accuracy": 0.99,
                "MAPE Threshold": 100.0,
                "MAPE Valid Points": 288,
                "MAPE Excluded Points": 0,
                "MAPE Excluded Ratio": 0.0,
                "corr": 0.8,
                "tail_MAE": 4.0,
                "tail_bias": -3.0,
                "plot_nan": 0,
                "selection_reason": "least_bad_visual_fit_among_two_models",
                "note": "relative best among two candidate models; overall fit still weak",
            },
            {
                "scenario": "AIDC/route_B",
                "selected_model": "lightgbm-df_power-usmdp-2",
                "selected_date": "2026-04-14",
                "time_range": "2026-04-14 00:00:00~2026-04-14 23:55:00",
                "R2": 0.6,
                "MSE": 20.0,
                "RMSE": 4.0,
                "MAE": 3.0,
                "MAPE": 0.02,
                "MAPE Accuracy": 0.98,
                "MAPE Threshold": 100.0,
                "MAPE Valid Points": 288,
                "MAPE Excluded Points": 0,
                "MAPE Excluded Ratio": 0.0,
                "corr": 0.85,
                "tail_MAE": 5.0,
                "tail_bias": -4.0,
                "plot_nan": 0,
                "selection_reason": "least_bad_visual_fit_among_two_models",
                "note": "relative best among two candidate models; overall fit still weak",
            },
        ]
        pd.DataFrame(final_rows).to_csv(
            final_dir.joinpath("leadership_day_summary.csv"), index=False
        )
        final_dir.joinpath("A1_IT").mkdir(parents=True, exist_ok=True)
        final_dir.joinpath("AIDC/route_B").mkdir(parents=True, exist_ok=True)
        final_dir.joinpath("A1_IT", "leadership_day_plot.png").write_bytes(b"a1it")
        final_dir.joinpath("AIDC/route_B", "leadership_day_plot.png").write_bytes(b"routeb")

        candidate_rows = {
            "A1_01a": {
                "selected_model": "lightgbm-df_power-usmd-15",
                "selected_date": "2026-05-24",
            },
            "A1_201": {
                "selected_model": "lightgbm-df_power-usmd-15",
                "selected_date": "2026-04-24",
            },
            "A3_01e": {
                "selected_model": "lightgbm-df_power-usmdp-2",
                "selected_date": "2026-04-14",
            },
            "AIDC/route_A": {
                "selected_model": "lightgbm-df_power-usmd-15",
                "selected_date": "2026-04-24",
            },
        }
        for scenario, meta in candidate_rows.items():
            row = {
                "scenario": scenario,
                "selected_model": meta["selected_model"],
                "selected_date": meta["selected_date"],
                "time_range": f"{meta['selected_date']} 00:00:00~{meta['selected_date']} 23:55:00",
                "R2": 0.1,
                "MSE": 11.0,
                "RMSE": 3.5,
                "MAE": 2.5,
                "MAPE": 0.011,
                "MAPE Accuracy": 0.989,
                "MAPE Threshold": 100.0,
                "MAPE Valid Points": 288,
                "MAPE Excluded Points": 0,
                "MAPE Excluded Ratio": 0.0,
                "corr": 0.7,
                "tail_MAE": 6.0,
                "tail_bias": 1.0,
                "plot_nan": 0,
                "selection_reason": "manual_candidate_pick",
                "note": "manual pick",
            }
            scenario_dir = candidates_dir.joinpath(scenario)
            scenario_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row]).to_csv(scenario_dir.joinpath("candidate_details.csv"), index=False)
            filename = f"{meta['selected_date']}-{meta['selected_model']}.png"
            scenario_dir.joinpath(filename).write_bytes(filename.encode("utf-8"))

        selected_rows = {
            "A1_01a": ("lightgbm-df_power-usmd-15", "2026-05-24"),
            "A1_201": ("lightgbm-df_power-usmd-15", "2026-04-24"),
            "A1_IT": ("lightgbm-df_power-usmd-15", "2026-05-16"),
            "A3_01e": ("lightgbm-df_power-usmdp-2", "2026-04-14"),
            "AIDC/route_A": ("lightgbm-df_power-usmd-15", "2026-04-24"),
            "AIDC/route_B": ("lightgbm-df_power-usmdp-2", "2026-04-14"),
        }
        for scenario, (model_name, selected_date) in selected_rows.items():
            model_dir = Path(results_root).joinpath(scenario, model_name)
            model_dir.mkdir(parents=True, exist_ok=True)
            curve_rows = self._make_day_curve(selected_date, 1000.0, pred_shift=2.0, inject_nan=True)
            pd.DataFrame(curve_rows).to_csv(model_dir.joinpath("cv_plot_df.csv"), index=False)

    def test_fill_display_series_fills_nan_for_plotting_only(self):
        values = pd.Series([1.0, float("nan"), 3.0, float("nan"), 5.0])

        filled = fill_display_series(values)

        self.assertEqual(filled.isna().sum(), 0)
        self.assertAlmostEqual(filled.iloc[1], 2.0)
        self.assertAlmostEqual(filled.iloc[3], 4.0)

    def test_resolve_report_font_returns_existing_font_file(self):
        font_prop = resolve_report_font()
        self.assertTrue(Path(font_prop.get_file()).exists())

    def test_build_candidate_pool_excludes_median_and_keeps_visual_candidates(self):
        score_defaults = {
            "R2": 0.5,
            "MSE": 100.0,
            "RMSE": 10.0,
            "MAPE Accuracy": 0.99,
            "MAPE Threshold": 100.0,
            "MAPE Valid Points": 288,
            "MAPE Excluded Points": 0,
            "MAPE Excluded Ratio": 0.0,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_model_result(
                tmpdir,
                "A1_01a",
                "lightgbm-df_power-usmd-15",
                [
                    {
                        "time_range": "2026-05-31 00:00:00~2026-05-31 23:55:00",
                        "MAPE": 0.0070,
                        "MAE": 35.0,
                        "RMSE": 66.0,
                        **score_defaults,
                    },
                    {
                        "time_range": "2026-05-24 00:00:00~2026-05-24 23:55:00",
                        "MAPE": 0.0071,
                        "MAE": 32.0,
                        "RMSE": 45.0,
                        **score_defaults,
                    },
                    {
                        "time_range": "中位数",
                        "MAPE": 0.02,
                        "MAE": 50.0,
                        "RMSE": 70.0,
                        **score_defaults,
                    },
                ],
                self._make_day_curve("2026-05-31", 4000.0, pred_shift=2.0, tail_shift=85.0)
                + self._make_day_curve("2026-05-24", 4000.0, pred_shift=3.0, tail_shift=8.0),
            )
            self._write_model_result(
                tmpdir,
                "A1_01a",
                "lightgbm-df_power-usmdp-2",
                [
                    {
                        "time_range": "2026-05-15 00:00:00~2026-05-15 23:55:00",
                        "MAPE": 0.0072,
                        "MAE": 33.0,
                        "RMSE": 46.0,
                        **score_defaults,
                    }
                ],
                self._make_day_curve("2026-05-15", 4000.0, pred_shift=4.0, tail_shift=15.0),
            )

            pool = build_candidate_pool(Path(tmpdir), "A1_01a", candidate_top_k=1, tail_points=48)

        self.assertTrue((pool["time_range"] != "中位数").all())
        self.assertIn("2026-05-24", set(pool["selected_date"]))
        self.assertIn("corr", pool.columns)
        self.assertIn("tail_MAE", pool.columns)
        self.assertIn("plot_nan", pool.columns)

    def test_run_selection_uses_tail_rule_for_a1_01a(self):
        score_defaults = {
            "R2": 0.5,
            "MSE": 100.0,
            "MAPE Accuracy": 0.99,
            "MAPE Threshold": 100.0,
            "MAPE Valid Points": 288,
            "MAPE Excluded Points": 0,
            "MAPE Excluded Ratio": 0.0,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_model_result(
                tmpdir,
                "A1_01a",
                "lightgbm-df_power-usmd-15",
                [
                    {
                        "time_range": "2026-05-31 00:00:00~2026-05-31 23:55:00",
                        "MAPE": 0.0070,
                        "MAE": 35.0,
                        "RMSE": 66.0,
                        **score_defaults,
                    },
                    {
                        "time_range": "2026-05-24 00:00:00~2026-05-24 23:55:00",
                        "MAPE": 0.0071,
                        "MAE": 32.0,
                        "RMSE": 45.0,
                        **score_defaults,
                    },
                ],
                self._make_day_curve("2026-05-31", 4000.0, pred_shift=2.0, tail_shift=90.0)
                + self._make_day_curve("2026-05-24", 4000.0, pred_shift=3.0, tail_shift=5.0),
            )
            self._write_model_result(
                tmpdir,
                "A1_01a",
                "lightgbm-df_power-usmdp-2",
                [
                    {
                        "time_range": "2026-05-15 00:00:00~2026-05-15 23:55:00",
                        "MAPE": 0.0072,
                        "MAE": 33.0,
                        "RMSE": 46.0,
                        **score_defaults,
                    },
                ],
                self._make_day_curve("2026-05-15", 4000.0, pred_shift=4.0, tail_shift=10.0),
            )

            with patch("select_aidc_leadership_days.SCENARIOS", ("A1_01a",)):
                summary = run_selection(
                    Path(tmpdir), Path(tmpdir) / "output", candidate_top_k=2,
                    archive_previous=False,
                )
            selection = summary.iloc[0].to_dict()

        self.assertEqual(selection["selected_date"], "2026-05-24")
        self.assertEqual(selection["selection_reason"], "lower_tail_error_than_metric_best")

    def test_load_day_curve_returns_full_288_point_day(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_model_result(
                tmpdir,
                "A1_01a",
                "lightgbm-df_power-usmd-15",
                [
                    {
                        "time_range": "2026-05-31 00:00:00~2026-05-31 23:55:00",
                        "MAPE": 0.01,
                        "MAE": 1.0,
                        "RMSE": 1.0,
                    }
                ],
                self._make_day_curve("2026-05-31", 4000.0),
            )
            day_curve = load_day_curve(
                Path(tmpdir),
                "A1_01a",
                "lightgbm-df_power-usmd-15",
                "2026-05-31",
            )

            self.assertEqual(len(day_curve), 288)
            self.assertEqual(str(day_curve.iloc[0]["time"]), "2026-05-31 00:00:00")
            self.assertEqual(str(day_curve.iloc[-1]["time"]), "2026-05-31 23:55:00")

    def test_load_day_curve_rejects_incomplete_day(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_model_result(
                tmpdir,
                "A1_01a",
                "lightgbm-df_power-usmd-15",
                [
                    {
                        "time_range": "2026-06-01 00:00:00~2026-06-01 23:55:00",
                        "MAPE": 0.01,
                        "MAE": 1.0,
                        "RMSE": 1.0,
                    }
                ],
                [
                    {
                        "series_id": "__local__",
                        "time": "2026-06-01 00:00:00",
                        "target": "power",
                        "actual_value": 1.0,
                        "predict_value": 1.0,
                        "window": 1,
                        "plot_valid": True,
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "Expected 288 points"):
                load_day_curve(
                    Path(tmpdir),
                    "A1_01a",
                    "lightgbm-df_power-usmd-15",
                    "2026-06-01",
                )

    def test_canonical_scores_and_curve_are_normalized_for_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "A1_01a" / "canonical"
            model_dir.mkdir(parents=True)
            times = pd.date_range("2026-06-01", periods=288, freq="5min")
            pd.DataFrame(
                {
                    "series_id": "__local__",
                    "time": times,
                    "target": "power",
                    "actual_value": 1000.0,
                    "predict_value": 1001.0,
                    "window": 1,
                    "plot_valid": True,
                }
            ).to_csv(model_dir / "cv_plot_df.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "window": 1,
                        "scope": "target",
                        "target": "power",
                        "MAE": 1.0,
                        "RMSE": 1.0,
                        "MAPE": 0.001,
                        "Accuracy": 0.999,
                        "Valid Points": 288,
                        "n_points": 288,
                    },
                    {
                        "window": 1,
                        "scope": "aggregate",
                        "target": "__aggregate__",
                        "MAE": 1.0,
                        "RMSE": 1.0,
                        "MAPE": 0.001,
                        "Accuracy": 0.999,
                        "Valid Points": 288,
                        "n_points": 288,
                    },
                ]
            ).to_csv(model_dir / "test_scores_df.csv", index=False)

            scores = _read_daily_scores(root, "A1_01a", "canonical")
            curve = load_day_curve(root, "A1_01a", "canonical", "2026-06-01")

            self.assertEqual(scores.loc[0, "time_range"], "2026-06-01 00:00:00~2026-06-01 23:55:00")
            self.assertEqual(scores.loc[0, "MAPE Accuracy"], 0.999)
            self.assertEqual(list(curve.columns)[3:5], ["actual_value", "predict_value"])

    def test_run_selection_writes_final_candidates_and_archive_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir).joinpath("results_test")
            output_root = Path(tmpdir).joinpath("leadership_selection")
            self._write_full_fixture(results_root)

            legacy_rows = []
            for scenario in ["A1_201", "A3_01e"]:
                legacy_dir = output_root.joinpath(scenario)
                legacy_dir.mkdir(parents=True, exist_ok=True)
                legacy_dir.joinpath("leadership_day_plot.png").write_bytes(b"old-plot")
                legacy_rows.append(
                    {
                        "scenario": scenario,
                        "selected_model": "lightgbm-df_power-usmd-15",
                        "selected_date": "2026-04-24",
                        "time_range": "2026-04-24 00:00:00~2026-04-24 23:55:00",
                        "R2": 0.1,
                        "MSE": 1.0,
                        "RMSE": 1.0,
                        "MAE": 1.0,
                        "MAPE": 0.01,
                        "MAPE Accuracy": 0.99,
                        "MAPE Threshold": 100.0,
                        "MAPE Valid Points": 288,
                        "MAPE Excluded Points": 0,
                        "MAPE Excluded Ratio": 0.0,
                    }
                )
            pd.DataFrame(legacy_rows).to_csv(
                output_root.joinpath("leadership_day_summary.csv"), index=False
            )

            summary_df = run_selection(results_root, output_root, candidate_top_k=1)

            self.assertEqual(len(summary_df), len(SCENARIOS))
            self.assertEqual(list(summary_df.columns), FINAL_SUMMARY_COLUMNS)
            self.assertTrue(
                output_root.joinpath("final", "leadership_day_summary.csv").exists()
            )
            for scenario in SCENARIOS:
                self.assertTrue(
                    output_root.joinpath("final", scenario, "leadership_day_plot.png").exists(),
                    scenario,
                )
                self.assertTrue(
                    output_root.joinpath("candidates", scenario, "candidate_details.csv").exists(),
                    scenario,
                )
            for scenario in ARCHIVE_SCENARIOS:
                self.assertTrue(
                    output_root.joinpath("archive", scenario, "previous_leadership_day_plot.png").exists(),
                    scenario,
                )

    def test_run_selection_adds_limit_notes_for_poor_fit_scenarios(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir).joinpath("results_test")
            output_root = Path(tmpdir).joinpath("leadership_selection")
            self._write_full_fixture(results_root)

            summary_df = run_selection(results_root, output_root, candidate_top_k=1)
            summary_df = summary_df.set_index("scenario")

            self.assertIn("relative best", summary_df.loc["A1_IT", "note"])
            self.assertIn("relative best", summary_df.loc["AIDC/route_A", "note"])
            self.assertIn("relative best", summary_df.loc["AIDC/route_B", "note"])

    def test_build_report_package_redraws_localized_report_images_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            selection_root = Path(tmpdir).joinpath("selection")
            report_root = Path(tmpdir).joinpath("reports")
            results_root = Path(tmpdir).joinpath("results_test")
            self._write_report_fixture(selection_root, results_root)

            summary_df = build_report_package(selection_root, report_root, results_root=results_root)

            self.assertEqual(len(summary_df), 6)
            self.assertEqual(list(summary_df.columns), FINAL_SUMMARY_COLUMNS)
            self.assertTrue(report_root.joinpath("leadership_day_summary.csv").exists())
            for scenario in SCENARIOS:
                localized_name = REPORT_PLOT_META[scenario]["filename"]
                self.assertTrue(
                    report_root.joinpath(scenario, localized_name).exists(),
                    scenario,
                )
                self.assertFalse(
                    report_root.joinpath(scenario, "leadership_day_plot.png").exists(),
                    scenario,
                )
                self.assertEqual(
                    report_root.joinpath(scenario, localized_name).read_bytes()[:8],
                    b"\x89PNG\r\n\x1a\n",
                )
            saved = pd.read_csv(report_root.joinpath("leadership_day_summary.csv"))
            metrics_saved = pd.read_csv(report_root.joinpath("leadership_day_summary_metrics.csv"))
            self.assertEqual(
                set(saved["scenario"]),
                {"A1_01a", "A1_201", "A1_IT", "A3_01e", "AIDC/route_A", "AIDC/route_B"},
            )
            self.assertEqual(list(metrics_saved.columns), REPORT_METRICS_COLUMNS)
            a101a = saved[saved["scenario"] == "A1_01a"].iloc[0]
            self.assertEqual(a101a["selected_date"], "2026-05-24")
            self.assertEqual(a101a["selected_model"], "lightgbm-df_power-usmd-15")

    def test_build_report_package_rejects_non_unique_candidate_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            selection_root = Path(tmpdir).joinpath("selection")
            report_root = Path(tmpdir).joinpath("reports")
            results_root = Path(tmpdir).joinpath("results_test")
            self._write_report_fixture(selection_root, results_root)
            scenario_dir = selection_root.joinpath("candidates", "A1_01a")
            df = pd.read_csv(scenario_dir.joinpath("candidate_details.csv"))
            df = pd.concat([df, df], ignore_index=True)
            df.to_csv(scenario_dir.joinpath("candidate_details.csv"), index=False)

            with self.assertRaisesRegex(ValueError, "A1_01a"):
                build_report_package(selection_root, report_root, results_root=results_root)

    def test_export_report_metrics_summary_reads_existing_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_root = Path(tmpdir).joinpath("reports")
            report_root.mkdir(parents=True, exist_ok=True)
            summary_df = pd.DataFrame(
                [
                    {
                        "scenario": "A1_01a",
                        "selected_model": "lightgbm-df_power-usmd-15",
                        "selected_date": "2026-05-24",
                        "time_range": "2026-05-24 00:00:00~2026-05-24 23:55:00",
                        "R2": 0.1,
                        "MSE": 1.0,
                        "RMSE": 1.0,
                        "MAE": 1.0,
                        "MAPE": 0.01,
                        "MAPE Accuracy": 0.99,
                        "MAPE Threshold": 100.0,
                        "MAPE Valid Points": 288,
                        "MAPE Excluded Points": 0,
                        "MAPE Excluded Ratio": 0.0,
                        "corr": 0.9,
                        "tail_MAE": 1.2,
                        "tail_bias": 0.1,
                        "plot_nan": 0,
                        "selection_reason": "manual_candidate_pick",
                        "note": "test",
                    }
                ],
                columns=FINAL_SUMMARY_COLUMNS,
            )
            summary_df.to_csv(report_root.joinpath("leadership_day_summary.csv"), index=False)

            metrics_df = export_report_metrics_summary(report_root)

            self.assertEqual(list(metrics_df.columns), REPORT_METRICS_COLUMNS)
            self.assertTrue(report_root.joinpath("leadership_day_summary_metrics.csv").exists())
            self.assertEqual(metrics_df.iloc[0]["scenario"], "A1_01a")


if __name__ == "__main__":
    unittest.main()
