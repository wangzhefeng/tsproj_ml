# -*- coding: utf-8 -*-

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

# AIDC 专项脚本目录（路径含日期段，不是合法 Python 包，按脚本自身约定做 sys.path 引导）
_AIDC_SCRIPTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "config" / "aidc_electricity_computility" / "electricity" / "2026-06-11" / "scripts"
)
if str(_AIDC_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AIDC_SCRIPTS_DIR))

from fuse_aidc_results import FUSED_MODEL_NAME, fuse_scenario_results  # noqa: E402


class FuseAidcResultsTest(unittest.TestCase):
    def _write_model_result(self, root: Path, scenario: str, model_name: str, pred_offset: float):
        model_dir = root.joinpath(scenario, model_name)
        model_dir.mkdir(parents=True, exist_ok=True)

        times = pd.date_range("2026-06-01 00:00:00", periods=288, freq="5min")
        true_values = [1000.0 + i for i in range(288)]
        pred_values = [v + pred_offset for v in true_values]
        curve_df = pd.DataFrame(
            {
                "series_id": ["__local__"] * 288,
                "time": times,
                "target": ["power"] * 288,
                "actual_value": true_values,
                "predict_value": pred_values,
                "window": [1] * 288,
                "plot_valid": [True] * 288,
            }
        )
        score_df = pd.DataFrame(
            [
                {
                    "time_range": "2026-06-01 00:00:00~2026-06-01 23:55:00",
                    "R2": 0.99,
                    "MSE": 1.0,
                    "RMSE": 1.0,
                    "MAE": abs(pred_offset),
                    "MAPE": abs(pred_offset) / 1000.0,
                    "MAPE Accuracy": 1 - abs(pred_offset) / 1000.0,
                    "MAPE Threshold": 1000.0,
                    "MAPE Upper Threshold": None,
                    "MAPE Valid Points": 288,
                    "MAPE Excluded Points": 0,
                    "MAPE Excluded Ratio": 0.0,
                }
            ]
        )
        report_df = pd.DataFrame(
            columns=[
                "window",
                "start_time",
                "end_time",
                "index",
                "time",
                "target",
                "original_value",
                "cleaned_value",
                "outlier_type",
            ]
        )

        score_df.to_csv(model_dir.joinpath("test_scores_df.csv"), index=False)
        curve_df.to_csv(model_dir.joinpath("cv_plot_df.csv"), index=False)
        report_df.to_csv(model_dir.joinpath("train_outlier_report.csv"), index=False)

    def _append_extra_day(self, root: Path, scenario: str, model_name: str):
        model_dir = root.joinpath(scenario, model_name)
        curve_df = pd.read_csv(model_dir.joinpath("cv_plot_df.csv"))
        extra_times = pd.date_range("2026-06-02 00:00:00", periods=288, freq="5min")
        extra_curve_df = pd.DataFrame(
            {
                "series_id": ["__local__"] * 288,
                "time": extra_times,
                "target": ["power"] * 288,
                "actual_value": [2000.0 + i for i in range(288)],
                "predict_value": [2001.0 + i for i in range(288)],
                "window": [1] * 288,
                "plot_valid": [True] * 288,
            }
        )
        pd.concat([curve_df, extra_curve_df], axis=0).to_csv(
            model_dir.joinpath("cv_plot_df.csv"), index=False
        )

    def _write_canonical_model_result(
        self,
        root: Path,
        scenario: str,
        model_name: str,
        pred_offset: float,
    ):
        model_dir = root.joinpath(scenario, model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        times = pd.date_range("2026-06-01 00:00:00", periods=288, freq="5min")
        actual = pd.Series([1000.0 + i for i in range(288)])
        pd.DataFrame(
            {
                "series_id": "__local__",
                "time": times,
                "target": "power",
                "actual_value": actual,
                "predict_value": actual + pred_offset,
                "window": 1,
                "plot_valid": True,
            }
        ).to_csv(model_dir / "cv_plot_df.csv", index=False)

    def test_fuse_scenario_results_writes_average_outputs_and_plot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scenario = "AIDC/route_A"
            model_a = "lightgbm-df_power-usmd-15"
            model_b = "lightgbm-df_power-usmdp-2"
            self._write_model_result(root, scenario, model_a, pred_offset=10.0)
            self._write_model_result(root, scenario, model_b, pred_offset=-6.0)

            fused_dir = fuse_scenario_results(root, scenario, model_a, model_b)

            self.assertEqual(fused_dir, root.joinpath(scenario, FUSED_MODEL_NAME))
            self.assertTrue(fused_dir.joinpath("cv_plot_df.csv").exists())
            self.assertTrue(fused_dir.joinpath("test_scores_df.csv").exists())
            self.assertTrue(fused_dir.joinpath("test_prediction.png").exists())
            # per-window 可视化（2026-09-02）：单窗场景也有 windows_results/
            self.assertTrue(
                fused_dir.joinpath("windows_results", "window_01.png").exists()
            )
            self.assertTrue(fused_dir.joinpath("train_outlier_report.csv").exists())

            fused_curve_df = pd.read_csv(fused_dir.joinpath("cv_plot_df.csv"))
            self.assertEqual(fused_curve_df.loc[0, "predict_value"], 1002.0)
            self.assertEqual(fused_curve_df.loc[10, "predict_value"], 1012.0)

            fused_scores_df = pd.read_csv(fused_dir.joinpath("test_scores_df.csv"))
            # 拆分后（2026-09-03 方案 B）：主文件 = target + aggregate 两行；
            # H=288 单 target 单窗的 per-horizon 明细在 test_scores_horizon_df.csv
            self.assertEqual(len(fused_scores_df), 2)
            self.assertEqual(
                fused_scores_df["scope"].tolist(), ["target", "aggregate"]
            )
            fused_horizon_df = pd.read_csv(
                fused_dir.joinpath("test_scores_horizon_df.csv")
            )
            self.assertEqual(len(fused_horizon_df), 2 * 288)
            self.assertAlmostEqual(fused_scores_df.loc[0, "MAE"], 2.0)

    def test_fuse_scenario_results_uses_intersection_when_source_lengths_differ(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scenario = "AIDC/route_A"
            model_a = "lightgbm-df_power-usmd-15"
            model_b = "lightgbm-df_power-usmdp-2"
            self._write_model_result(root, scenario, model_a, pred_offset=4.0)
            self._write_model_result(root, scenario, model_b, pred_offset=0.0)
            self._append_extra_day(root, scenario, model_b)

            fused_dir = fuse_scenario_results(root, scenario, model_a, model_b)

            fused_curve_df = pd.read_csv(fused_dir.joinpath("cv_plot_df.csv"))
            fused_scores_df = pd.read_csv(fused_dir.joinpath("test_scores_df.csv"))
            self.assertEqual(len(fused_curve_df), 288)
            # 拆分后主文件 = target + aggregate；288×2 horizon 行在明细文件
            self.assertEqual(len(fused_scores_df), 2)
            self.assertEqual(
                len(
                    pd.read_csv(fused_dir.joinpath("test_scores_horizon_df.csv"))
                ),
                2 * 288,
            )

    def test_fuse_scenario_results_reads_mixed_legacy_and_canonical_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scenario = "AIDC/route_A"
            model_a = "legacy"
            model_b = "canonical"
            self._write_model_result(root, scenario, model_a, pred_offset=6.0)
            self._write_canonical_model_result(
                root,
                scenario,
                model_b,
                pred_offset=-2.0,
            )

            fused_dir = fuse_scenario_results(root, scenario, model_a, model_b)

            fused_curve_df = pd.read_csv(fused_dir / "cv_plot_df.csv")
            self.assertEqual(fused_curve_df.loc[0, "predict_value"], 1002.0)
            self.assertEqual(
                list(fused_curve_df.columns),
                [
                    "series_id",
                    "time",
                    "target",
                    "window",
                    "actual_value",
                    "predict_value",
                    "plot_valid",
                ],
            )


if __name__ == "__main__":
    unittest.main()
