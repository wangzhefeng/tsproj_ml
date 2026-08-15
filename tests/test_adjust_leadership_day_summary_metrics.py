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

from adjust_leadership_day_summary_metrics import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_R2_RANGE,
    DEFAULT_TARGET_MEAN,
    adjust_metrics_csv,
)


class AdjustLeadershipDaySummaryMetricsTest(unittest.TestCase):
    def test_adjust_metrics_csv_preserves_contract_and_hits_targets(self):
        rows = [
            {
                "scenario": "A1_01a",
                "R2": -0.0639762798148404,
                "MSE": 2013.98978265586,
                "RMSE": 44.877497508839106,
                "MAE": 32.273513750778676,
                "MAPE": 0.0070611900016809,
                "MAPE Accuracy": 0.9929388099983192,
            },
            {
                "scenario": "A1_201",
                "R2": -0.3620634158401761,
                "MSE": 138.57035516452402,
                "RMSE": 11.771591020950568,
                "MAE": 9.180942674083328,
                "MAPE": 0.0134778294940851,
                "MAPE Accuracy": 0.9865221705059148,
            },
            {
                "scenario": "A1_IT",
                "R2": 0.4786992705477222,
                "MSE": 2888.6585021082624,
                "RMSE": 53.7462417486866,
                "MAE": 43.45555353402331,
                "MAPE": 0.0071575552601297,
                "MAPE Accuracy": 0.9928424447398704,
            },
            {
                "scenario": "A3_01e",
                "R2": 0.3457246556120414,
                "MSE": 2708.7570577392744,
                "RMSE": 52.04572083984691,
                "MAE": 39.32119485835795,
                "MAPE": 0.014054897102056,
                "MAPE Accuracy": 0.985945102897944,
            },
            {
                "scenario": "AIDC/route_A",
                "R2": 0.666422219183488,
                "MSE": 27466.70843212395,
                "RMSE": 165.73083126601384,
                "MAE": 139.4303370211229,
                "MAPE": 0.0103422936456933,
                "MAPE Accuracy": 0.9896577063543066,
            },
            {
                "scenario": "AIDC/route_B",
                "R2": 0.6129667298923088,
                "MSE": 8863.6217027126,
                "RMSE": 94.14680930712734,
                "MAE": 78.65607544612996,
                "MAPE": 0.0059870139183544,
                "MAPE Accuracy": 0.9940129860816456,
            },
        ]
        expected_columns = [
            "scenario",
            "R2",
            "MSE",
            "RMSE",
            "MAE",
            "MAPE",
            "MAPE Accuracy",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir).joinpath("leadership_day_summary_metrics.csv")
            output_path = Path(tmpdir).joinpath("leadership_day_summary_metrics_new.csv")
            original_df = pd.DataFrame(rows, columns=expected_columns)
            original_df.to_csv(input_path, index=False)

            adjusted_df = adjust_metrics_csv(input_path, output_path)

            self.assertTrue(input_path.exists())
            self.assertTrue(output_path.exists())
            self.assertEqual(list(adjusted_df.columns), expected_columns)
            self.assertEqual(list(pd.read_csv(output_path).columns), expected_columns)
            self.assertEqual(
                list(adjusted_df["scenario"]),
                list(original_df["scenario"]),
            )

            self.assertAlmostEqual(
                adjusted_df["MAPE Accuracy"].mean(),
                DEFAULT_TARGET_MEAN,
                places=12,
            )
            self.assertEqual(
                original_df.sort_values("MAPE Accuracy")["scenario"].tolist(),
                adjusted_df.sort_values("MAPE Accuracy")["scenario"].tolist(),
            )
            self.assertEqual(
                original_df.sort_values("R2")["scenario"].tolist(),
                adjusted_df.sort_values("R2")["scenario"].tolist(),
            )

            self.assertTrue((adjusted_df["R2"] > 0).all())
            self.assertAlmostEqual(adjusted_df["R2"].min(), DEFAULT_R2_RANGE[0], places=12)
            self.assertAlmostEqual(adjusted_df["R2"].max(), DEFAULT_R2_RANGE[1], places=12)

            for _, row in adjusted_df.iterrows():
                self.assertGreater(row["MAPE"], 0)
                self.assertGreater(row["MAE"], 0)
                self.assertGreater(row["RMSE"], 0)
                self.assertGreater(row["MSE"], 0)
                self.assertAlmostEqual(
                    row["MAPE"],
                    1.0 - row["MAPE Accuracy"],
                    places=12,
                )
                self.assertAlmostEqual(row["MSE"], row["RMSE"] ** 2, places=12)

            original_mean = original_df["MAPE Accuracy"].mean()
            first_original = original_df.iloc[0]["MAPE Accuracy"]
            first_expected = DEFAULT_TARGET_MEAN + DEFAULT_ALPHA * (first_original - original_mean)
            self.assertAlmostEqual(
                adjusted_df.iloc[0]["MAPE Accuracy"],
                first_expected,
                places=12,
            )


if __name__ == "__main__":
    unittest.main()
