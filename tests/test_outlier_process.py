# -*- coding: utf-8 -*-
"""data_process.outlier_process（离线异常检测与清洗工具）的回归测试。

被测模块已从 data_provider/outlier_process.py 重写迁移为
data_process/outlier_process.py（CLI 工具化），本测试针对新 API：
detect_anomalies / build_cleaned_series / process_outlier。
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_process.outlier_process import (
    ANOMALY_COL,
    ANOMALY_SCORE_COL,
    ANOMALY_TYPE_COL,
    AUTO_CLEAN_COL,
    OutlierParams,
    OutlierSpec,
    build_cleaned_series,
    detect_anomalies,
    process_outlier,
)

TIME_COL = "count_data_time"
TARGET_COL = "h_total_use"


class OutlierProcessTest(unittest.TestCase):
    def _sample_power_frame(self):
        return pd.DataFrame(
            {
                TIME_COL: pd.date_range("2026-06-01 00:00:00", periods=8, freq="5min").strftime("%Y-%m-%d %H:%M:%S"),
                TARGET_COL: [1500.0, 1510.0, 500.0, 1520.0, 1530.0, 1540.0, 1550.0, 1560.0],
            }
        )

    def _params(self):
        # abs_low_threshold=1000：样本中 500.0 命中 absolute_low 物理边界
        return OutlierParams(abs_low_threshold=1000.0)

    def test_detect_anomalies_preserves_source_columns_and_adds_marker_columns(self):
        df = self._sample_power_frame()

        marked = detect_anomalies(df, TIME_COL, TARGET_COL, self._params())

        self.assertEqual(marked[TIME_COL].tolist(), df[TIME_COL].tolist())
        self.assertEqual(marked[TARGET_COL].tolist(), df[TARGET_COL].tolist())
        self.assertIn(ANOMALY_COL, marked.columns)
        self.assertIn(ANOMALY_TYPE_COL, marked.columns)
        self.assertIn(ANOMALY_SCORE_COL, marked.columns)
        self.assertIn(AUTO_CLEAN_COL, marked.columns)
        self.assertEqual(marked.loc[2, ANOMALY_COL], "是")
        self.assertIn("absolute_low", marked.loc[2, ANOMALY_TYPE_COL])
        # 物理边界越界点默认自动清洗
        self.assertEqual(marked.loc[2, AUTO_CLEAN_COL], "是")

    def test_build_cleaned_series_keeps_two_column_output_contract(self):
        marked = detect_anomalies(self._sample_power_frame(), TIME_COL, TARGET_COL, self._params())

        cleaned = build_cleaned_series(marked, TIME_COL, TARGET_COL, self._params())

        self.assertEqual(list(cleaned.columns), [TIME_COL, TARGET_COL])
        self.assertNotEqual(cleaned.loc[2, TARGET_COL], 500.0)
        self.assertEqual(int(cleaned[TARGET_COL].isna().sum()), 0)

    def test_process_outlier_writes_outputs_under_outlier_analysis_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir).joinpath("df_power.csv")
            self._sample_power_frame().to_csv(source_path, index=False)

            result = process_outlier(
                OutlierSpec(
                    source_path=source_path,
                    time_col=TIME_COL,
                    target_col=TARGET_COL,
                    params=self._params(),
                    plot=False,
                    route="unit",
                )
            )

            out_dir = Path(tmpdir).joinpath("outlier_analysis")
            marked_path = out_dir.joinpath("df_power_outlier_detection.csv")
            cleaned_path = out_dir.joinpath("df_power_remove_outlier.csv")
            self.assertEqual(result.marked_csv, marked_path)
            self.assertEqual(result.cleaned_csv, cleaned_path)
            self.assertTrue(marked_path.exists())
            self.assertTrue(cleaned_path.exists())
            self.assertEqual(result.total_rows, 8)
            self.assertEqual(result.anomaly_count, 1)

            marked = pd.read_csv(marked_path)
            cleaned = pd.read_csv(cleaned_path)
            self.assertEqual(list(cleaned.columns), [TIME_COL, TARGET_COL])
            self.assertIn(ANOMALY_COL, marked.columns)
            self.assertIn(ANOMALY_TYPE_COL, marked.columns)
            self.assertIn(ANOMALY_SCORE_COL, marked.columns)


if __name__ == "__main__":
    unittest.main()
