# -*- coding: utf-8 -*-
"""Canonical long-format result schema tests."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from model_evaluation.point import evaluate_point_forecasts
from model_forecasting.results import (
    CanonicalResultReader,
    backtest_tensors_to_long,
    distribution_to_long,
    point_tensor_to_long,
    write_forecast_results,
)
from model_testing.reporting import write_backtest_results
from forecasting_core.tensors import (
    MarginalQuantileForecastTensor,
    PointForecastTensor,
)
from forecasting_core.artifacts import MarginalForecastDistribution


class ForecastResultSchemaTest(unittest.TestCase):
    def setUp(self):
        self.times = pd.date_range("2026-09-01", periods=3, freq="1h")
        self.point = PointForecastTensor(
            values=np.arange(12.0).reshape(2, 3, 2),
            series_ids=("A", "B"),
            forecast_times=self.times,
            targets=("load", "power"),
        )

    def test_point_long_schema_has_one_unique_row_per_series_time_target(self):
        frame = point_tensor_to_long(self.point)

        self.assertEqual(len(frame), 12)
        self.assertEqual(
            list(frame.columns),
            ["series_id", "time", "target", "predict_value"],
        )
        self.assertFalse(frame.duplicated(["series_id", "time", "target"]).any())
        self.assertEqual(frame.groupby("target").size().to_dict(), {"load": 6, "power": 6})

    def test_marginal_distribution_long_schema_keeps_target_quantiles(self):
        quantile_values = np.stack(
            [self.point.values - 1.0, self.point.values, self.point.values + 1.0],
            axis=-1,
        )
        quantiles = MarginalQuantileForecastTensor(
            values=quantile_values,
            levels=(0.1, 0.5, 0.9),
            point_level=0.5,
            series_ids=self.point.series_ids,
            forecast_times=self.times,
            targets=self.point.targets,
        )
        distribution = MarginalForecastDistribution(
            point=self.point,
            quantiles=quantiles,
            dependence_model=None,
        )

        frame = distribution_to_long(distribution)

        self.assertEqual(len(frame), 12)
        self.assertEqual(
            [column for column in frame if column.startswith("predict_q")],
            ["predict_q10", "predict_q50", "predict_q90"],
        )
        np.testing.assert_allclose(frame["predict_q50"], frame["predict_value"])

    def test_metrics_default_to_equal_target_weighting_and_include_naive(self):
        actual = PointForecastTensor(
            values=self.point.values + np.array([1.0, 10.0])[None, None, :],
            series_ids=self.point.series_ids,
            forecast_times=self.times,
            targets=self.point.targets,
        )
        seasonal_naive = PointForecastTensor(
            values=actual.values + np.array([2.0, 4.0])[None, None, :],
            series_ids=actual.series_ids,
            forecast_times=actual.forecast_times,
            targets=actual.targets,
        )
        default_report = evaluate_point_forecasts(
            actual,
            self.point,
            seasonal_naive=seasonal_naive,
            window=3,
        )
        default_aggregate = default_report[default_report["scope"] == "aggregate"]
        self.assertAlmostEqual(default_aggregate["MAE"].iloc[0], 5.5)
        self.assertEqual(default_report.attrs["aggregate_weighting"], {"load": 0.5, "power": 0.5})

        report = evaluate_point_forecasts(
            actual,
            self.point,
            aggregate_weighting={"load": 0.25, "power": 0.75},
            seasonal_naive=seasonal_naive,
            window=3,
        )

        target_rows = report[report["scope"] == "target"]
        aggregate = report[report["scope"] == "aggregate"]
        self.assertEqual(set(target_rows["target"]), {"load", "power"})
        self.assertEqual(len(aggregate), 1)
        self.assertAlmostEqual(aggregate["MAE"].iloc[0], 7.75)
        self.assertEqual(set(report["window"]), {3})
        self.assertTrue(
            {
                "MAE",
                "RMSE",
                "Bias",
                "MAPE",
                "Accuracy",
                "Valid Points",
                "Naive MAE",
                "Naive RMSE",
                "Naive Bias",
                "Naive MAPE",
                "Naive Accuracy",
            }.issubset(report.columns)
        )

    def test_backtest_long_schema_keeps_actual_quantiles_window_and_plot_mask(self):
        actual = PointForecastTensor(
            values=self.point.values + 1.0,
            series_ids=self.point.series_ids,
            forecast_times=self.times,
            targets=self.point.targets,
        )
        quantile_values = np.stack(
            [self.point.values - 1.0, self.point.values, self.point.values + 1.0],
            axis=-1,
        )
        distribution = MarginalForecastDistribution(
            point=self.point,
            quantiles=MarginalQuantileForecastTensor(
                values=quantile_values,
                levels=(0.1, 0.5, 0.9),
                point_level=0.5,
                series_ids=self.point.series_ids,
                forecast_times=self.times,
                targets=self.point.targets,
            ),
            dependence_model=None,
        )

        frame = backtest_tensors_to_long(actual, distribution, window=2)

        self.assertEqual(
            list(frame.columns),
            [
                "series_id",
                "time",
                "target",
                "actual_value",
                "predict_value",
                "predict_q10",
                "predict_q50",
                "predict_q90",
                "window",
                "plot_valid",
            ],
        )
        self.assertEqual(len(frame), 12)
        self.assertFalse(frame.duplicated(["series_id", "time", "target", "window"]).any())
        self.assertTrue(frame["plot_valid"].all())

    def test_writer_emits_only_canonical_prediction_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            write_forecast_results(output, self.point)
            saved = pd.read_csv(output / "prediction.csv")

            self.assertEqual(
                list(saved.columns),
                ["series_id", "time", "target", "predict_value"],
            )
            self.assertNotIn("Y_preds", saved)
            self.assertNotIn("pred_method", saved)
            # 预测可视化（2026-09-02）：多 target → prediction_plots/<target>.png
            self.assertTrue((output / "prediction_plots" / "load.png").exists())
            self.assertTrue((output / "prediction_plots" / "power.png").exists())

    def test_forecast_writer_emits_single_target_plot_with_quantile_band(self):
        times = pd.date_range("2026-09-01", periods=3, freq="1h")
        point = PointForecastTensor(
            values=np.arange(3.0).reshape(1, 3, 1),
            series_ids=("A",),
            forecast_times=times,
            targets=("load",),
        )
        quantile_values = np.stack(
            [point.values - 1.0, point.values, point.values + 1.0], axis=-1
        )
        distribution = MarginalForecastDistribution(
            point=point,
            quantiles=MarginalQuantileForecastTensor(
                values=quantile_values,
                levels=(0.1, 0.5, 0.9),
                point_level=0.5,
                series_ids=point.series_ids,
                forecast_times=times,
                targets=point.targets,
            ),
            dependence_model=None,
        )
        # 历史参照段（5×horizon=15 步，as-of origin，末点紧邻预测段首点）
        history_times = pd.date_range("2026-08-31 09:00", periods=15, freq="1h")
        history = PointForecastTensor(
            values=(100.0 + np.arange(15.0)).reshape(1, 15, 1),
            series_ids=("A",),
            forecast_times=history_times,
            targets=("load",),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            write_forecast_results(output, distribution, history=history)

            # 单 target → forecast_prediction.png（quantile 模式同路径）
            self.assertTrue((output / "forecast_prediction.png").exists())
            self.assertFalse((output / "prediction_plots").exists())

    def test_forecast_writer_without_history_keeps_prediction_only_plot(self):
        times = pd.date_range("2026-09-01", periods=3, freq="1h")
        point = PointForecastTensor(
            values=np.arange(3.0).reshape(1, 3, 1),
            series_ids=("A",),
            forecast_times=times,
            targets=("load",),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            write_forecast_results(output, point)

            self.assertTrue((output / "forecast_prediction.png").exists())

    def test_backtest_writer_creates_long_csv_metadata_and_target_plots(self):
        actual = PointForecastTensor(
            values=self.point.values + 1.0,
            series_ids=self.point.series_ids,
            forecast_times=self.times,
            targets=self.point.targets,
        )
        frame = backtest_tensors_to_long(actual, self.point, window=1)
        scores = evaluate_point_forecasts(actual, self.point, window=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            write_backtest_results(
                output,
                frame,
                scores,
                aggregate_weighting=scores.attrs["aggregate_weighting"],
            )

            self.assertTrue((output / "cv_plot_df.csv").exists())
            self.assertTrue((output / "test_scores_df.csv").exists())
            self.assertTrue((output / "result_metadata.json").exists())
            self.assertTrue((output / "target_plots" / "load.png").exists())
            self.assertTrue((output / "target_plots" / "power.png").exists())
            # per-window 可视化（2026-09-02）：每窗一文件
            windows_dir = output / "windows_results"
            self.assertTrue((windows_dir / "window_01.png").exists())
            self.assertEqual(
                sorted(path.name for path in windows_dir.glob("window_*.png")),
                ["window_01.png"],
            )

    def test_stitched_overview_raises_on_overlapping_windows(self):
        # stride < horizon：同 (series,time,target) 多窗口命中 → 拼接总图 RAISE，
        # windows_results/ 单窗图不受影响（先于总图写出）。
        actual = PointForecastTensor(
            values=self.point.values + 1.0,
            series_ids=self.point.series_ids,
            forecast_times=self.times,
            targets=self.point.targets,
        )
        frame = backtest_tensors_to_long(actual, self.point, window=1)
        overlap = frame.copy()
        overlap["window"] = 2
        frame = pd.concat([frame, overlap], ignore_index=True)
        scores = evaluate_point_forecasts(actual, self.point, window=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            with self.assertRaisesRegex(ValueError, "windows overlap"):
                write_backtest_results(
                    output,
                    frame,
                    scores,
                    aggregate_weighting=scores.attrs["aggregate_weighting"],
                )
            # 窗口图在总图之前已产出
            self.assertTrue((output / "windows_results" / "window_01.png").exists())
            self.assertTrue((output / "windows_results" / "window_02.png").exists())


if __name__ == "__main__":
    unittest.main()
