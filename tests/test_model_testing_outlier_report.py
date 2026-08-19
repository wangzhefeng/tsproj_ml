# -*- coding: utf-8 -*-

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from models.ModelForecasting import Forecaster
from models.ModelTesting import Tester
from data_provider.outlier_handling import empty_train_outlier_report
from utils.eval_mask import build_eval_mask


class _FakeTargetScaler:
    def get_prediction_target_columns(self, pred_method, target_output_features):
        return target_output_features

    def restore_predictions(self, values, pred_target_columns):
        return np.asarray(values).reshape(-1)

    def prepare_eval_target(self, values, target_output_features):
        return np.asarray(values).reshape(-1)


class _FakeTrainer:
    def __init__(self, args, log_prefix):
        self.args = args
        self.log_prefix = log_prefix

    def train(self, X_train, Y_train, feature_scaler, target_scaler, categorical_features):
        return object(), None, _FakeTargetScaler(), []


class _FakeForecaster:
    captured_test_frame = None

    def __init__(
        self,
        args,
        horizon,
        model,
        feature_scaler,
        target_scaler,
        df_history,
        df_future,
        df_date_future,
        df_weather_future,
        endogenous_features,
        target_feature,
        target_output_features,
        categorical_features,
        selected_features,
        log_prefix,
        df_custom_future=None,
        target_decomposer=None,
    ):
        self.horizon = horizon
        _FakeForecaster.captured_test_frame = df_future.copy()

    def _predict_by_method(self):
        return np.array([5000.0, 5000.0])


class ModelTestingOutlierReportTest(unittest.TestCase):
    def _history_frame(self):
        return pd.DataFrame(
            {
                "time": pd.date_range("2026-06-01 00:00:00", periods=8, freq="5min"),
                "y": [5000.0, 5100.0, 20000.0, 21000.0, 5200.0, 5300.0, 3000.0, 3100.0],
            }
        )

    def test_window_test_cleans_train_window_only_and_returns_report(self):
        args = SimpleNamespace(
            enable_train_outlier_handling=True,
            train_outlier_method="local_interpolate",
            high_outlier_threshold=15000.0,
            high_outlier_max_run_points=4,
            drop_outlier_max_run_points=0,
            drop_rebound_min_abs_diff=900.0,
            mode="percentile",
            percentile=5.0,
            min_value=None,
            max_value=None,
            pred_method="univariate-single-multistep-direct-output",
        )
        captured_train = {}

        def fake_build_window_train_xy(
            args,
            log_prefix,
            df_history_train,
            df_date_history,
            df_weather_history,
            endogenous_features_with_target,
            target_feature,
            horizon,
            df_custom_history=None,
        ):
            captured_train["df"] = df_history_train.copy()
            return (
                pd.DataFrame({"feature": [1.0, 2.0]}),
                pd.Series([5000.0, 5100.0]),
                ["y_t+1"],
                [],
            )

        payload = {
            "args": args,
            "log_prefix": "[test]",
            "horizon": 2,
            "window_len": 6,
            "window": 1,
            "df_history": self._history_frame(),
            "df_date_history": None,
            "df_weather_history": None,
            "endogenous_features_with_target": ["y"],
            "target_feature": "y",
        }

        with patch.object(Tester, "_build_window_train_xy", side_effect=fake_build_window_train_xy), \
             patch("models.ModelTesting.Trainer", _FakeTrainer), \
             patch("models.ModelTesting.Forecaster", _FakeForecaster):
            result = Tester._window_test(payload)

        self.assertIn("train_outlier_report", result)
        self.assertFalse(result["train_outlier_report"].empty)
        self.assertEqual(result["train_outlier_report"]["outlier_type"].tolist(), ["high_short_run", "high_short_run"])
        self.assertNotEqual(captured_train["df"].iloc[2]["y"], 20000.0)
        self.assertNotEqual(captured_train["df"].iloc[3]["y"], 21000.0)
        self.assertEqual(_FakeForecaster.captured_test_frame["time"].tolist(), self._history_frame().iloc[6:8]["time"].tolist())
        self.assertNotIn("y", _FakeForecaster.captured_test_frame.columns)

    def test_window_test_fits_linear_decomposition_on_train_slice_only(self):
        history = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=8, freq="1D"),
                "y": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 1000.0, 2000.0],
            }
        )
        args = SimpleNamespace(
            decomposition_method="linear",
            decomposition_periods=[],
            decomposition_trend_degree=1,
            decomposition_trend_forecast="polynomial",
            decomposition_damping=0.98,
            enable_train_outlier_handling=False,
            mode="percentile",
            percentile=5.0,
            min_value=None,
            max_value=None,
            pred_method="univariate-single-multistep-direct-output",
            n_per_day=1,
        )
        captured_train = {}

        def fake_build_window_train_xy(
            args,
            log_prefix,
            df_history_train,
            df_date_history,
            df_weather_history,
            endogenous_features_with_target,
            target_feature,
            horizon,
            df_custom_history=None,
        ):
            captured_train["df"] = df_history_train.copy()
            return (
                pd.DataFrame({"feature": [1.0, 2.0]}),
                pd.Series([0.0, 0.0]),
                ["y_t+1"],
                [],
            )

        payload = {
            "args": args,
            "log_prefix": "[test]",
            "horizon": 2,
            "window_len": 6,
            "window": 1,
            "df_history": history,
            "df_date_history": None,
            "df_weather_history": None,
            "endogenous_features_with_target": ["y"],
            "target_feature": "y",
        }

        with patch.object(Tester, "_build_window_train_xy", side_effect=fake_build_window_train_xy), \
             patch("models.ModelTesting.Trainer", _FakeTrainer), \
             patch("models.ModelTesting.Forecaster", _FakeForecaster):
            Tester._window_test(payload)

        np.testing.assert_allclose(captured_train["df"]["y"], 0.0, atol=1e-10)

    def test_test_results_save_always_writes_train_outlier_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(test_results_dir=Path(tmpdir))
            test_scores_df = pd.DataFrame({"time_range": ["x"], "MAE": [1.0]})
            cv_plot_df = pd.DataFrame()
            report = empty_train_outlier_report()

            Tester.test_results_save(args, "[test]", test_scores_df, cv_plot_df, report)

            saved = pd.read_csv(Path(tmpdir).joinpath("train_outlier_report.csv"))
            self.assertEqual(list(saved.columns), list(report.columns))

    def test_evaluate_score_uses_window_p5_relative_threshold_for_mape(self):
        df_history_test = pd.DataFrame(
            {"time": pd.date_range("2026-06-01 00:00:00", periods=5, freq="5min")}
        )
        y_test = np.array([0.0, 100.0, 200.0, 4000.0, 5000.0])
        y_pred = np.array([3500.0, 110.0, 210.0, 3900.0, 5100.0])

        result = Tester._evaluate_score(
            y_test=y_test,
            y_pred=y_pred,
            window=1,
            df_history_test=df_history_test,
            log_prefix="[test]",
        )

        self.assertIn("MAPE Threshold", result.columns)
        self.assertIn("MAPE Valid Points", result.columns)
        self.assertIn("MAPE Excluded Points", result.columns)
        self.assertIn("MAPE Excluded Ratio", result.columns)

        expected_threshold = np.percentile(np.array([100.0, 200.0, 4000.0, 5000.0]), 5)
        expected_mape = (
            abs(210.0 - 200.0) / 200.0
            + abs(3900.0 - 4000.0) / 4000.0
            + abs(5100.0 - 5000.0) / 5000.0
        ) / 3

        self.assertAlmostEqual(result.loc[1, "MAPE Threshold"], expected_threshold)
        self.assertEqual(result.loc[1, "MAPE Valid Points"], 3)
        self.assertEqual(result.loc[1, "MAPE Excluded Points"], 2)
        self.assertAlmostEqual(result.loc[1, "MAPE Excluded Ratio"], 2 / 5)
        self.assertAlmostEqual(result.loc[1, "MAPE"], expected_mape)
        self.assertAlmostEqual(result.loc[1, "MAPE Accuracy"], 1 - expected_mape)

    def test_evaluate_result_marks_invalid_points_for_plot_breaks(self):
        df_history_test = pd.DataFrame(
            {"time": pd.date_range("2026-06-01 00:00:00", periods=5, freq="5min")}
        )
        y_test = np.array([0.0, 100.0, 200.0, 4000.0, 5000.0])
        y_pred = np.array([3500.0, 110.0, 210.0, 3900.0, 5100.0])

        result = Tester._evaluate_result(
            y_test=y_test,
            y_pred=y_pred,
            df_history_test=df_history_test,
            log_prefix="[test]",
        )

        self.assertIn("mape_valid", result.columns)
        self.assertEqual(result["mape_valid"].tolist(), [False, False, True, True, True])
        self.assertTrue(np.isnan(result.loc[0, "Y_trues_plot"]))
        self.assertTrue(np.isnan(result.loc[0, "Y_preds_plot"]))
        self.assertTrue(np.isnan(result.loc[1, "Y_trues_plot"]))
        self.assertTrue(np.isnan(result.loc[1, "Y_preds_plot"]))
        self.assertEqual(result.loc[3, "Y_trues_plot"], 4000.0)
        self.assertEqual(result.loc[4, "Y_preds_plot"], 5100.0)

    def test_forecast_plot_mask_uses_history_context_positive_p5(self):
        y_true = np.array([4300.0, 4200.0, 4100.0, 4000.0, 3900.0, 3800.0, 3700.0, 100.0])

        meta = build_eval_mask(y_true)

        expected_threshold = np.percentile(y_true[y_true > 0], 5)
        self.assertAlmostEqual(meta["threshold"], expected_threshold)
        self.assertEqual(meta["valid_points"], 7)
        self.assertEqual(meta["excluded_points"], 1)
        self.assertAlmostEqual(meta["excluded_ratio"], 1 / 8)
        self.assertEqual(meta["valid_mask"].tolist(), [True, True, True, True, True, True, True, False])

    def test_eval_mask_default_min_value_none_reproduces_percentile(self):
        # min_value=None + 默认 percentile 模式 == 历史行为（仅按 P5 过滤）
        y_true = np.array([5000.0, 5100.0, 100.0, 5200.0, 5300.0])
        meta = build_eval_mask(y_true, mode="percentile", percentile=5.0, min_value=None)
        expected_threshold = np.percentile(y_true[y_true > 0], 5)
        self.assertAlmostEqual(meta["threshold"], expected_threshold)
        self.assertEqual(meta["valid_mask"].tolist(), [True, True, False, True, True])

    def test_eval_mask_absolute_mode_excludes_below_floor(self):
        # 绝对下限模式：精准砍掉 < min_value 的断点，不碰正常低谷
        y_true = np.array([2600.0, 2650.0, 2.0, 2550.0, 1400.0, 2700.0])
        meta = build_eval_mask(y_true, mode="absolute", min_value=1500.0)
        self.assertAlmostEqual(meta["threshold"], 1500.0)
        self.assertEqual(meta["valid_mask"].tolist(), [True, True, False, True, False, True])
        self.assertEqual(meta["valid_points"], 4)
        self.assertEqual(meta["excluded_points"], 2)

    def test_eval_mask_combined_mode_uses_max_threshold(self):
        # combined：阈值取 max(P{percentile}, min_value)
        y_true = np.array([2600.0, 2650.0, 2.0, 2550.0, 1400.0, 2700.0])
        pct = np.percentile(y_true[y_true > 0], 5)
        meta = build_eval_mask(y_true, mode="combined", percentile=5.0, min_value=1500.0)
        self.assertAlmostEqual(meta["threshold"], max(pct, 1500.0))
        # 2.0 与 1400.0 均 < max(pct, 1500) → 排除
        self.assertEqual(meta["valid_mask"].tolist(), [True, True, False, True, False, True])

    def test_eval_mask_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            build_eval_mask(np.array([1.0, 2.0]), mode="bogus")

    def test_eval_mask_max_value_excludes_above_cap(self):
        # 上限与 mode 正交：y > max_value 视为异常；未设时 upper_threshold 为 NaN
        y_true = np.array([2600.0, 2650.0, 30000.0, 2550.0, 31000.0, 2700.0])
        # 不设上限
        meta_no_cap = build_eval_mask(y_true, mode="absolute", min_value=1500.0)
        self.assertTrue(np.isnan(meta_no_cap["upper_threshold"]))
        self.assertEqual(meta_no_cap["valid_mask"].tolist(), [True, True, True, True, True, True])
        # 设上限 15000
        meta = build_eval_mask(y_true, mode="absolute", min_value=1500.0, max_value=15000.0)
        self.assertAlmostEqual(meta["upper_threshold"], 15000.0)
        self.assertEqual(meta["valid_mask"].tolist(), [True, True, False, True, False, True])
        self.assertEqual(meta["valid_points"], 4)
        self.assertEqual(meta["excluded_points"], 2)

    def test_evaluate_score_absolute_mode_threads_config(self):
        # 端到端：_evaluate_score 接收 absolute 配置后，MAPE 只在 y>=min_value 上计算
        df_history_test = pd.DataFrame(
            {"time": pd.date_range("2026-06-01 00:00:00", periods=5, freq="5min")}
        )
        y_test = np.array([2600.0, 2.0, 2700.0, 1400.0, 2800.0])
        y_pred = np.array([2620.0, 2610.0, 2680.0, 2630.0, 2790.0])

        result = Tester._evaluate_score(
            y_test=y_test,
            y_pred=y_pred,
            window=1,
            df_history_test=df_history_test,
            log_prefix="[test]",
            mode="absolute",
            min_value=1500.0,
        )

        self.assertAlmostEqual(result.loc[1, "MAPE Threshold"], 1500.0)
        self.assertEqual(result.loc[1, "MAPE Valid Points"], 3)
        # 仅在 y_test>=1500 的 3 个点上算 MAPE（2600/2700/2800）
        expected_mape = (
            abs(2620.0 - 2600.0) / 2600.0
            + abs(2680.0 - 2700.0) / 2700.0
            + abs(2790.0 - 2800.0) / 2800.0
        ) / 3
        self.assertAlmostEqual(result.loc[1, "MAPE"], expected_mape)

    def test_forecast_results_save_masks_history_outliers_but_keeps_future_raw_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            forecaster = Forecaster.__new__(Forecaster)
            forecaster.args = SimpleNamespace(
                pred_results_dir=Path(tmpdir),
                pred_method="univariate-single-multistep-direct-pointwise",
                mode="percentile",
                percentile=5.0,
                min_value=None,
                max_value=None,
            )
            forecaster.log_prefix = "[test]"

            df_history = pd.DataFrame(
                {
                    "time": pd.date_range("2026-06-10 00:00:00", periods=8, freq="5min"),
                    "y": [4300.0, 4200.0, 4100.0, 4000.0, 3900.0, 3800.0, 3700.0, 100.0],
                }
            )
            df_future = pd.DataFrame(
                {
                    "time": pd.date_range("2026-06-10 00:40:00", periods=2, freq="5min"),
                    "predict_value": [4020.0, 4030.0],
                }
            )

            forecaster.forecast_results_save(df_history, df_future, n_per_day=4)

            prediction_df = pd.read_csv(Path(tmpdir).joinpath("prediction.csv"))
            plot_df = pd.read_csv(Path(tmpdir).joinpath("prediction_plot_concat.csv"))

            self.assertEqual(list(prediction_df.columns), ["time", "predict_value"])
            self.assertEqual(prediction_df["predict_value"].tolist(), [4020.0, 4030.0])

            self.assertIn("raw_value", plot_df.columns)
            self.assertIn("plot_value", plot_df.columns)
            self.assertIn("plot_valid", plot_df.columns)
            self.assertIn("series_type", plot_df.columns)

            invalid_history = plot_df[(plot_df["series_type"] == "history_true") & (plot_df["raw_value"] == 100.0)]
            self.assertEqual(len(invalid_history), 1)
            self.assertFalse(bool(invalid_history.iloc[0]["plot_valid"]))
            self.assertTrue(np.isnan(invalid_history.iloc[0]["plot_value"]))

            future_rows = plot_df[plot_df["series_type"] == "future_pred"].reset_index(drop=True)
            self.assertEqual(future_rows["raw_value"].tolist(), [4020.0, 4030.0])
            self.assertEqual(future_rows["plot_value"].tolist(), [4020.0, 4030.0])
            self.assertEqual(future_rows["plot_valid"].tolist(), [True, True])


if __name__ == "__main__":
    unittest.main()
