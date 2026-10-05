# -*- coding: utf-8 -*-
"""SMAPE / MASE / RMSSE 手算契约与回测评分接缝接线测试（2026-10-05 指标补充）。

数值均手工核算：
- actual=[10, 20]、prediction=[12, 18]、naive=[11, 19]（单 series 单 target，H=2）；
- MAE=2、RMSE=2；naive MAE=1；
- SMAPE = mean(4/22, 4/38) ≈ 0.14354；
- in-sample 历史 [5,7,6,8,7,9]、lag=1：|diff| 均值 1.6（MASE 分母），
  diff² 均值 14/5=2.8（RMSSE 分母 √2.8 ≈ 1.67332）；
- MASE = 2/1.6 = 1.25；RMSSE = 2/√2.8 ≈ 1.19523；Naive MASE = 1/1.6 = 0.625。
"""

import unittest

import numpy as np
import pandas as pd

from forecasting_core.tensors import PointForecastTensor
from model_evaluation.point import evaluate_point_forecasts
from model_testing.loops.scoring import score_holdout_fold


def _tensor(values_2d, times, targets=("load",)):
    values = np.asarray(values_2d, dtype=float).reshape(1, len(values_2d), len(targets))
    return PointForecastTensor(
        values=values,
        series_ids=("s0",),
        forecast_times=pd.DatetimeIndex(times),
        targets=targets,
    )


def _fixtures():
    times = pd.date_range("2026-01-03", periods=2, freq="h")
    actual = _tensor([10.0, 20.0], times)
    prediction = _tensor([12.0, 18.0], times)
    naive = _tensor([11.0, 19.0], times)
    history = _tensor(
        [5.0, 7.0, 6.0, 8.0, 7.0, 9.0],
        pd.date_range("2026-01-01", periods=6, freq="h"),
    )
    return actual, prediction, naive, history


class ScaledPointMetricsTest(unittest.TestCase):
    def test_smape_mase_rmsse_match_hand_calculation(self):
        actual, prediction, naive, history = _fixtures()
        report = evaluate_point_forecasts(
            actual,
            prediction,
            seasonal_naive=naive,
            insample_history=history,
            naive_lag=1,
        )
        row = report[report["scope"] == "target"].iloc[0]
        self.assertAlmostEqual(row["SMAPE"], (4.0 / 22.0 + 4.0 / 38.0) / 2.0)
        self.assertAlmostEqual(row["MASE"], 1.25)
        self.assertAlmostEqual(row["RMSSE"], 2.0 / np.sqrt(2.8))
        self.assertAlmostEqual(row["Naive MASE"], 0.625)
        # 单 target 时 aggregate 与 target 同值
        aggregate = report[report["scope"] == "aggregate"].iloc[0]
        self.assertAlmostEqual(aggregate["MASE"], 1.25)
        # per-target horizon 行用同一 in-sample 尺度：h=1 MAE=2 → MASE=1.25
        horizon = report[(report["scope"] == "horizon") & (report["horizon"] == 1)].iloc[0]
        self.assertAlmostEqual(horizon["MASE"], 1.25)
        # aggregate_horizon 无单一尺度：MASE/RMSSE 记 NaN，不伪造
        aggregate_h = report[report["scope"] == "aggregate_horizon"]
        self.assertTrue(aggregate_h["MASE"].isna().all())
        self.assertTrue(aggregate_h["RMSSE"].isna().all())

    def test_default_without_insample_keeps_nan_columns(self):
        actual, prediction, naive, _ = _fixtures()
        report = evaluate_point_forecasts(actual, prediction, seasonal_naive=naive)
        row = report[report["scope"] == "target"].iloc[0]
        self.assertTrue(np.isnan(row["MASE"]))
        self.assertTrue(np.isnan(row["RMSSE"]))
        # SMAPE 无新输入，始终计算
        self.assertAlmostEqual(row["SMAPE"], (4.0 / 22.0 + 4.0 / 38.0) / 2.0)

    def test_insample_arguments_validated(self):
        actual, prediction, naive, history = _fixtures()
        with self.assertRaisesRegex(ValueError, "provided together"):
            evaluate_point_forecasts(actual, prediction, insample_history=history)
        with self.assertRaisesRegex(ValueError, "provided together"):
            evaluate_point_forecasts(actual, prediction, naive_lag=1)
        mismatched = PointForecastTensor(
            values=history.values,
            series_ids=history.series_ids,
            forecast_times=history.forecast_times,
            targets=("other",),
        )
        with self.assertRaisesRegex(ValueError, "targets must match"):
            evaluate_point_forecasts(
                actual, prediction, insample_history=mismatched, naive_lag=1
            )
        with self.assertRaisesRegex(ValueError, "must exceed seasonal lag"):
            evaluate_point_forecasts(
                actual, prediction, insample_history=history, naive_lag=6
            )


class _StubRunner:
    """score_holdout_fold 最小协议替身（点模式 + target_history）。"""

    def __init__(self):
        actual, prediction, naive, history = _fixtures()
        self._actual = actual
        self._prediction = prediction
        self._naive = naive
        self._history = history
        self._times = actual.forecast_times

    def forecast_designs(self, origin, feature_scaler, target_transform):
        return None, None

    def forecast_times(self, origin):
        return self._times

    def predict(self, artifact, designs, provider, forecast_times, target_transform):
        return self._prediction

    def actual(self, origin_index, forecast_times):
        return self._actual

    def seasonal_naive(self, origin, forecast_times, *, history=None):
        return self._naive

    def target_history(self, origin):
        return self._history

    def execution_evidence(self, artifact, target_transform):
        return {}


class ScaledMetricsScoringSeamTest(unittest.TestCase):
    """naive_lag 经 score_holdout_fold 接线后 MASE/RMSSE 真实到达评分帧。"""

    def test_naive_lag_enables_mase_through_scoring_seam(self):
        runner = _StubRunner()
        result = score_holdout_fold(
            runner=runner,
            fit_result=(None, None, None, None, object()),
            origin=runner._times[0] - pd.Timedelta(hours=1),
            origin_index=0,
            window=1,
            calibration_tracker=None,
            aggregate_weights=None,
            eval_mask_config=None,
            naive_lag=1,
        )
        row = result.point_scores[
            result.point_scores["scope"] == "target"
        ].iloc[0]
        self.assertAlmostEqual(row["MASE"], 1.25)

    def test_without_naive_lag_mase_stays_nan(self):
        runner = _StubRunner()
        result = score_holdout_fold(
            runner=runner,
            fit_result=(None, None, None, None, object()),
            origin=runner._times[0] - pd.Timedelta(hours=1),
            origin_index=0,
            window=1,
            calibration_tracker=None,
            aggregate_weights=None,
            eval_mask_config=None,
        )
        row = result.point_scores[
            result.point_scores["scope"] == "target"
        ].iloc[0]
        self.assertTrue(np.isnan(row["MASE"]))


if __name__ == "__main__":
    unittest.main()
