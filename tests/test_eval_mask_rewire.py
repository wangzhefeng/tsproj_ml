# -*- coding: utf-8 -*-
"""D13 rewire：评估掩码接入 canonical 指标计算的定向测试。

覆盖：
- model_testing.backtest.build_eval_mask 三模式（percentile/absolute/combined）+ 上限正交；
- evaluate_point_forecasts 掩码接入：掩码后 MAPE/Accuracy/Valid Points 与手工计算一致；
- 掩码未配置 = 行为零变化（与不传 eval_mask 的结果逐值一致）；
- naive 对照列与模型列消费同一掩码。
"""

import unittest

import numpy as np

from forecasting_core.tensors import PointForecastTensor
from model_evaluation.point import _metric_values, evaluate_point_forecasts
from model_evaluation.mask import build_eval_mask


def _tensor(values_2d):
    values = np.asarray(values_2d, dtype=float).reshape(1, len(values_2d), 1)
    return PointForecastTensor(
        values=values,
        series_ids=("s0",),
        forecast_times=__import__("pandas").date_range("2026-06-01", periods=len(values_2d), freq="15min"),
        targets=("y",),
    )


class BuildEvalMaskTest(unittest.TestCase):
    def test_percentile_mode_filters_low_values(self):
        values = np.array([0.0, 1.0, 5.0, 10.0, 100.0])
        result = build_eval_mask(values, mode="percentile", percentile=5.0)
        # P5 of positive values [1,5,10,100]（线性插值）= 1.6；阈值含等号 >=，
        # 因此 y=1 (< 1.6) 也被排除——掩码语义「低出力点不计入」
        self.assertEqual(result["threshold"], 1.6)
        self.assertEqual(result["valid_mask"].tolist(), [False, False, True, True, True])

    def test_absolute_mode(self):
        values = np.array([0.0, 2.0, 50.0])
        result = build_eval_mask(values, mode="absolute", min_value=5.0)
        self.assertEqual(result["valid_mask"].tolist(), [False, False, True])

    def test_combined_mode_takes_stricter_lower_bound(self):
        values = np.array([0.0, 1.0, 3.0, 100.0])
        result = build_eval_mask(
            values, mode="combined", percentile=5.0, min_value=2.0
        )
        self.assertEqual(result["valid_mask"].tolist(), [False, False, True, True])

    def test_max_value_is_orthogonal_upper_filter(self):
        values = np.array([1.0, 50.0, 500.0])
        result = build_eval_mask(values, mode="absolute", min_value=0.5, max_value=100.0)
        self.assertEqual(result["valid_mask"].tolist(), [True, True, False])

    def test_unknown_mode_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown eval_mask mode"):
            build_eval_mask(np.array([1.0]), mode="nope")


class EvaluateWithMaskTest(unittest.TestCase):
    def _setup(self):
        actual_values = [10.0, 2.0, 20.0, 5.0]
        pred_values = [11.0, 4.0, 19.0, 6.0]
        actual = _tensor(actual_values)
        prediction = _tensor(pred_values)
        mask = build_eval_mask(np.array(actual_values), mode="absolute", min_value=4.0)
        return actual, prediction, mask, actual_values, pred_values

    def test_masked_metrics_match_manual_computation(self):
        actual, prediction, mask, actual_values, pred_values = self._setup()
        frame = evaluate_point_forecasts(
            actual, prediction, eval_mask={"mode": "absolute", "min_value": 4.0}
        )
        row = frame[frame["scope"] == "target"].iloc[0]
        # 掩码保留 idx 0,2,3（值 10,20,5），排除 idx1（2 < 4）
        expected_mape = np.mean(
            [
                abs(11.0 - 10.0) / 10.0,
                abs(19.0 - 20.0) / 20.0,
                abs(6.0 - 5.0) / 5.0,
            ]
        )
        self.assertAlmostEqual(row["MAPE"], expected_mape)
        self.assertAlmostEqual(row["Accuracy"], 1.0 - expected_mape)
        self.assertEqual(int(row["Valid Points"]), 3)
        self.assertAlmostEqual(row["MAE"], np.mean([1.0, 1.0, 1.0]))

    def test_naive_column_shares_same_mask(self):
        actual, prediction, mask, actual_values, pred_values = self._setup()
        naive = _tensor([12.0, 3.0, 25.0, 1.0])
        frame = evaluate_point_forecasts(
            actual,
            prediction,
            seasonal_naive=naive,
            eval_mask={"mode": "absolute", "min_value": 4.0},
        )
        row = frame[frame["scope"] == "target"].iloc[0]
        expected_naive_mape = np.mean(
            [
                abs(12.0 - 10.0) / 10.0,
                abs(25.0 - 20.0) / 20.0,
                abs(1.0 - 5.0) / 5.0,
            ]
        )
        self.assertAlmostEqual(row["Naive MAPE"], expected_naive_mape)
        self.assertEqual(int(row["Naive Valid Points"]), 3)

    def test_no_mask_is_unchanged(self):
        actual, prediction, _, actual_values, pred_values = self._setup()
        default = evaluate_point_forecasts(actual, prediction)
        explicit_none = evaluate_point_forecasts(actual, prediction, eval_mask=None)
        self.assertTrue(default.equals(explicit_none))
        # 未配置掩码时 Valid Points 与原口径（isfinite + actual!=0）一致
        row = default[default["scope"] == "target"].iloc[0]
        self.assertEqual(int(row["Valid Points"]), 4)

    def test_metric_values_direct_mask(self):
        m = _metric_values(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.5, 2.5, 3.5]),
            build_eval_mask(np.array([1.0, 2.0, 3.0]), mode="absolute", min_value=1.5),
        )
        self.assertEqual(m["Valid Points"], 2)


if __name__ == "__main__":
    unittest.main()
