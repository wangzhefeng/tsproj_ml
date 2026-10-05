# -*- coding: utf-8 -*-
"""概率预测指标的手算契约测试。"""

import unittest

import numpy as np

from model_evaluation.metrics import (
    crps_from_pinball,
    crossing_metrics,
    interval_metrics,
    pinball_loss,
    seasonal_insample_scales,
)


class ProbabilisticMetricsTest(unittest.TestCase):
    def test_pinball_and_interval_metrics_match_hand_calculation(self):
        losses = pinball_loss(
            y_true=np.array([0.0, 2.0]),
            y_pred=np.array([1.0, 1.0]),
            quantile=0.1,
        )
        np.testing.assert_allclose(losses, [0.9, 0.1])

        metrics = interval_metrics(
            y_true=np.array([0.0, 5.0, 10.0]),
            lower=np.array([1.0, 4.0, 8.0]),
            upper=np.array([3.0, 6.0, 9.0]),
            target_coverage=0.8,
        )

        self.assertAlmostEqual(metrics["coverage"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["width"], 5.0 / 3.0)
        self.assertAlmostEqual(metrics["winkler"], 25.0 / 3.0)
        self.assertAlmostEqual(metrics["coverage_gap"], 1.0 / 3.0 - 0.8)
        self.assertAlmostEqual(metrics["calibration_error"], abs(1.0 / 3.0 - 0.8))
        self.assertEqual(metrics["n_points"], 3)
        self.assertAlmostEqual(metrics["coverage_ci_lower"], 0.06149194472039621)
        self.assertAlmostEqual(metrics["coverage_ci_upper"], 0.7923403991979522)

    def test_crossing_metrics_report_raw_violations_and_repair_changes(self):
        raw = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
        processed = np.array([[2.0, 2.0, 2.0], [1.0, 2.0, 3.0]])

        metrics = crossing_metrics(
            raw_quantiles=raw,
            quantile_levels=(0.1, 0.5, 0.9),
            processed_quantiles=processed,
        )

        self.assertEqual(metrics["row_crossing_rate"], 0.5)
        self.assertEqual(metrics["adjacent_crossing_rate"], 0.5)
        self.assertEqual(metrics["crossing_magnitude"], 0.5)
        self.assertAlmostEqual(metrics["repair_changed_ratio"], 1.0 / 3.0)
        self.assertEqual(metrics["q50_changed_ratio"], 0.0)

    def test_crps_trapezoid_matches_hand_calculation(self):
        # levels (0.25, 0.75)、pinball 均 0.25：梯形面积 0.25*0.5，×2 = 0.25
        crps = crps_from_pinball((0.25, 0.75), (0.25, 0.25))
        self.assertAlmostEqual(crps, 0.25)
        # 非对称 pinball：levels (0.1, 0.5, 0.9)，pinball (0.2, 0.1, 0.3)
        # 梯形 = (0.2+0.1)/2*0.4 + (0.1+0.3)/2*0.4 = 0.06 + 0.08 = 0.14，×2 = 0.28
        crps = crps_from_pinball((0.1, 0.5, 0.9), (0.2, 0.1, 0.3))
        self.assertAlmostEqual(crps, 0.28)
        with self.assertRaises(ValueError):
            crps_from_pinball((0.5,), (0.1,))
        with self.assertRaises(ValueError):
            crps_from_pinball((0.9, 0.1), (0.1, 0.2))
        with self.assertRaises(ValueError):
            crps_from_pinball((0.1, 0.9), (0.1,))

    def test_seasonal_insample_scales_do_not_cross_series_boundary(self):
        # (N=2, T=4)、lag=1：series0 差分 |1,2,4|、series1 全 0；
        # 若误跨 series 边界会混入 |100-8|=92
        mae_scale, rmse_scale = seasonal_insample_scales(
            np.array([[1.0, 2.0, 4.0, 8.0], [100.0, 100.0, 100.0, 100.0]]),
            1,
        )
        self.assertAlmostEqual(mae_scale, 7.0 / 6.0)
        self.assertAlmostEqual(rmse_scale, float(np.sqrt(21.0 / 6.0)))
        # 常数历史 → 尺度为零 → NaN（调用方按 NaN 传播，不伪造）
        mae_scale, rmse_scale = seasonal_insample_scales(np.ones((1, 5)), 1)
        self.assertTrue(np.isnan(mae_scale))
        self.assertTrue(np.isnan(rmse_scale))
        # 历史长度 <= lag 属显式误用：RAISE
        with self.assertRaises(ValueError):
            seasonal_insample_scales(np.ones((1, 2)), 2)


if __name__ == "__main__":
    unittest.main()
