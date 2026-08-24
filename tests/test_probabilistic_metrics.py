# -*- coding: utf-8 -*-
"""概率预测指标的手算契约测试。"""

import unittest

import numpy as np

from probabilistic.metrics import crossing_metrics, interval_metrics, pinball_loss


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


if __name__ == "__main__":
    unittest.main()
