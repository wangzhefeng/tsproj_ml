# -*- coding: utf-8 -*-
"""Conformal CQR 校准的单测。"""

import unittest

import numpy as np

from utils.conformal import compute_nonconformity_scores, calibrate_quantile_band


class TestNonconformityScores(unittest.TestCase):
    def test_negative_score_when_y_inside_interval(self):
        """CQR score 允许为负：y 在区间内时 q_low-y<0 且 y-q_high<0。"""
        y = np.array([5.0, 5.0, 5.0])
        q_low = np.array([4.0, 5.0, 3.0])
        q_high = np.array([6.0, 5.0, 7.0])
        scores = compute_nonconformity_scores(y, q_low, q_high)
        # 点0: max(4-5, 5-6) = -1; 点1: max(5-5, 5-5)=0; 点2: max(3-5, 5-7)=-2
        np.testing.assert_array_almost_equal(scores, [-1.0, 0.0, -2.0])

    def test_positive_score_when_y_outside(self):
        y = np.array([10.0, 0.0])
        q_low = np.array([4.0, 2.0])
        q_high = np.array([6.0, 8.0])
        scores = compute_nonconformity_scores(y, q_low, q_high)
        # y=10 above q_high=6 → score=4; y=0 below q_low=2 → score=2
        np.testing.assert_array_almost_equal(scores, [4.0, 2.0])

    def test_length_truncation(self):
        y = np.array([1.0, 2.0, 3.0])
        q_low = np.array([0.0])
        q_high = np.array([5.0])
        scores = compute_nonconformity_scores(y, q_low, q_high)
        self.assertEqual(len(scores), 1)


class TestCalibrateQuantileBand(unittest.TestCase):
    def test_insufficient_scores_returns_none(self):
        q_low = np.array([1.0, 2.0])
        q_high = np.array([3.0, 4.0])
        scores = np.array([0.1, 0.2])  # < min_scores=30
        cal_low, cal_high, E = calibrate_quantile_band(q_low, q_high, scores, alpha=0.1, min_scores=30)
        self.assertIsNone(cal_low)
        self.assertIsNone(cal_high)
        self.assertIsNone(E)

    def test_symmetric_expansion(self):
        q_low = np.array([10.0, 20.0])
        q_high = np.array([12.0, 22.0])
        # 构造 50 个 score 全为 1.0
        scores = np.ones(50)
        cal_low, cal_high, E = calibrate_quantile_band(q_low, q_high, scores, alpha=0.1, min_scores=10)
        self.assertIsNotNone(E)
        self.assertAlmostEqual(E, 1.0, places=4)
        np.testing.assert_array_almost_equal(cal_low, [9.0, 19.0])
        np.testing.assert_array_almost_equal(cal_high, [13.0, 23.0])

    def test_E_alpha_uses_plus_one_quantile(self):
        # scores = [0..99]，alpha=0.1，n=100
        # level = ceil(0.9 * 101) / 100 = ceil(90.9) / 100 = 91/100 = 0.91
        # method="higher" → 取 >= 0.91 分位的最小值 = scores[90] = 90
        scores = np.arange(100, dtype=float)
        q_low = np.array([0.0])
        q_high = np.array([1.0])
        _, _, E = calibrate_quantile_band(q_low, q_high, scores, alpha=0.1, min_scores=10)
        self.assertAlmostEqual(E, 91.0, places=4)

    def test_nan_scores_filtered(self):
        q_low = np.array([1.0])
        q_high = np.array([3.0])
        scores = np.array([0.5] * 40 + [np.nan] * 10)
        cal_low, cal_high, E = calibrate_quantile_band(q_low, q_high, scores, alpha=0.1, min_scores=30)
        self.assertIsNotNone(E)


if __name__ == "__main__":
    unittest.main()
