# -*- coding: utf-8 -*-
"""概率预测 Phase 0 后处理流水线测试。"""

import unittest

import numpy as np
import pandas as pd

from probabilistic.pipeline import finalize_quantile_forecast


class FinalizeQuantileForecastTest(unittest.TestCase):
    def test_processed_quantiles_and_cqr_interval_are_separate(self):
        frame = pd.DataFrame(
            {
                "predict_value": [999.0],
                "predict_q10": [12.0],
                "predict_q50": [10.0],
                "predict_q90": [9.0],
            }
        )

        result, correction = finalize_quantile_forecast(
            frame,
            monotone_enabled=True,
            conformal_scores=np.full(30, 2.0),
            alpha=0.1,
            min_scores=30,
        )

        self.assertEqual(result.loc[0, "predict_value"], 10.0)
        self.assertEqual(result.loc[0, "predict_q10"], 10.0)
        self.assertEqual(result.loc[0, "predict_q90"], 10.0)
        self.assertEqual(result.loc[0, "predict_pi90_lower"], 8.0)
        self.assertEqual(result.loc[0, "predict_pi90_upper"], 12.0)
        self.assertEqual(correction, 2.0)


if __name__ == "__main__":
    unittest.main()
