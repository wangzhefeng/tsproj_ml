# -*- coding: utf-8 -*-
"""概率分位数 crossing 后处理测试。"""

import unittest

import numpy as np
import pandas as pd

from probabilistic.postprocessing import repair_quantile_crossing
from utils.quantile import monotonize_quantile_columns


class QuantileCrossingRepairTest(unittest.TestCase):
    def test_q50_anchor_is_unchanged_and_point_is_synchronized(self):
        frame = pd.DataFrame(
            {
                "predict_value": [999.0, 999.0],
                "predict_q10": [12.0, 8.0],
                "predict_q50": [10.0, 10.0],
                "predict_q90": [9.0, 14.0],
            }
        )

        repaired = repair_quantile_crossing(frame, enabled=True)

        np.testing.assert_array_equal(repaired["predict_q50"], [10.0, 10.0])
        np.testing.assert_array_equal(repaired["predict_value"], [10.0, 10.0])
        np.testing.assert_array_equal(repaired["predict_q10"], [10.0, 8.0])
        np.testing.assert_array_equal(repaired["predict_q90"], [10.0, 14.0])

    def test_legacy_entrypoint_uses_q50_anchor(self):
        frame = pd.DataFrame(
            {
                "predict_value": [999.0],
                "predict_q10": [12.0],
                "predict_q50": [9.0],
                "predict_q90": [10.0],
            }
        )

        repaired = monotonize_quantile_columns(frame, enabled=True)

        self.assertEqual(repaired.loc[0, "predict_q50"], 9.0)
        self.assertEqual(repaired.loc[0, "predict_value"], 9.0)


if __name__ == "__main__":
    unittest.main()
