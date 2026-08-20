# -*- coding: utf-8 -*-

import importlib
import unittest

import numpy as np
import pandas as pd


class LoadStateFeatureDerivationTest(unittest.TestCase):
    def test_build_load_state_features_excludes_target_and_future_labels(self):
        module = importlib.import_module("config.aidc_power_month.derive_load_state_features")
        build = module.build_load_state_features
        frame = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=3, freq="1D"),
                "value": [100.0, 110.0, 120.0],
                "feat_z30_robust": [np.nan, 0.2, 0.3],
                "feat_slope30": [np.nan, 2.0, 3.0],
                "xf_intraday_std": [10.0, 11.0, 12.0],
                "xf_intraday_range": [20.0, 21.0, 22.0],
                "xf_intraday_p95_p5_gap": [18.0, 19.0, 20.0],
                "xf_intraday_cv": [0.1, 0.2, 0.3],
                "xf_intraday_max_abs_step": [5.0, 6.0, 7.0],
                "xf_intraday_peak_time_frac": [0.4, 0.5, 0.6],
                "xf_intraday_range_pct": [2.0, 2.1, 2.2],
                "xr_route_diff_pct": [1.0, 1.1, 1.2],
                "lbl_volatile_day": [0, 1, 0],
                "lbl_event_day": [1, 1, 1],
            }
        )

        result = build(frame)

        self.assertNotIn("value", result.columns)
        self.assertNotIn("lbl_event_day", result.columns)
        self.assertNotIn("lbl_volatile_day", result.columns)
        self.assertEqual(result["state_z30_ready"].tolist(), [0, 1, 1])
        self.assertEqual(result["state_z30_robust"].tolist(), [0.0, 0.2, 0.3])
        self.assertEqual(result["state_volatile_count_7d"].tolist(), [0.0, 1.0, 1.0])
        self.assertFalse(result.isna().any().any())


if __name__ == "__main__":
    unittest.main()
