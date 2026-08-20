# -*- coding: utf-8 -*-

import importlib
import unittest

import numpy as np
import pandas as pd

from data_process.load_event_detection import EventDetectionConfig


class PowerMonthEventFeatureTest(unittest.TestCase):
    def test_energy_feature_frame_preserves_energy_target_and_power_shape_units(self):
        module = importlib.import_module("config.aidc_power_month.load_event_analysis_1day")
        build_energy_feature_frame = module.build_energy_feature_frame
        days = 70
        idx15 = pd.date_range("2025-01-01", periods=days * 96, freq="15min")
        day_number = np.arange(len(idx15)) // 96
        slot = np.arange(len(idx15)) % 96
        load_15min = pd.Series(
            1000.0 + day_number * 2.0 + 40.0 * np.sin(2 * np.pi * slot / 96),
            index=idx15,
        )
        energy_daily = load_15min.resample("1D").mean() * 24.0
        peer_energy_daily = energy_daily * 1.05

        result = build_energy_feature_frame(
            energy_daily=energy_daily,
            peer_energy_daily=peer_energy_daily,
            load_15min=load_15min,
            cfg=EventDetectionConfig(),
        )
        frame = result["features"]

        self.assertEqual(len(frame), days)
        np.testing.assert_allclose(frame["value"], energy_daily)
        np.testing.assert_allclose(frame["xf_intraday_mean"] * 24.0, frame["value"])
        np.testing.assert_allclose(frame["xr_peer_energy_kwh"], peer_energy_daily)
        np.testing.assert_allclose(
            frame["xr_total_energy_kwh"],
            frame["value"] + frame["xr_peer_energy_kwh"],
        )
        self.assertIn("feat_z30_robust", frame.columns)
        self.assertIn("xf_intraday_range", frame.columns)
        self.assertIn("lbl_event_day", frame.columns)
        self.assertIn("lbl_event_type", frame.columns)


if __name__ == "__main__":
    unittest.main()
