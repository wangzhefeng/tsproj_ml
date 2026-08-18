import math
import unittest

import pandas as pd

from config.aidc_ess_selfuse_load.strategy_features.profiles import (
    summarize_dispatch_profiles,
)
from config.aidc_ess_selfuse_load.strategy_features.states import (
    encode_actual_operating_state,
)


class EssStrategyProfilesTest(unittest.TestCase):
    def test_profile_summary_calculates_operating_statistics(self):
        timestamps = pd.date_range("2026-08-17 22:00:00", periods=7, freq="5min")
        power = pd.Series([-2000.0, -2000.0, 0.0, 6000.0, 7000.0, 0.0, -2500.0])
        states = encode_actual_operating_state(power)
        frame = pd.concat(
            [pd.Series(timestamps, name="time"), power.rename("power_kw"), states],
            axis=1,
        )

        summary = summarize_dispatch_profiles(frame).iloc[0]

        self.assertAlmostEqual(summary["charge_hours"], 3 * 5 / 60)
        self.assertAlmostEqual(summary["standby_hours"], 2 * 5 / 60)
        self.assertAlmostEqual(summary["discharge_hours"], 2 * 5 / 60)
        self.assertEqual(summary["charge_segment_count"], 2)
        self.assertEqual(summary["standby_segment_count"], 2)
        self.assertEqual(summary["discharge_segment_count"], 1)
        self.assertAlmostEqual(summary["charge_energy_kwh"], -6500 * 5 / 60)
        self.assertAlmostEqual(summary["discharge_energy_kwh"], 13000 * 5 / 60)
        self.assertEqual(summary["switch_count"], 4)
        self.assertEqual(summary["max_ramp_kw"], 7000.0)
        self.assertEqual(summary["first_charge_slot"], 0)
        self.assertEqual(summary["first_discharge_slot"], 3)
        self.assertAlmostEqual(summary["first_charge_slot_sin"], 0.0)
        self.assertAlmostEqual(summary["first_charge_slot_cos"], 1.0)
        self.assertAlmostEqual(
            summary["first_discharge_slot_sin"], math.sin(2 * math.pi * 3 / 288)
        )
        self.assertAlmostEqual(
            summary["first_discharge_slot_cos"], math.cos(2 * math.pi * 3 / 288)
        )
        self.assertTrue(summary["has_charge"])
        self.assertTrue(summary["has_discharge"])

    def test_charge_segment_crossing_midnight_remains_one_dispatch_segment(self):
        timestamps = pd.date_range("2026-08-17 23:55:00", periods=3, freq="5min")
        power = pd.Series([-2000.0, -2100.0, -2200.0])
        states = encode_actual_operating_state(power)
        frame = pd.concat(
            [pd.Series(timestamps, name="time"), power.rename("power_kw"), states],
            axis=1,
        )

        summary = summarize_dispatch_profiles(frame).iloc[0]

        self.assertEqual(summary["cycle_start"], pd.Timestamp("2026-08-17 22:00:00"))
        self.assertEqual(summary["charge_segment_count"], 1)
        self.assertEqual(summary["first_charge_slot"], 23)
        self.assertEqual(summary["switch_count"], 0)

    def test_absent_state_uses_flags_and_missing_first_slot(self):
        timestamps = pd.date_range("2026-08-17 22:00:00", periods=2, freq="5min")
        power = pd.Series([0.0, 0.0])
        states = encode_actual_operating_state(power)
        frame = pd.concat(
            [pd.Series(timestamps, name="time"), power.rename("power_kw"), states],
            axis=1,
        )

        summary = summarize_dispatch_profiles(frame).iloc[0]

        self.assertFalse(summary["has_charge"])
        self.assertFalse(summary["has_discharge"])
        self.assertTrue(pd.isna(summary["first_charge_slot"]))
        self.assertTrue(pd.isna(summary["first_discharge_slot"]))
        self.assertEqual(summary["first_charge_slot_sin"], 0.0)
        self.assertEqual(summary["first_charge_slot_cos"], 0.0)
        self.assertEqual(summary["first_discharge_slot_sin"], 0.0)
        self.assertEqual(summary["first_discharge_slot_cos"], 0.0)

    def test_profile_summary_rejects_rows_that_are_not_exactly_one_hot(self):
        timestamps = pd.date_range("2026-08-17 22:00:00", periods=2, freq="5min")
        base = pd.DataFrame(
            {
                "time": timestamps,
                "power_kw": [0.0, 0.0],
                "actual_operating_charge": [0.0, 0.0],
                "actual_operating_standby": [1.0, 1.0],
                "actual_operating_discharge": [0.0, 0.0],
            }
        )
        cases = {
            "no_active_state": (0, [0, 0, 0]),
            "multiple_active_states": (1, [1, 0, 1]),
            "non_binary_state": (0, [0, 0.5, 0.5]),
        }
        for name, (row_index, values) in cases.items():
            with self.subTest(name=name):
                frame = base.copy()
                frame.loc[
                    row_index,
                    [
                        "actual_operating_charge",
                        "actual_operating_standby",
                        "actual_operating_discharge",
                    ],
                ] = values
                with self.assertRaises(ValueError):
                    summarize_dispatch_profiles(frame)

    def test_profile_summary_rejects_non_finite_power(self):
        timestamps = pd.date_range("2026-08-17 22:00:00", periods=2, freq="5min")
        frame = pd.DataFrame(
            {
                "time": timestamps,
                "power_kw": [0.0, float("inf")],
                "actual_operating_charge": [0, 0],
                "actual_operating_standby": [1, 0],
                "actual_operating_discharge": [0, 1],
            }
        )

        with self.assertRaises(ValueError):
            summarize_dispatch_profiles(frame)


if __name__ == "__main__":
    unittest.main()
