import unittest

import pandas as pd

from config.aidc_ess_selfuse_load.strategy_features.windows import (
    audit_history_timestamps,
    calendar_day_slot,
    dispatch_cycle_slot,
    dispatch_cycle_start,
    validate_future_timestamps,
)


class EssStrategyWindowsTest(unittest.TestCase):
    def test_calendar_day_slots_cover_full_day(self):
        timestamps = pd.DatetimeIndex(
            ["2026-08-17 00:00:00", "2026-08-17 12:00:00", "2026-08-17 23:55:00"]
        )

        self.assertEqual(calendar_day_slot(timestamps).tolist(), [0, 144, 287])

    def test_dispatch_cycle_slots_start_at_2200(self):
        timestamps = pd.DatetimeIndex(
            ["2026-08-17 22:00:00", "2026-08-18 00:00:00", "2026-08-18 21:55:00"]
        )

        self.assertEqual(dispatch_cycle_slot(timestamps).tolist(), [0, 24, 287])

    def test_timestamp_maps_to_correct_dispatch_cycle_start(self):
        timestamps = pd.DatetimeIndex(
            [
                "2026-08-17 21:55:00",
                "2026-08-17 22:00:00",
                "2026-08-18 00:00:00",
                "2026-08-18 21:55:00",
            ]
        )

        self.assertEqual(
            dispatch_cycle_start(timestamps).tolist(),
            [
                pd.Timestamp("2026-08-16 22:00:00"),
                pd.Timestamp("2026-08-17 22:00:00"),
                pd.Timestamp("2026-08-17 22:00:00"),
                pd.Timestamp("2026-08-17 22:00:00"),
            ],
        )

    def test_future_validator_rejects_missing_duplicate_and_off_grid_timestamps(self):
        complete = pd.date_range("2026-08-18", periods=288, freq="5min")
        validate_future_timestamps(complete)

        cases = {
            "missing": complete.delete(100),
            "duplicate": complete.insert(100, complete[100]),
            "off_grid": complete.delete(100).insert(100, complete[100] + pd.Timedelta(minutes=1)),
        }
        for name, timestamps in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_future_timestamps(timestamps)

    def test_history_validator_reports_incomplete_days_without_failing(self):
        day1 = pd.date_range("2026-08-15", periods=288, freq="5min")
        day2 = pd.date_range("2026-08-16", periods=288, freq="5min").delete(42)

        audit = audit_history_timestamps(day1.append(day2))

        self.assertEqual(audit.incomplete_days, (pd.Timestamp("2026-08-16"),))
        self.assertEqual(audit.missing_slots_by_day[pd.Timestamp("2026-08-16")], (42,))
        self.assertFalse(audit.has_duplicates)
        self.assertFalse(audit.has_off_grid_timestamps)

    def test_history_validator_reports_wholly_missing_day(self):
        day1 = pd.date_range("2026-08-15", periods=288, freq="5min")
        day3 = pd.date_range("2026-08-17", periods=288, freq="5min")

        audit = audit_history_timestamps(day1.append(day3))

        missing_day = pd.Timestamp("2026-08-16")
        self.assertIn(missing_day, audit.incomplete_days)
        missing_slots = next(
            slots
            for day, slots in audit.missing_slots_by_day.items()
            if day == missing_day
        )
        self.assertEqual(missing_slots, tuple(range(288)))


if __name__ == "__main__":
    unittest.main()
