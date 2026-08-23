# -*- coding: utf-8 -*-
"""通用外生信息集契约测试。"""
import unittest

import pandas as pd

from utils.exogenous_contract import (
    select_asof_rows,
    split_role_frames,
    validate_daily_coverage,
    validate_exact_coverage,
)
from data_provider.data_loader import materialize_custom_future_sources


class RoleSplitContractTest(unittest.TestCase):
    def test_strict_split_rejects_overlap(self):
        history = pd.DataFrame({
            "time": pd.to_datetime(["2026-07-28 23:55", "2026-07-29 00:00"]),
            "value": [1.0, 2.0],
        })
        future = pd.DataFrame({
            "time": pd.to_datetime(["2026-07-29 00:00", "2026-07-29 00:05"]),
            "value": [3.0, 4.0],
        })
        with self.assertRaisesRegex(ValueError, "overlap"):
            split_role_frames(
                history,
                future,
                ts_col="time",
                forecast_start=pd.Timestamp("2026-07-29 00:00"),
                label="weather",
            )

    def test_strict_split_requires_exact_boundary(self):
        history = pd.DataFrame({
            "time": pd.to_datetime(["2026-07-28 23:55"]),
            "value": [1.0],
        })
        future = pd.DataFrame({
            "time": pd.to_datetime(["2026-07-29 00:05"]),
            "value": [2.0],
        })
        with self.assertRaisesRegex(ValueError, "start"):
            split_role_frames(
                history,
                future,
                ts_col="time",
                forecast_start=pd.Timestamp("2026-07-29 00:00"),
                label="weather",
            )

    def test_strict_split_keeps_roles_separate(self):
        history = pd.DataFrame({
            "time": pd.to_datetime(["2026-07-28 23:50", "2026-07-28 23:55"]),
            "value": [1.0, 2.0],
        })
        future = pd.DataFrame({
            "time": pd.to_datetime(["2026-07-29 00:00", "2026-07-29 00:05"]),
            "value": [3.0, 4.0],
        })
        history_out, future_out = split_role_frames(
            history,
            future,
            ts_col="time",
            forecast_start=pd.Timestamp("2026-07-29 00:00"),
            label="weather",
        )
        self.assertEqual(history_out["value"].tolist(), [1.0, 2.0])
        self.assertEqual(future_out["value"].tolist(), [3.0, 4.0])


class AsOfSelectionContractTest(unittest.TestCase):
    def test_selects_latest_version_available_at_origin(self):
        frame = pd.DataFrame({
            "time": pd.to_datetime([
                "2026-07-29 00:00",
                "2026-07-29 00:00",
                "2026-07-29 00:05",
            ]),
            "available_at": pd.to_datetime([
                "2026-07-28 18:00",
                "2026-07-29 00:01",
                "2026-07-28 18:00",
            ]),
            "value": [10.0, 99.0, 20.0],
        })
        selected = select_asof_rows(
            frame,
            expected_times=pd.to_datetime(["2026-07-29 00:00", "2026-07-29 00:05"]),
            forecast_origin=pd.Timestamp("2026-07-28 23:55"),
            ts_col="time",
            available_at_col="available_at",
            label="plan",
        )
        self.assertEqual(selected["value"].tolist(), [10.0, 20.0])

    def test_rejects_rows_only_available_after_origin(self):
        frame = pd.DataFrame({
            "time": pd.to_datetime(["2026-07-29 00:00"]),
            "available_at": pd.to_datetime(["2026-07-29 00:01"]),
            "value": [10.0],
        })
        with self.assertRaisesRegex(ValueError, "forecast origin"):
            select_asof_rows(
                frame,
                expected_times=pd.to_datetime(["2026-07-29 00:00"]),
                forecast_origin=pd.Timestamp("2026-07-28 23:55"),
                ts_col="time",
                available_at_col="available_at",
                label="plan",
            )

    def test_exact_coverage_rejects_missing_timestamp(self):
        frame = pd.DataFrame({
            "time": pd.to_datetime(["2026-07-29 00:00"]),
            "value": [10.0],
        })
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_exact_coverage(
                frame,
                expected_times=pd.to_datetime(["2026-07-29 00:00", "2026-07-29 00:05"]),
                ts_col="time",
                label="plan",
            )


class DateAndCustomStrictContractTest(unittest.TestCase):
    def test_date_requires_every_target_day(self):
        calendar = pd.DataFrame({
            "date": pd.to_datetime(["2026-07-28", "2026-07-30"]),
            "date_type": [1, 1],
        })
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_daily_coverage(
                calendar,
                expected_times=pd.to_datetime([
                    "2026-07-28 00:00",
                    "2026-07-29 12:00",
                    "2026-07-30 23:55",
                ]),
                ts_col="date",
                value_columns=["date_type"],
                label="date",
            )

    def test_strict_explicit_custom_selects_asof_target_rows(self):
        future_times = pd.to_datetime(["2026-07-29 00:00", "2026-07-29 00:05"])
        source = {
            "name": "pcs_plan",
            "ts_col": "time",
            "columns": ["pcs_plan"],
            "categorical_columns": [],
            "future_strategy": "explicit",
            "availability": "forecast_origin",
            "strict_information_set": True,
            "available_at_col": "available_at",
            "df": pd.DataFrame({
                "time": future_times,
                "available_at": pd.to_datetime([
                    "2026-07-28 23:55", "2026-07-28 23:55",
                ]),
                "pcs_plan": [10.0, 20.0],
            }),
        }
        resolved = materialize_custom_future_sources(
            custom_history=None,
            custom_future=[source],
            future_times=future_times,
            cutoff=pd.Timestamp("2026-07-28 23:55"),
        )
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["df"]["pcs_plan"].tolist(), [10.0, 20.0])

    def test_strict_explicit_custom_rejects_missing_target(self):
        future_times = pd.to_datetime(["2026-07-29 00:00", "2026-07-29 00:05"])
        source = {
            "name": "pcs_plan",
            "ts_col": "time",
            "columns": ["pcs_plan"],
            "future_strategy": "explicit",
            "strict_information_set": True,
            "available_at_col": "available_at",
            "df": pd.DataFrame({
                "time": future_times[:1],
                "available_at": pd.to_datetime(["2026-07-28 23:55"]),
                "pcs_plan": [10.0],
            }),
        }
        with self.assertRaisesRegex(ValueError, "missing"):
            materialize_custom_future_sources(
                custom_history=None,
                custom_future=[source],
                future_times=future_times,
                cutoff=pd.Timestamp("2026-07-28 23:55"),
            )


if __name__ == "__main__":
    unittest.main()
