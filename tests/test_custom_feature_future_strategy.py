# -*- coding: utf-8 -*-

import importlib
import unittest

import pandas as pd


class CustomFeatureFreezeStrategyTest(unittest.TestCase):
    def test_freeze_last_observation_uses_only_rows_at_or_before_cutoff(self):
        module = importlib.import_module("data_provider.data_loader")
        materialize = module.materialize_custom_future_sources
        history = [
            {
                "name": "load_state",
                "ts_col": "time",
                "columns": ["state_value"],
                "categorical_columns": [],
                "future_strategy": "freeze_last_observation",
                "availability": "end_of_period",
                "df": pd.DataFrame(
                    {
                        "time": pd.date_range("2026-01-01", periods=10, freq="1D"),
                        "state_value": range(1, 11),
                    }
                ),
            }
        ]
        future_times = pd.date_range("2026-01-11", periods=3, freq="1D")

        resolved = materialize(
            custom_history=history,
            custom_future=[],
            future_times=future_times,
            cutoff=pd.Timestamp("2026-01-07"),
        )

        self.assertEqual(len(resolved), 1)
        frame = resolved[0]["df"]
        self.assertEqual(resolved[0]["availability"], "end_of_period")
        self.assertEqual(frame["time"].tolist(), future_times.tolist())
        self.assertEqual(frame["state_value"].tolist(), [7, 7, 7])


if __name__ == "__main__":
    unittest.main()
