# -*- coding: utf-8 -*-
"""ESS严格外生数据在DataLoader/CV层的集成契约。"""
import unittest
from types import SimpleNamespace

import pandas as pd

from data_provider.data_loader import DataLoader
from models.ModelTesting import Tester


class StrictDataLoaderSplitTest(unittest.TestCase):
    def _loader(self):
        args = SimpleNamespace(freq="5min")
        return DataLoader(
            args=args,
            train_start_time=pd.Timestamp("2026-07-01"),
            train_end_time=pd.Timestamp("2026-07-29"),
            forecast_start_time=pd.Timestamp("2026-07-29"),
            forecast_end_time=pd.Timestamp("2026-07-30"),
            log_prefix="[test]",
        )

    def test_strict_role_split_does_not_canonical_merge(self):
        history = pd.DataFrame({
            "time": pd.to_datetime(["2026-07-28 23:50", "2026-07-28 23:55"]),
            "value": [1.0, 2.0],
        })
        future = pd.DataFrame({
            "time": pd.to_datetime(["2026-07-29 00:00", "2026-07-29 00:05"]),
            "value": [3.0, 4.0],
        })
        canonical, history_out, future_out = self._loader()._prepare_exogenous_splits(
            history_df=history,
            future_df=future,
            ts_col="time",
            label="weather",
            strict_roles=True,
        )
        self.assertIsNone(canonical)
        self.assertEqual(history_out["value"].tolist(), [1.0, 2.0])
        self.assertEqual(future_out["value"].tolist(), [3.0, 4.0])


class FoldWeatherAsOfTest(unittest.TestCase):
    def test_fold_uses_latest_forecast_available_at_origin(self):
        args = SimpleNamespace(
            enable_weather_features=True,
            strict_weather_information_set=True,
            weather_ts_feat="time",
        )
        test_frame = pd.DataFrame({
            "time": pd.to_datetime(["2026-07-29 00:00", "2026-07-29 00:05"]),
        })
        backtest = pd.DataFrame({
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
            "rt_tt2": [10.0, 99.0, 20.0],
        })
        selected = Tester._resolve_window_weather_future(
            args=args,
            df_history_test=test_frame,
            df_weather_history=None,
            df_weather_backtest=backtest,
            fold_origin=pd.Timestamp("2026-07-28 23:55"),
            log_prefix="[test]",
        )
        self.assertEqual(selected["rt_tt2"].tolist(), [10.0, 20.0])

    def test_fold_rejects_forecast_published_after_origin(self):
        args = SimpleNamespace(
            enable_weather_features=True,
            strict_weather_information_set=True,
            weather_ts_feat="time",
        )
        test_frame = pd.DataFrame({"time": pd.to_datetime(["2026-07-29 00:00"])})
        backtest = pd.DataFrame({
            "time": pd.to_datetime(["2026-07-29 00:00"]),
            "available_at": pd.to_datetime(["2026-07-29 00:01"]),
            "rt_tt2": [10.0],
        })
        with self.assertRaisesRegex(ValueError, "forecast origin"):
            Tester._resolve_window_weather_future(
                args=args,
                df_history_test=test_frame,
                df_weather_history=None,
                df_weather_backtest=backtest,
                fold_origin=pd.Timestamp("2026-07-28 23:55"),
                log_prefix="[test]",
            )


if __name__ == "__main__":
    unittest.main()
