# -*- coding: utf-8 -*-
"""ESS 派生天气严格信息集与历史/未来切分测试。"""
import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "config/aidc_ess_selfuse_load/derive_weather.py"
SPEC = importlib.util.spec_from_file_location("ess_derive_weather", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EssDerivedWeatherContractTest(unittest.TestCase):
    def test_ranges_split_after_last_known_target_point(self):
        self.assertEqual(MODULE.HISTORY_END, pd.Timestamp("2026-07-28 23:55:00"))
        self.assertEqual(MODULE.FUTURE_START, pd.Timestamp("2026-07-29 00:00:00"))
        self.assertLess(MODULE.HISTORY_END, MODULE.FUTURE_START)

    def test_historical_forecast_backtest_uses_pred_columns(self):
        raw_times = pd.date_range("2025-12-31 21:00", periods=5, freq="1h")
        raw = pd.DataFrame({
            "ts": raw_times,
            "pred_ssrd": [1.0, 2.0, 3.0, 40.0, 50.0],
            "pred_tt2": [280.0, 281.0, 282.0, 290.0, 291.0],
            "pred_rh": [50.0, 51.0, 52.0, 60.0, 61.0],
            "pred_ws10": [1.0, 1.1, 1.2, 2.0, 2.1],
        })

        forecast = MODULE._build_historical_forecast(
            raw,
            pd.Timestamp("2026-01-01 00:00:00"),
            pd.Timestamp("2026-01-01 00:10:00"),
        )

        self.assertEqual(len(forecast), 3)
        self.assertEqual(set(forecast["weather_source"]), {"historical_forecast"})
        self.assertEqual(forecast["rt_ssr"].tolist(), [40.0, 40.0, 40.0])
        self.assertEqual(forecast["rt_tt2"].tolist(), [290.0, 290.0, 290.0])
        self.assertEqual(forecast["cal_rh"].tolist(), [60.0, 60.0, 60.0])
        self.assertEqual(
            pd.to_datetime(forecast["available_at"]).unique().tolist(),
            [pd.Timestamp("2025-12-31 23:55:00")],
        )
        self.assertEqual(
            pd.to_datetime(forecast["source_ts"]).tolist(),
            pd.to_datetime(forecast["time"]).tolist(),
        )


if __name__ == "__main__":
    unittest.main()
