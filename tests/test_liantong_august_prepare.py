"""联通离线缺口处理：真实日期、仅过去信息；不运行业务模型。"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "config/aidc_electricity_computility/electricity/2026-08-31/liantong_IT"
sys.path.insert(0, str(SCENARIO))
import liantong_august_prepare as prepare


class LiantongAugustPrepareTest(unittest.TestCase):
    def test_target_preserves_dates_observations_and_marks_causal_fill(self):
        times = pd.date_range("2026-08-01", "2026-08-31 23:55", freq="5min")
        values = 100 + np.arange(len(times), dtype=float) / len(times)
        missing = times.normalize() == pd.Timestamp("2026-08-19")
        values[missing] = np.nan
        original = pd.DataFrame({"time": times, "value": values})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.csv"
            original.to_csv(path, index=False)
            before = path.read_bytes()
            result = prepare.prepare_target(path)
            self.assertTrue(pd.DatetimeIndex(result.time).equals(times))
            np.testing.assert_array_equal(result.value[~missing], values[~missing])
            self.assertEqual(list(result.columns), ["time", "value"])
            np.testing.assert_array_equal(result.value[missing], values[:18 * 288].reshape(18, 288).mean(axis=0))
            self.assertTrue(np.isfinite(result.value).all())
            self.assertEqual(path.read_bytes(), before)
            # 改动缺口之后的真值不得改变填充或方法选择。
            changed = original.copy()
            changed.loc[times >= "2026-08-20", "value"] += 10000
            changed.to_csv(path, index=False)
            repeated = prepare.prepare_target(path)
            np.testing.assert_array_equal(result.value[missing], repeated.value[missing])
            self.assertEqual(result.attrs["fill_audit"], repeated.attrs["fill_audit"])

    def test_invalid_target_grid_gap_or_observation_is_rejected(self):
        times = pd.date_range("2026-08-01", "2026-08-31 23:55", freq="5min")
        base = pd.DataFrame({"time": times, "value": 100.0})
        gap = base.time.between("2026-08-19", "2026-08-20", inclusive="left")
        base.loc[gap, "value"] = np.nan
        outside_gap = base.copy()
        outside_gap.loc[0, "value"] = np.nan
        partial_day = base.copy()
        partial_day.loc[gap[gap].index[0], "value"] = 100.0
        nonfinite = base.copy()
        nonfinite.loc[0, "value"] = np.inf
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.csv"
            for frame in (base.iloc[1:], base.iloc[::-1], outside_gap, partial_day, nonfinite):
                with self.subTest(rows=len(frame)):
                    frame.to_csv(path, index=False)
                    with self.assertRaises(ValueError):
                        prepare.prepare_target(path)

    def test_eighteen_day_slot_mean_and_audit(self):
        index = pd.date_range("2026-08-01", "2026-08-19", freq="5min", inclusive="left")
        daily = np.arange(18)[:, None] * 100.0 + np.arange(288)[None, :]
        past = pd.Series(daily.ravel(), index=index)
        predicted, audit = prepare.fill_past_slot_mean(past)
        self.assertEqual(audit["method"], "past_18day_slot_mean")
        np.testing.assert_array_equal(predicted, daily.mean(axis=0))
        self.assertEqual(audit["history_days"], 18)
        self.assertEqual(audit["fill_history_start"], "2026-08-01T00:00:00")
        self.assertEqual(audit["fill_history_end"], "2026-08-18T23:55:00")
        for invalid in (past.iloc[1:], past.iloc[::-1], past * np.nan):
            with self.assertRaises(ValueError):
                prepare.fill_past_slot_mean(invalid)

    def test_weather_preserves_august_dates_and_hourly_values(self):
        hourly = pd.date_range("2026-08-01", "2026-08-31 23:00", freq="1h")
        weather = pd.DataFrame({"ts": hourly})
        for column in ("rt_ssr", "rt_tt2", "rt_dt", "rt_ws10", "rt_ps", "rt_rain", *prepare.PRED_COLUMNS):
            weather[column] = 280.0 + np.arange(len(hourly)) / 100
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weather.csv"
            weather.to_csv(path, index=False)
            before = path.read_bytes()
            combined = prepare.prepare_weather(path)
            self.assertTrue(pd.DatetimeIndex(combined.ts).equals(
                pd.date_range("2026-08-01", "2026-08-31 23:55", freq="5min")))
            expected = pd.read_csv(path).rt_tt2.to_numpy().repeat(12)
            np.testing.assert_array_equal(combined.rt_tt2, expected)
            self.assertEqual(set(prepare.RT_COLUMNS), {"rt_ssr", "rt_tt2", "cal_rh", "rt_ws10", "rt_ps", "rt_rain"})
            for column in ("rt_ps", "rt_rain", "pred_ps", "pred_rain"):
                np.testing.assert_array_equal(combined[column], pd.read_csv(path)[column].to_numpy().repeat(12))
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(combined.ts.iloc[-1], pd.Timestamp("2026-08-31 23:55"))
            weather.loc[0, "pred_tt2"] = np.inf
            weather.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                prepare.prepare_weather(path)

    def test_cli_publishes_audited_original_timeline_and_is_repeatable(self):
        times = pd.date_range("2026-08-01", "2026-08-31 23:55", freq="5min")
        target = pd.DataFrame({"time": times, "value": 123.0})
        target.loc[target.time.between("2026-08-19", "2026-08-20", inclusive="left"), "value"] = np.nan
        weather = pd.DataFrame({"ts": pd.date_range("2026-08-01", periods=744, freq="1h")})
        for column in ("rt_ssr", "rt_tt2", "rt_dt", "rt_ws10", "rt_ps", "rt_rain", *prepare.PRED_COLUMNS):
            weather[column] = 280.0
        weather.loc[weather.ts == "2026-08-04 21:00", "rt_ssr"] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            power_path, weather_path = root / "power.csv", root / "weather.csv"
            target.to_csv(power_path, index=False)
            weather.to_csv(weather_path, index=False)
            input_bytes = {p: p.read_bytes() for p in (power_path, weather_path)}
            command = [sys.executable, str(SCENARIO / "liantong_august_prepare.py"),
                       "--power-csv", str(power_path), "--weather-csv", str(weather_path),
                       "--output-dir", str(root / "outputs")]
            snapshots = []
            for _ in range(2):
                result = subprocess.run(command, cwd=root, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual([entry["rows"] for entry in report["outputs"]], [8928, 8928])
                snapshots.append({p.name: p.read_bytes() for p in (root / "outputs").iterdir()})
            self.assertEqual(snapshots[0], snapshots[1])
            for source, before in input_bytes.items():
                self.assertEqual(source.read_bytes(), before)
            for output in report["outputs"]:
                path = Path(output["file"])
                meta = json.loads(path.with_suffix(".meta.json").read_text())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), meta["sha256_file"])
                self.assertEqual(hashlib.sha256(Path(meta["source"]).read_bytes()).hexdigest(), meta["source_sha256"])
                if meta["role"] == "target":
                    audit = meta["processing_audit"]["fill_audit"]
                    self.assertEqual(audit["method"], "past_18day_slot_mean")
                    self.assertEqual(audit["filled_rows"], 288)
                    frame = pd.read_csv(path)
                    self.assertEqual(list(frame.columns), ["time", "value"])
                    self.assertTrue(frame.value.eq(123.0).all())
                else:
                    self.assertEqual(meta["processing_audit"]["weather_audit"]["night_rt_ssr_zero_filled_times"],
                                     ["2026-08-04T21:00:00"])


if __name__ == "__main__":
    unittest.main()
