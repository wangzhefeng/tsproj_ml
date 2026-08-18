# -*- coding: utf-8 -*-

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_0712 = ROOT / "config/aidc_electricity_computility/electricity/2026-07-12/scripts/process_power.py"
SCRIPT_0814 = ROOT / "config/aidc_electricity_computility/electricity/2026-08-14/scripts/process_power.py"


def load_script(path: Path):
    spec = importlib.util.spec_from_file_location(f"process_power_{path.parent.parent.name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AidcPointPowerProcessingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_script(SCRIPT_0712)

    def _write_config(
        self,
        root: Path,
        *,
        start: str,
        end: str,
        enabled: bool = True,
        max_fill_gap_slots: int = 288,
    ) -> Path:
        config = {
            "scene": "unit/route_A",
            "enabled": enabled,
            "disabled_reason": "fixture disabled" if not enabled else "",
            "dataset_dir": str(root),
            "start_time": start,
            "end_time": end,
            "freq": "5min",
            "time_col": "time",
            "value_col": "value",
            "input_glob": "*.csv",
            "exclude_globs": [
                "date_*.csv",
                "weather_*.csv",
                "df_power.csv",
                "df_power_audit.json",
            ],
            "output_file": "df_power.csv",
            "audit_file": "df_power_audit.json",
            "max_fill_gap_slots": max_fill_gap_slots,
            "outlier": {
                "spike_half_window": 2,
                "spike_z_threshold": 8.0,
                "spike_abs_diff_mad_multiplier": 8.0,
                "spike_min_abs_diff": 5.0,
            },
        }
        path = root / "data_process.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return path

    @staticmethod
    def _write_point(path: Path, times: pd.DatetimeIndex, values) -> None:
        pd.DataFrame({"time": times, "value": values}).to_csv(path, index=False)

    def test_date_scripts_are_complete_and_identical(self):
        self.assertTrue(SCRIPT_0712.exists())
        self.assertTrue(SCRIPT_0814.exists())
        self.assertEqual(SCRIPT_0712.read_bytes(), SCRIPT_0814.read_bytes())
        second = load_script(SCRIPT_0814)
        self.assertTrue(callable(second.process_scene))
        self.assertTrue(callable(second.run_config_root))

    def test_discovery_excludes_external_and_generated_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in [
                "point_b.csv",
                "point_a.csv",
                "date_in.csv",
                "weather_in.csv",
                "df_power.csv",
            ]:
                (root / name).write_text("time,value\n", encoding="utf-8")
            config_path = self._write_config(
                root,
                start="2026-01-01 00:00:00",
                end="2026-01-01 00:05:00",
            )
            config = self.module.load_process_config(config_path)

            files = self.module.discover_input_files(config)

            self.assertEqual([path.name for path in files], ["point_a.csv", "point_b.csv"])

    def test_point_gaps_are_filled_before_summing_and_zero_is_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            full_index = pd.date_range("2026-01-01 00:00:00", periods=5, freq="5min")
            self._write_point(root / "point_a.csv", full_index.delete(2), [0.0, 2.0, 6.0, 8.0])
            self._write_point(root / "point_b.csv", full_index, [10.0, 10.0, 10.0, 10.0, 10.0])
            config_path = self._write_config(
                root,
                start="2026-01-01 00:00:00",
                end="2026-01-01 00:20:00",
            )

            result = self.module.process_scene(config_path)

            output = pd.read_csv(root / "df_power.csv")
            self.assertEqual(result["status"], "success")
            self.assertEqual(list(output.columns), ["time", "value"])
            self.assertEqual(output["value"].tolist(), [10.0, 12.0, 14.0, 16.0, 18.0])
            self.assertEqual(output.loc[0, "value"], 10.0, "zero must remain a valid point value")
            audit = json.loads((root / "df_power_audit.json").read_text(encoding="utf-8"))
            point_a = next(item for item in audit["points"] if item["source_file"] == "point_a.csv")
            self.assertEqual(point_a["missing_before_fill"], 1)
            self.assertEqual(point_a["filled_value_count"], 1)

    def test_negative_nonfinite_and_isolated_spike_are_repaired(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            full_index = pd.date_range("2026-01-01 00:00:00", periods=21, freq="5min")
            values = [10.0 + (i % 3) for i in range(21)]
            values[5] = -1.0
            values[10] = 1000.0
            values[15] = np.inf
            self._write_point(root / "point.csv", full_index, values)
            config_path = self._write_config(
                root,
                start="2026-01-01 00:00:00",
                end="2026-01-01 01:40:00",
            )

            self.module.process_scene(config_path)

            output = pd.read_csv(root / "df_power.csv")
            self.assertTrue(np.isfinite(output["value"]).all())
            self.assertGreaterEqual(float(output["value"].min()), 0.0)
            self.assertLess(float(output["value"].max()), 20.0)
            audit = json.loads((root / "df_power_audit.json").read_text(encoding="utf-8"))
            point = audit["points"][0]
            self.assertEqual(point["physical_invalid_count"], 2)
            self.assertEqual(point["spike_count"], 1)

    def test_gap_limit_allows_288_slots_and_rejects_289_slots(self):
        start = pd.Timestamp("2026-01-01 00:00:00")
        for gap_slots, should_raise in [(288, False), (289, True)]:
            with self.subTest(gap_slots=gap_slots), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                full_index = pd.date_range(start, periods=gap_slots + 2, freq="5min")
                self._write_point(root / "point.csv", full_index[[0, -1]], [10.0, 20.0])
                config_path = self._write_config(
                    root,
                    start=str(full_index[0]),
                    end=str(full_index[-1]),
                    max_fill_gap_slots=288,
                )

                if should_raise:
                    with self.assertRaisesRegex(ValueError, "289.*288"):
                        self.module.process_scene(config_path)
                    self.assertFalse((root / "df_power.csv").exists())
                else:
                    self.module.process_scene(config_path)
                    output = pd.read_csv(root / "df_power.csv")
                    self.assertEqual(len(output), gap_slots + 2)
                    self.assertEqual(int(output["value"].isna().sum()), 0)

    def test_duplicate_timestamp_keeps_last_value_and_audit_is_complete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frame = pd.DataFrame(
                {
                    "time": [
                        "2026-01-01 00:00:00",
                        "2026-01-01 00:00:00",
                        "2026-01-01 00:05:00",
                    ],
                    "value": [1.0, 3.0, 5.0],
                }
            )
            frame.to_csv(root / "point.csv", index=False)
            config_path = self._write_config(
                root,
                start="2026-01-01 00:00:00",
                end="2026-01-01 00:05:00",
            )

            self.module.process_scene(config_path)

            output = pd.read_csv(root / "df_power.csv")
            self.assertEqual(output["value"].tolist(), [3.0, 5.0])
            audit = json.loads((root / "df_power_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "success")
            self.assertEqual(audit["output"]["rows"], 2)
            self.assertEqual(audit["points"][0]["duplicate_timestamp_count"], 1)
            self.assertEqual(audit["output"]["start_time"], "2026-01-01 00:00:00")
            self.assertEqual(audit["output"]["end_time"], "2026-01-01 00:05:00")

    def test_disabled_scene_is_skipped_without_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = self._write_config(
                root,
                start="2026-01-01 00:00:00",
                end="2026-01-01 00:05:00",
                enabled=False,
            )

            result = self.module.process_scene(config_path)

            self.assertEqual(result["status"], "skipped")
            self.assertEqual(result["reason"], "fixture disabled")
            self.assertFalse((root / "df_power.csv").exists())
            self.assertFalse((root / "df_power_audit.json").exists())


if __name__ == "__main__":
    unittest.main()
