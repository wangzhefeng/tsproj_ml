import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "config/aidc_electricity_computility/derive_cal_rh.py"
)
SPEC = importlib.util.spec_from_file_location("computility_derive_cal_rh", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ComputilityWeatherDerivationTest(unittest.TestCase):
    def test_calculate_cal_rh_matches_legacy_formula_and_fill_contract(self):
        tt2 = pd.Series([np.nan, 293.15, 293.15, np.nan])
        dt = pd.Series([np.nan, 283.15, 288.15, np.nan])

        actual = MODULE.calculate_cal_rh(tt2, dt)

        t_air = np.array([20.0, 20.0])
        t_dew = np.array([10.0, 15.0])
        expected_valid = np.clip(
            np.exp(
                17.2693 * t_dew / (237.29 + t_dew)
                - 17.2693 * t_air / (237.29 + t_air)
            )
            * 100,
            0,
            100,
        )
        np.testing.assert_allclose(
            actual.to_numpy(),
            [expected_valid[0], expected_valid[0], expected_valid[1], expected_valid[1]],
        )

    def test_migrate_path_appends_only_cal_rh_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weather.csv"
            original_rows = [
                ["ts", "rt_tt2", "rt_dt", "pred_rh"],
                ["2026-01-01 00:00:00", "", "", "61.5"],
                ["2026-01-01 01:00:00", "293.15000", "283.15000", "62.5"],
                ["2026-01-01 02:00:00", "293.15000", "288.15000", "63.5"],
            ]
            with path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerows(original_rows)

            preview = MODULE.migrate_path(path, write=False)
            self.assertEqual(preview.status, "would_write")
            with path.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(list(csv.reader(stream)), original_rows)

            result = MODULE.migrate_path(path, write=True)
            self.assertEqual(result.status, "written")
            with path.open(newline="", encoding="utf-8") as stream:
                migrated_rows = list(csv.reader(stream))
            self.assertEqual(
                [row[:-1] for row in migrated_rows],
                original_rows,
            )
            self.assertEqual(migrated_rows[0][-1], "cal_rh")
            self.assertTrue(all(np.isfinite(float(row[-1])) for row in migrated_rows[1:]))

            repeated = MODULE.migrate_path(path, write=True)
            self.assertEqual(repeated.status, "unchanged")
            with path.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(list(csv.reader(stream)), migrated_rows)


if __name__ == "__main__":
    unittest.main()
