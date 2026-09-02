# -*- coding: utf-8 -*-
"""活动配置 catalog 必须由两类生产 typed spec 构建。"""

import unittest
from collections import Counter
from pathlib import Path

from scripts.audit_forecast_configs import REQUIRED_KEYS, build_catalog


ROOT = Path(__file__).resolve().parents[1]


class ForecastConfigAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = build_catalog(ROOT / "config")

    def test_catalog_covers_every_active_model_yaml(self):
        self.assertEqual(len(self.rows), 707)
        self.assertEqual(
            Counter(row["config_kind"] for row in self.rows),
            {"single_model": 677, "ensemble": 30},
        )
        self.assertEqual(len({row["path"] for row in self.rows}), 707)

    def test_catalog_has_typed_canonical_fields(self):
        self.assertEqual(set(REQUIRED_KEYS), set(self.rows[0]))
        for row in self.rows:
            self.assertEqual(set(row), set(REQUIRED_KEYS))
            self.assertEqual(len(row["fingerprint"]), 64)
            self.assertEqual(row["schema_version"], 2)
            self.assertIn(row["probabilistic_mode"], {"point", "quantile"})
            self.assertGreater(row["horizon"], 0)
            self.assertTrue(row["targets"])
            if row["config_kind"] == "single_model":
                self.assertIsNotNone(row["strategy"])
                self.assertIsNotNone(row["model_type"])
                self.assertIsNone(row["ensemble_method"])
                self.assertEqual(row["member_count"], 0)
            else:
                self.assertIsNone(row["strategy"])
                self.assertIsNone(row["model_type"])
                self.assertIn(
                    row["ensemble_method"],
                    {"averaging", "weighted", "linear_blending", "stacking"},
                )
                self.assertGreaterEqual(row["member_count"], 2)


if __name__ == "__main__":
    unittest.main()
