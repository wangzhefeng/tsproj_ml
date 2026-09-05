# -*- coding: utf-8 -*-
"""活动配置 batch eligibility inventory 测试。"""
from __future__ import annotations

import unittest
from pathlib import Path

from scripts.audit_batch_eligibility import audit_configs


class BatchEligibilityAuditTest(unittest.TestCase):
    def test_inventory_uses_compiler_decision_for_batch_and_row_configs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        paths = (
            root
            / "config/aidc_load_15min_short/route_A/baseline/lgbm_direct.yaml",
            root
            / (
                "config/aidc_power_month/route_A/freq_1day/baseline/"
                "ensemble_members/lgbm_usbr_prob_mean_conformal_recursive.yaml"
            ),
        )

        rows = audit_configs(paths)

        self.assertEqual(len(rows), 2)
        by_path = {Path(row["path"]).name: row for row in rows}
        self.assertTrue(by_path["lgbm_direct.yaml"]["eligible"])
        recursive = by_path["lgbm_usbr_prob_mean_conformal_recursive.yaml"]
        self.assertFalse(recursive["eligible"])
        self.assertIn("provider_dependent_lag", recursive["reason_codes"])
        self.assertGreater(recursive["estimated_origin_call_count"], 0)


if __name__ == "__main__":
    unittest.main()
