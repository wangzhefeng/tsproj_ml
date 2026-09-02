# -*- coding: utf-8 -*-
"""引用式 Ensemble 审计必须验证成员实际为单模型 spec。"""

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_ensemble_configs.py"
SPEC = importlib.util.spec_from_file_location("ensemble_config_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EnsembleConfigAuditTest(unittest.TestCase):
    def test_member_type_check_distinguishes_single_and_ensemble_specs(self):
        single = (
            ROOT
            / "config/aidc_load_15min_short/route_A/baseline/"
            / "lgbm_direct.yaml"
        )
        ensemble = (
            ROOT
            / "config/aidc_load_15min_short/route_A/add_ensemble/"
            / "ensemble_latin-a_averaging.yaml"
        )

        self.assertTrue(MODULE._is_single_model_config(single))
        self.assertFalse(MODULE._is_single_model_config(ensemble))


if __name__ == "__main__":
    unittest.main()
