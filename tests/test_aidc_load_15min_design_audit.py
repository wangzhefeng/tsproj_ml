# -*- coding: utf-8 -*-
"""AIDC 15min 活动单模型真实训练设计编译门禁。"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_aidc_load_15min_designs.py"


class AidcLoad15minDesignAuditTest(unittest.TestCase):
    def test_all_active_single_models_compile_one_real_training_design(self):
        self.assertTrue(SCRIPT.exists(), SCRIPT)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--json"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["scenario_count"], 3)
        self.assertEqual(payload["single_model_count"], 4617)
        self.assertEqual(payload["compiled_count"], 4617)
        self.assertEqual(payload["failure_count"], 0)
        self.assertEqual(payload["failures"], [])


if __name__ == "__main__":
    unittest.main()
