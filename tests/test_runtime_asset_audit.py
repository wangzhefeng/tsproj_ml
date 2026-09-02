# -*- coding: utf-8 -*-
"""现役模型配置运行资产审计门禁。"""

import json
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_runtime_assets.py"
SPEC = importlib.util.spec_from_file_location("runtime_asset_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RuntimeAssetAuditTest(unittest.TestCase):
    def test_audit_reports_current_missing_asset_inventory(self):
        self.assertTrue(SCRIPT.exists(), SCRIPT)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(ROOT / "config"), "--json"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["model_config_count"], 707)
        self.assertEqual(payload["missing_unique_path_count"], 0)
        self.assertEqual(payload["affected_config_count"], 0)
        self.assertEqual(payload["missing_reference_count"], 0)
        self.assertEqual(payload["missing_paths"], [])
        self.assertEqual(payload["missing_declared_column_reference_count"], 0)
        self.assertEqual(payload["column_affected_config_count"], 0)
        self.assertEqual(payload["missing_declared_columns"], [])

    def test_header_audit_requires_non_ignored_declared_columns(self):
        with self.subTest("ignored metadata remains optional"):
            source = {
                "time_col": "time",
                "series_id_cols": [],
                "columns": [
                    {"name": "value", "role": "target"},
                    {"name": "needed", "role": "observed_past"},
                    {"name": "optional_meta", "role": "ignored"},
                ],
            }
            actual = {"time", "value", "optional_physical_column"}
            self.assertEqual(
                MODULE._missing_required_columns(source, actual),
                ["needed"],
            )


if __name__ == "__main__":
    unittest.main()
