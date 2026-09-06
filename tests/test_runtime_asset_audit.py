# -*- coding: utf-8 -*-
"""现役模型配置运行资产审计门禁。"""

import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from forecasting_core.specs import ColumnSpec, DataSourceSpec


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
        self.assertEqual(payload["model_config_count"], 5150)
        self.assertEqual(payload["missing_unique_path_count"], 0)
        self.assertEqual(payload["affected_config_count"], 0)
        self.assertEqual(payload["missing_reference_count"], 0)
        self.assertEqual(payload["missing_paths"], [])
        self.assertEqual(payload["missing_declared_column_reference_count"], 0)
        self.assertEqual(payload["column_affected_config_count"], 0)
        self.assertEqual(payload["missing_declared_columns"], [])

    def test_header_audit_requires_non_ignored_declared_columns(self):
        with self.subTest("ignored metadata remains optional"):
            source = DataSourceSpec(
                name="target", source_type="file", history_path="source.csv",
                time_col="time", availability="source_time", provider="persistence",
                columns=(
                    ColumnSpec("value", "target"),
                    ColumnSpec("needed", "observed_past"),
                    ColumnSpec("optional_meta", "ignored"),
                ),
            )
            actual = {"time", "value", "optional_physical_column"}
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "model.yaml").write_text("schema_version: 2\n")
                (root / "source.csv").write_text(",".join(sorted(actual)) + "\n")

                with patch.object(MODULE, "is_model_yaml", return_value=True), \
                     patch.object(MODULE, "load_yaml_config", return_value=None), \
                     patch.object(MODULE, "_config_sources", return_value=(source,)):
                    report = MODULE.audit_runtime_assets(root, repository_root=root)
                self.assertEqual(report["missing_declared_column_reference_count"], 1)
                missing = report["missing_declared_columns"][0]
                self.assertEqual(missing["path"], "source.csv")
                self.assertEqual(missing["references"][0]["missing_columns"], ["needed"])


if __name__ == "__main__":
    unittest.main()
