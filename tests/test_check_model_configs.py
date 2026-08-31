# -*- coding: utf-8 -*-
"""模型 YAML 与独立工具 YAML 的结构识别测试。"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from config.config_loader import is_model_yaml, load_yaml_config
from forecasting_core.specs.config import parse_model_config


ROOT = Path(__file__).resolve().parents[1]


class ModelYamlDetectionTest(unittest.TestCase):
    def test_cli_counts_ensemble_semantic_problem_as_hard_failure(self):
        source = (
            ROOT
            / "config/aidc_load_15min_short/route_A/baseline/"
            / "lgbm_usbr_prob_mean_conformal.yaml"
        )
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        payload["ensemble"]["method"] = {
            "name": "stacking",
            "params": {"alpha": 1.0, "fit_intercept": True},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad-ensemble.yaml"
            path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/check_model_configs.py"),
                    str(path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("stacking 只支持 point 模式", result.stdout)

    def test_calendar_month_intraday_schedule_is_a_hard_failure(self):
        config_path = (
            ROOT
            / "config/aidc_power_month/route_A/freq_1day/add_decomposition/"
            / "lgbm_usmd_mean_prob_horizon_decomp_linear.yaml"
        )
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload["validation"]["schedule_mode"] = "intraday"

        with self.assertRaisesRegex(
            ValueError,
            "horizon_mode=calendar_month requires schedule_mode=daily",
        ):
            parse_model_config(payload, source=config_path)

    def test_legacy_override_schema_is_no_longer_a_model_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_model = root / "base.yaml"
            override_model = root / "override.yaml"
            tool_config = root / "periodicity.yaml"
            base_model.write_text("base_config: config.univariate_config\n", encoding="utf-8")
            override_model.write_text(
                "overrides:\n  runtime:\n    history_length: 30\n",
                encoding="utf-8",
            )
            tool_config.write_text(
                "source_path: dataset/x.csv\nseasonal_period: 96\n",
                encoding="utf-8",
            )

            self.assertFalse(is_model_yaml(base_model))
            self.assertFalse(is_model_yaml(override_model))
            self.assertFalse(is_model_yaml(tool_config))

    def test_any_model_schema_mapping_is_routed_to_model_parser(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing_groups = root / "missing-groups.yaml"
            unknown_groups = root / "unknown-groups.yaml"
            missing_groups.write_text("schema_version: 2\n", encoding="utf-8")
            unknown_groups.write_text(
                "schema_version: 2\nsource_path: dataset/x.csv\n",
                encoding="utf-8",
            )

            self.assertTrue(is_model_yaml(missing_groups))
            self.assertTrue(is_model_yaml(unknown_groups))

    def test_model_schema_mapping_without_schema_version_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing-version.yaml"
            path.write_text(
                "problem: {}\ndata: {}\nfeatures: {}\nstrategy: {}\nestimator: {}\n",
                encoding="utf-8",
            )

            self.assertTrue(is_model_yaml(path))
            with self.assertRaisesRegex(ValueError, "Missing YAML schema_version"):
                load_yaml_config(path)

    def test_unknown_schema_with_model_groups_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            integer_version = root / "future-model.yaml"
            string_version = root / "string-version-model.yaml"
            integer_version.write_text(
                "schema_version: 3\nproblem: {}\n",
                encoding="utf-8",
            )
            string_version.write_text(
                "schema_version: future\nproblem: {}\n",
                encoding="utf-8",
            )

            self.assertTrue(is_model_yaml(integer_version))
            self.assertTrue(is_model_yaml(string_version))


if __name__ == "__main__":
    unittest.main()
