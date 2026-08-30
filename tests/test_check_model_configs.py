# -*- coding: utf-8 -*-
"""模型 YAML 与独立工具 YAML 的结构识别测试。"""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from config.config_loader import load_yaml_config
from scripts.check_model_configs import _resolve_runtime_shape, is_model_yaml


class ModelYamlDetectionTest(unittest.TestCase):
    def test_calendar_month_intraday_schedule_is_a_hard_failure(self):
        cfg = SimpleNamespace(
            horizon_mode="calendar_month",
            freq="1D",
            schedule_mode="intraday",
            train_window_length=60,
            now_time=datetime(2026, 8, 31, 12, 0),
            history_length=180,
        )

        with self.assertRaisesRegex(
            ValueError,
            "horizon_mode=calendar_month requires schedule_mode=daily.",
        ):
            _resolve_runtime_shape(cfg)

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
