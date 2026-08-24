# -*- coding: utf-8 -*-
"""模型 YAML 与独立工具 YAML 的结构识别测试。"""

import tempfile
import unittest
from pathlib import Path

from scripts.check_model_configs import is_model_yaml


class ModelYamlDetectionTest(unittest.TestCase):
    def test_only_base_config_or_overrides_schema_is_a_model_yaml(self):
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

            self.assertTrue(is_model_yaml(base_model))
            self.assertTrue(is_model_yaml(override_model))
            self.assertFalse(is_model_yaml(tool_config))


if __name__ == "__main__":
    unittest.main()
