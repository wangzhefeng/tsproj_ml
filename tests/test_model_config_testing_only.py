# -*- coding: utf-8 -*-
"""跨场景模型配置统一为仅测试模式。"""
import unittest
from pathlib import Path

from config.config_loader import load_yaml_config


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = (
    "aidc_ess_selfuse_load",
    "aidc_load_15min_daily",
    "aidc_load_15min_rolling",
    "aidc_load_15min_short",
    "aidc_power_month",
)


class ModelConfigTestingOnlyContractTest(unittest.TestCase):
    def test_all_model_yamls_are_testing_only(self):
        for scenario in SCENARIOS:
            paths = sorted((ROOT / "config" / scenario).glob("route_*/**/*.yaml"))
            self.assertTrue(paths, scenario)
            for path in paths:
                cfg = load_yaml_config(str(path))
                self.assertTrue(cfg.is_testing, path)
                self.assertFalse(cfg.is_forecasting, path)


if __name__ == "__main__":
    unittest.main()
