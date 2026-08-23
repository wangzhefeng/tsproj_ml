# -*- coding: utf-8 -*-
"""ESS quantile 配置矩阵契约。"""
import unittest
from pathlib import Path

from config.config_loader import load_yaml_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "config/aidc_ess_selfuse_load"
WEATHER_COLS = [
    "rt_ssr", "rt_tt2", "cal_rh", "rt_ws10",
    "tt2_mean_3h", "tt2_diff_1h", "ssr_mean_3h",
]
METHODS = ["usmd", "usmdp", "usmdr", "usmr"]


class EssQuantileConfigMatrixTest(unittest.TestCase):
    def test_matrix_contains_only_expected_quantile_configs(self):
        expected_counts = {
            "baseline": 5,
            "add_decomposition": 8,
            "add_exogenous_weather_date": 4,
            "add_exogenous_plan_strategy": 4,
            "add_exogenous_weather_date_plan_strategy": 4,
        }
        for route in ("A", "B"):
            for group, count in expected_counts.items():
                files = sorted((CONFIG_ROOT / f"route_{route}" / group).glob("*.yaml"))
                self.assertEqual(len(files), count, (route, group))
                for path in files:
                    cfg = load_yaml_config(str(path))
                    self.assertEqual(cfg.predict_type, "quantile", path)
                    self.assertTrue(cfg.quantile_monotone, path)
                    self.assertTrue((Path(cfg.data_dir) / cfg.data_path).exists(), path)

    def test_decomposition_matrix_is_linear_and_stl288(self):
        for route in ("A", "B"):
            folder = CONFIG_ROOT / f"route_{route}" / "add_decomposition"
            for method in METHODS:
                linear = load_yaml_config(str(folder / f"lgbm_{method}_prob_mean_decomp_linear.yaml"))
                stl = load_yaml_config(str(folder / f"lgbm_{method}_prob_mean_decomp_stl288.yaml"))
                self.assertEqual(linear.decomposition_method, "linear")
                self.assertEqual(stl.decomposition_method, "stl")
                self.assertEqual(stl.decomposition_periods, [288])
                self.assertEqual(linear.scenario_subpath, f"aidc_ess_selfuse_load/route_{route}/add_decomposition")
                self.assertEqual(stl.scenario_subpath, f"aidc_ess_selfuse_load/route_{route}/add_decomposition")

    def test_weather_date_uses_strict_native_weather_and_date_type(self):
        for route in ("A", "B"):
            for group, suffix in (
                ("add_exogenous_weather_date", "weather_date"),
                ("add_exogenous_weather_date_plan_strategy", "all"),
            ):
                folder = CONFIG_ROOT / f"route_{route}" / group
                for method in METHODS:
                    cfg = load_yaml_config(str(folder / f"lgbm_{method}_prob_mean_{suffix}.yaml"))
                    self.assertTrue(cfg.enable_date_features)
                    self.assertEqual(cfg.datetype_features, ["date_type"])
                    self.assertEqual(cfg.datetype_categorical_features, ["date_type"])
                    self.assertTrue(cfg.enable_weather_features)
                    self.assertTrue(cfg.strict_weather_information_set)
                    self.assertEqual(cfg.weather_history_source, "actual")
                    self.assertTrue(cfg.strict_date_information_set)
                    self.assertEqual(cfg.weather_backtest_source, "forecast")
                    self.assertEqual(cfg.weather_future_source, "forecast")
                    self.assertEqual(cfg.weather_features, WEATHER_COLS)
                    self.assertTrue((Path(cfg.data_dir) / cfg.weather_history_path).exists())
                    self.assertTrue((Path(cfg.data_dir) / cfg.weather_backtest_path).exists())
                    self.assertTrue((Path(cfg.data_dir) / cfg.weather_future_path).exists())
                    if method in {"usmd", "usmdr"}:
                        self.assertTrue(cfg.use_horizon_exogenous_for_direct)

    def test_plan_is_explicit_custom_future_for_all_methods(self):
        for route in ("A", "B"):
            for group, suffix in (
                ("add_exogenous_plan_strategy", "plan"),
                ("add_exogenous_weather_date_plan_strategy", "all"),
            ):
                folder = CONFIG_ROOT / f"route_{route}" / group
                for method in METHODS:
                    cfg = load_yaml_config(str(folder / f"lgbm_{method}_prob_mean_{suffix}.yaml"))
                    sources = [source for source in cfg.custom_features if source.get("name") == "pcs_plan"]
                    self.assertEqual(len(sources), 1)
                    source = sources[0]
                    self.assertEqual(source.get("future_strategy"), "explicit")
                    self.assertEqual(source.get("availability"), "forecast_origin")
                    self.assertTrue(source.get("strict_information_set"))
                    self.assertEqual(source.get("available_at_col"), "available_at")
                    self.assertTrue((Path(cfg.data_dir) / source["history_path"]).exists())
                    self.assertTrue((Path(cfg.data_dir) / source["future_path"]).exists())
                    if method in {"usmd", "usmdr"}:
                        self.assertTrue(cfg.use_horizon_exogenous_for_direct)

    def test_usmdp_explicitly_disables_noop_lags(self):
        for route in ("A", "B"):
            for group, filename in (
                ("baseline", "lgbm_usmdp_prob_mean.yaml"),
                ("add_exogenous_weather_date", "lgbm_usmdp_prob_mean_weather_date.yaml"),
                ("add_exogenous_plan_strategy", "lgbm_usmdp_prob_mean_plan.yaml"),
                ("add_exogenous_weather_date_plan_strategy", "lgbm_usmdp_prob_mean_all.yaml"),
            ):
                cfg = load_yaml_config(str(CONFIG_ROOT / f"route_{route}" / group / filename))
                self.assertFalse(cfg.enable_lags_features)
                self.assertEqual(cfg.lags, [])

    def test_usmdr_uses_three_real_blocks(self):
        for route in ("A", "B"):
            for group, filename in (
                ("baseline", "lgbm_usmdr_prob_mean.yaml"),
                ("add_decomposition", "lgbm_usmdr_prob_mean_decomp_linear.yaml"),
                ("add_decomposition", "lgbm_usmdr_prob_mean_decomp_stl288.yaml"),
                ("add_exogenous_weather_date", "lgbm_usmdr_prob_mean_weather_date.yaml"),
                ("add_exogenous_plan_strategy", "lgbm_usmdr_prob_mean_plan.yaml"),
                ("add_exogenous_weather_date_plan_strategy", "lgbm_usmdr_prob_mean_all.yaml"),
            ):
                cfg = load_yaml_config(str(CONFIG_ROOT / f"route_{route}" / group / filename))
                self.assertEqual(cfg.block_size, 96)
                self.assertEqual(cfg.predict_steps // cfg.block_size, 3)


if __name__ == "__main__":
    unittest.main()
