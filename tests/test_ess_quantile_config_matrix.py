# -*- coding: utf-8 -*-
"""ESS 模型配置矩阵契约。"""
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
USMDP_SAFE_LAGS = [288, 576, 864, 1152, 1440, 1728, 2016]
STRATEGY_FILES = {
    "lgbm_usmd_prob_mean_conformal.yaml": "univariate-single-multistep-direct",
    "lgbm_usmd_mean_prob_horizon_conformal.yaml": "univariate-single-multistep-direct",
    "lgbm_usmdp_prob_mean_conformal.yaml": "univariate-single-multistep-direct-pointwise",
    "lgbm_usmdr_prob_mean_conformal.yaml": "univariate-single-multistep-direct-recursive",
    "lgbm_usmr_prob_mean_conformal.yaml": "univariate-single-multistep-recursive",
}


class EssQuantileConfigMatrixTest(unittest.TestCase):
    def test_all_model_configs_use_five_test_windows(self):
        config_paths = sorted(CONFIG_ROOT.glob("route_*/**/*.yaml"))
        self.assertEqual(len(config_paths), 82)

        for path in config_paths:
            cfg = load_yaml_config(str(path))
            self.assertEqual(cfg.freq, "5min", path)
            n_per_day = 288
            history_rows = int(cfg.history_length * n_per_day)
            window_rows = int(cfg.window_length * n_per_day)
            n_windows = (history_rows - window_rows) // int(cfg.predict_steps) + 1
            self.assertEqual(n_windows, 5, path)

    def test_matrix_contains_only_expected_quantile_configs(self):
        expected_counts = {
            "baseline": 5,
            "add_decomposition": 12,
            "add_exogenous_weather_date": 4,
            "add_exogenous_plan_strategy": 4,
            "add_exogenous_weather_date_plan_strategy": 4,
            "add_strategy_features": 5,
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

    def test_decomposition_matrix_is_linear_stl288_and_mstl(self):
        for route in ("A", "B"):
            folder = CONFIG_ROOT / f"route_{route}" / "add_decomposition"
            for method in METHODS:
                linear = load_yaml_config(str(folder / f"lgbm_{method}_prob_mean_decomp_linear.yaml"))
                stl = load_yaml_config(str(folder / f"lgbm_{method}_prob_mean_decomp_stl288.yaml"))
                mstl = load_yaml_config(str(folder / f"lgbm_{method}_prob_mean_decomp_mstl288-2016.yaml"))
                self.assertEqual(linear.decomposition_method, "linear")
                self.assertEqual(stl.decomposition_method, "stl")
                self.assertEqual(mstl.decomposition_method, "mstl")
                self.assertEqual(stl.decomposition_periods, [288])
                self.assertEqual(mstl.decomposition_periods, [288, 2016])
                self.assertFalse(linear.enable_datetime_features)
                self.assertFalse(stl.enable_datetime_features)
                self.assertFalse(mstl.enable_datetime_features)
                self.assertEqual(linear.scenario_subpath, f"aidc_ess_selfuse_load/route_{route}/add_decomposition")
                self.assertEqual(stl.scenario_subpath, f"aidc_ess_selfuse_load/route_{route}/add_decomposition")
                self.assertEqual(mstl.scenario_subpath, f"aidc_ess_selfuse_load/route_{route}/add_decomposition")

    def test_baseline_uses_only_target_derived_features(self):
        for route in ("A", "B"):
            folder = CONFIG_ROOT / f"route_{route}" / "baseline"
            for path in sorted(folder.glob("*.yaml")):
                cfg = load_yaml_config(str(path))
                self.assertFalse(cfg.enable_datetime_features, path)
                self.assertFalse(cfg.enable_date_features, path)
                self.assertFalse(cfg.enable_weather_features, path)
                self.assertEqual(cfg.custom_features, [], path)
                self.assertFalse(cfg.enable_ensemble, path)
                self.assertEqual(cfg.decomposition_method, "none", path)

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
                    self.assertTrue(cfg.enable_datetime_features)
                    self.assertEqual(cfg.datetype_features, ["date_type"])
                    self.assertEqual(cfg.datetype_categorical_features, ["date_type"])
                    self.assertTrue(cfg.enable_weather_features)
                    self.assertTrue(cfg.strict_weather_information_set)
                    self.assertEqual(cfg.weather_history_source, "actual")
                    self.assertTrue(cfg.strict_date_information_set)
                    self.assertEqual(cfg.weather_backtest_source, "forecast")
                    self.assertEqual(cfg.weather_future_source, "forecast")
                    self.assertEqual(cfg.weather_features, WEATHER_COLS)
                    self.assertFalse(cfg.enable_ensemble)
                    if group == "add_exogenous_weather_date":
                        self.assertEqual(cfg.custom_features, [])
                    self.assertEqual(cfg.decomposition_method, "none")
                    self.assertEqual(
                        cfg.scenario_subpath,
                        f"aidc_ess_selfuse_load/route_{route}/{group}",
                    )
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
                    if group == "add_exogenous_plan_strategy":
                        self.assertFalse(cfg.enable_datetime_features)
                    if method in {"usmd", "usmdr"}:
                        self.assertTrue(cfg.use_horizon_exogenous_for_direct)

    def test_endogenous_actual_strategy_is_point_only_without_decomposition(self):
        expected_files = {
            "lgbm_msbr_mean_pcs.yaml",
            "lgbm_msmd_mean_pcs.yaml",
            "lgbm_msmd_mean_pcs_horizon.yaml",
            "lgbm_msmdr_mean_pcs.yaml",
            "lgbm_msmdr_mean_pcs_aux.yaml",
            "lgbm_msmr_mean_pcs.yaml",
            "lgbm_msmr_mean_pcs_aux.yaml",
        }
        for route in ("A", "B"):
            folder = CONFIG_ROOT / f"route_{route}" / "add_endogenous_actual_strategy"
            paths = sorted(folder.glob("*.yaml"))
            self.assertEqual({path.name for path in paths}, expected_files)
            for path in paths:
                cfg = load_yaml_config(str(path))
                self.assertEqual(cfg.predict_type, "point", path)
                self.assertEqual(cfg.decomposition_method, "none", path)
                self.assertFalse(cfg.enable_ensemble, path)
                self.assertEqual(
                    cfg.scenario_subpath,
                    f"aidc_ess_selfuse_load/route_{route}/add_endogenous_actual_strategy",
                    path,
                )
                if cfg.endogenous_backfill_strategy == "auxiliary":
                    self.assertIn(cfg.pred_method, {
                        "multivariate-single-multistep-recursive",
                        "multivariate-single-multistep-direct-recursive",
                    })

    def test_strategy_features_use_c5_on_five_lightgbm_methods(self):
        for route in ("A", "B"):
            folder = CONFIG_ROOT / f"route_{route}" / "add_strategy_features"
            paths = sorted(folder.glob("*.yaml"))
            self.assertEqual({path.name for path in paths}, set(STRATEGY_FILES))
            settings = set()
            for path in paths:
                cfg = load_yaml_config(str(path))
                self.assertEqual(cfg.pred_method, STRATEGY_FILES[path.name], path)
                self.assertEqual(cfg.predict_type, "quantile", path)
                self.assertTrue(cfg.enable_conformal_calibration, path)
                self.assertEqual(cfg.decomposition_method, "none", path)
                self.assertFalse(cfg.enable_datetime_features, path)
                self.assertFalse(cfg.enable_date_features, path)
                self.assertFalse(cfg.enable_weather_features, path)
                self.assertFalse(cfg.enable_ensemble, path)
                self.assertTrue(cfg.is_testing, path)
                self.assertFalse(cfg.is_forecasting, path)
                self.assertEqual(len(cfg.custom_features), 1, path)
                source = cfg.custom_features[0]
                self.assertEqual(source.get("name"), "strategy_features_v2_c5_joint", path)
                self.assertEqual(len(source.get("columns", [])), 50, path)
                settings.add((cfg.pred_method, cfg.direct_strategy, cfg.setting_suffix))
                if path.name == "lgbm_usmd_mean_prob_horizon_conformal.yaml":
                    self.assertEqual(cfg.direct_strategy, "horizon_feature", path)
                    self.assertEqual(cfg.setting_suffix, "-horizon-conformal-strategy-c5", path)
                else:
                    self.assertEqual(cfg.setting_suffix, "-conformal-strategy-c5", path)
                if "usmdp" in path.name:
                    self.assertTrue(cfg.align_direct_features_to_target, path)
                    self.assertEqual(cfg.lags, USMDP_SAFE_LAGS, path)
            self.assertEqual(len(settings), 5)

    def test_usmdp_explicitly_enables_safe_lags(self):
        for route in ("A", "B"):
            for group, filename in (
                ("baseline", "lgbm_usmdp_prob_mean.yaml"),
                ("add_exogenous_weather_date", "lgbm_usmdp_prob_mean_weather_date.yaml"),
                ("add_exogenous_plan_strategy", "lgbm_usmdp_prob_mean_plan.yaml"),
                ("add_exogenous_weather_date_plan_strategy", "lgbm_usmdp_prob_mean_all.yaml"),
            ):
                cfg = load_yaml_config(str(CONFIG_ROOT / f"route_{route}" / group / filename))
                self.assertTrue(cfg.enable_lags_features)
                self.assertTrue(cfg.align_direct_features_to_target)
                self.assertEqual(cfg.lags, USMDP_SAFE_LAGS)
                self.assertGreaterEqual(min(cfg.lags), cfg.predict_steps)

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
