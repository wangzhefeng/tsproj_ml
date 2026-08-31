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
    "lgbm_usmd_prob_mean_conformal.yaml",
    "lgbm_usmd_mean_prob_horizon_conformal.yaml",
    "lgbm_usmdp_prob_mean_conformal.yaml",
    "lgbm_usmdr_prob_mean_conformal.yaml",
    "lgbm_usmr_prob_mean_conformal.yaml",
}


class EssQuantileConfigMatrixTest(unittest.TestCase):
    @staticmethod
    def _source(cfg, name):
        return next(source for source in cfg.data.sources if source.name == name)

    @staticmethod
    def _target_transform(cfg):
        return cfg.features.transformations.get("target", {})

    @classmethod
    def _decomposition_method(cls, cfg):
        return cls._target_transform(cfg).get("decomposition", {}).get(
            "method", "none"
        )

    def test_all_model_configs_use_nonoverlapping_backtest_contract(self):
        # 2026-08-30 折合同修复：history_steps 语义 = 折候选池大小，必须满足
        # E1 非重叠合同 history_steps > horizon（其余量级不限制窗口个数）。
        config_paths = sorted(
            path
            for path in CONFIG_ROOT.glob("route_*/**/*.yaml")
            if "ensemble_members" not in path.parts
        )
        self.assertEqual(len(config_paths), 82)

        from model_ensemble.specs import EnsembleConfigSpec

        for path in config_paths:
            cfg = load_yaml_config(str(path))
            if isinstance(cfg, EnsembleConfigSpec):
                continue
            self.assertEqual(cfg.problem.freq, "5min", path)
            n_per_day = 288
            history_rows = int(cfg.validation["history_steps"] * n_per_day)
            window_rows = int(cfg.validation["train_window_steps"] * n_per_day)
            self.assertGreater(history_rows, window_rows, path)
            self.assertGreater(
                cfg.validation["history_steps"],
                cfg.problem.horizon / n_per_day,
                path,
            )

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
                from model_ensemble.specs import EnsembleConfigSpec

                files = sorted((CONFIG_ROOT / f"route_{route}" / group).glob("*.yaml"))
                self.assertEqual(len(files), count, (route, group))
                for path in files:
                    cfg = load_yaml_config(str(path))
                    if isinstance(cfg, EnsembleConfigSpec):
                        continue
                    self.assertEqual(cfg.probabilistic["mode"], "quantile", path)
                    self.assertEqual(cfg.probabilistic["crossing_method"], "isotonic", path)
                    self.assertTrue(Path(self._source(cfg, "target_history").history_path).exists(), path)

    def test_decomposition_matrix_is_linear_stl288_and_mstl(self):
        for route in ("A", "B"):
            folder = CONFIG_ROOT / f"route_{route}" / "add_decomposition"
            for method in METHODS:
                linear = load_yaml_config(str(folder / f"lgbm_{method}_prob_mean_decomp_linear.yaml"))
                stl = load_yaml_config(str(folder / f"lgbm_{method}_prob_mean_decomp_stl288.yaml"))
                mstl = load_yaml_config(str(folder / f"lgbm_{method}_prob_mean_decomp_mstl288-2016.yaml"))
                self.assertEqual(self._decomposition_method(linear), "linear")
                self.assertEqual(self._decomposition_method(stl), "stl")
                self.assertEqual(self._decomposition_method(mstl), "mstl")
                self.assertEqual(stl.output["setting_suffix"], "-decomp-stl288")
                self.assertEqual(mstl.output["setting_suffix"], "-decomp-mstl288-2016")
                self.assertEqual(linear.features.datetime_features, ())
                self.assertEqual(stl.features.datetime_features, ())
                self.assertEqual(mstl.features.datetime_features, ())
                expected_subpath = f"aidc_ess_selfuse_load/route_{route}/add_decomposition"
                self.assertEqual(linear.output["scenario_subpath"], expected_subpath)
                self.assertEqual(stl.output["scenario_subpath"], expected_subpath)
                self.assertEqual(mstl.output["scenario_subpath"], expected_subpath)

    def test_baseline_uses_only_target_derived_features(self):
        for route in ("A", "B"):
            folder = CONFIG_ROOT / f"route_{route}" / "baseline"
            for path in sorted(folder.glob("*.yaml")):
                cfg = load_yaml_config(str(path))
                from model_ensemble.specs import EnsembleConfigSpec

                if isinstance(cfg, EnsembleConfigSpec):
                    # reference-based ensemble: check refs and move on
                    self.assertEqual(
                        tuple(member.name for member in cfg.members),
                        ("direct", "recursive"),
                        path,
                    )
                    continue
                self.assertEqual(cfg.features.datetime_features, (), path)
                self.assertEqual(tuple(source.name for source in cfg.data.sources), ("target_history",), path)
                self.assertEqual(self._decomposition_method(cfg), "none", path)

    def test_weather_date_uses_strict_native_weather_and_date_type(self):
        for route in ("A", "B"):
            for group, suffix in (
                ("add_exogenous_weather_date", "weather_date"),
                ("add_exogenous_weather_date_plan_strategy", "all"),
            ):
                folder = CONFIG_ROOT / f"route_{route}" / group
                for method in METHODS:
                    cfg = load_yaml_config(str(folder / f"lgbm_{method}_prob_mean_{suffix}.yaml"))
                    date_type = self._source(cfg, "date_type")
                    weather = self._source(cfg, "weather")
                    self.assertTrue(cfg.features.datetime_features)
                    self.assertEqual(tuple(column.name for column in date_type.columns), ("date_type",))
                    self.assertTrue(date_type.columns[0].categorical)
                    self.assertEqual(date_type.availability.value, "column")
                    self.assertEqual(date_type.available_at_col, "available_at")
                    self.assertEqual(weather.availability.value, "column")
                    self.assertEqual(weather.available_at_col, "available_at")
                    self.assertEqual([column.name for column in weather.columns], WEATHER_COLS)
                    if group == "add_exogenous_weather_date":
                        self.assertEqual(tuple(source.name for source in cfg.data.sources), ("target_history", "date_type", "weather"))
                    self.assertEqual(self._decomposition_method(cfg), "none")
                    self.assertEqual(
                        cfg.output["scenario_subpath"],
                        f"aidc_ess_selfuse_load/route_{route}/{group}",
                    )
                    self.assertTrue(Path(weather.history_path).exists())
                    self.assertTrue(Path(weather.backtest_path).exists())
                    self.assertTrue(Path(weather.future_path).exists())

    def test_plan_is_explicit_custom_future_for_all_methods(self):
        for route in ("A", "B"):
            for group, suffix in (
                ("add_exogenous_plan_strategy", "plan"),
                ("add_exogenous_weather_date_plan_strategy", "all"),
            ):
                folder = CONFIG_ROOT / f"route_{route}" / group
                for method in METHODS:
                    cfg = load_yaml_config(str(folder / f"lgbm_{method}_prob_mean_{suffix}.yaml"))
                    source = self._source(cfg, "pcs_plan")
                    self.assertEqual(source.availability.value, "column")
                    self.assertEqual(source.available_at_col, "available_at")
                    self.assertTrue(Path(source.history_path).exists())
                    self.assertTrue(Path(source.future_path).exists())
                    if group == "add_exogenous_plan_strategy":
                        self.assertEqual(cfg.features.datetime_features, ())

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
                from model_ensemble.specs import EnsembleConfigSpec

                if isinstance(cfg, EnsembleConfigSpec):
                    self.assertEqual(
                        tuple(member.name for member in cfg.members),
                        ("direct", "recursive"),
                        path,
                    )
                    continue
                self.assertEqual(cfg.probabilistic["mode"], "point", path)
                self.assertEqual(self._decomposition_method(cfg), "none", path)
                self.assertEqual(
                    cfg.output["scenario_subpath"],
                    f"aidc_ess_selfuse_load/route_{route}/add_endogenous_actual_strategy",
                    path,
                )
                if self._source(cfg, "target_history").provider == "auxiliary":
                    self.assertIsNotNone(cfg.strategy, path)
                    self.assertIn(cfg.strategy.name.value, {"recursive", "recmo"}, path)

    def test_strategy_features_use_c5_on_five_lightgbm_methods(self):
        for route in ("A", "B"):
            folder = CONFIG_ROOT / f"route_{route}" / "add_strategy_features"
            paths = sorted(folder.glob("*.yaml"))
            self.assertEqual({path.name for path in paths}, set(STRATEGY_FILES))
            settings = set()
            for path in paths:
                cfg = load_yaml_config(str(path))
                self.assertIsNotNone(cfg.strategy, path)
                self.assertEqual(cfg.probabilistic["mode"], "quantile", path)
                self.assertEqual(cfg.probabilistic["conformal"]["method"], "cqr", path)
                self.assertEqual(self._decomposition_method(cfg), "none", path)
                self.assertEqual(cfg.features.datetime_features, (), path)
                self.assertEqual(len(cfg.data.sources), 2, path)
                source = self._source(cfg, "strategy_features_v2_c5_joint")
                self.assertEqual(len(source.columns), 50, path)
                direct_layout = cfg.features.transformations.get("direct", {}).get(
                    "layout"
                )
                settings.add((path.name, direct_layout, cfg.output["setting_suffix"]))
                if path.name == "lgbm_usmd_mean_prob_horizon_conformal.yaml":
                    self.assertEqual(direct_layout, "single_model_horizon", path)
                    self.assertEqual(cfg.output["setting_suffix"], "-horizon-conformal-strategy-c5", path)
                else:
                    self.assertEqual(cfg.output["setting_suffix"], "-conformal-strategy-c5", path)
                if "usmdp" in path.name:
                    self.assertEqual(cfg.features.target_lags["value"], tuple(USMDP_SAFE_LAGS), path)
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
                lags = cfg.features.target_lags["value"]
                self.assertEqual(lags, tuple(USMDP_SAFE_LAGS))
                self.assertEqual(
                    cfg.features.transformations["direct"]["layout"],
                    "single_model_horizon",
                )
                self.assertGreaterEqual(min(lags), cfg.problem.horizon)

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
                self.assertEqual(cfg.strategy.output_chunk_length, 96)
                self.assertEqual(cfg.problem.horizon // cfg.strategy.output_chunk_length, 3)


if __name__ == "__main__":
    unittest.main()
