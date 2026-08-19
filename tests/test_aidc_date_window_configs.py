# -*- coding: utf-8 -*-

import copy
import unittest
from pathlib import Path

import yaml

from config.config_loader import load_yaml_config


ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = ROOT / "config/aidc_electricity_computility/electricity"
DATASET_ROOT = ROOT / "dataset/aidc_electricity_computility/electricity"
SCENES = [
    "A1_01a",
    "A1_201",
    "A1_IT",
    "A3_01e",
    "AIDC/route_A",
    "AIDC/route_B",
]
WINDOWS = {
    "2026-07-12": ("2026-06-11 00:00:00", "2026-07-12 23:55:00"),
    "2026-08-14": ("2026-07-14 00:00:00", "2026-08-14 23:55:00"),
}


class AidcDateWindowProcessConfigTest(unittest.TestCase):
    def test_all_twelve_process_configs_define_expected_contract(self):
        loaded_configs = {}
        for date, (start, end) in WINDOWS.items():
            for scene in SCENES:
                config_path = CONFIG_ROOT / date / scene / "data_process.yaml"
                self.assertTrue(config_path.exists(), config_path)
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                loaded_configs[(date, scene)] = config

                self.assertEqual(config["scene"], scene)
                self.assertEqual(config["start_time"], start)
                self.assertEqual(config["end_time"], end)
                self.assertEqual(config["freq"], "5min")
                self.assertEqual(config["time_col"], "time")
                self.assertEqual(config["value_col"], "value")
                self.assertEqual(config["input_glob"], "*.csv")
                self.assertEqual(config["max_fill_gap_slots"], 288)
                self.assertEqual(config["output_file"], "df_power.csv")
                self.assertEqual(config["audit_file"], "df_power_audit.json")
                self.assertEqual(
                    Path(config["dataset_dir"]),
                    Path("dataset/aidc_electricity_computility/electricity") / date / scene,
                )
                self.assertTrue((ROOT / config["dataset_dir"]).is_dir())
                for required_pattern in ["date_*.csv", "weather_*.csv", "df_power.csv"]:
                    self.assertIn(required_pattern, config["exclude_globs"])
                self.assertEqual(config["outlier"]["spike_half_window"], 2)
                self.assertEqual(config["outlier"]["spike_z_threshold"], 8.0)

        disabled = [
            key for key, config in loaded_configs.items() if not bool(config["enabled"])
        ]
        self.assertEqual(disabled, [("2026-07-12", "A3_01e")])
        reason = loaded_configs[("2026-07-12", "A3_01e")]["disabled_reason"]
        self.assertIn("7072", str(reason).replace(",", ""))

    def test_blocked_scene_has_data_insufficiency_note(self):
        note_path = CONFIG_ROOT / "2026-07-12/A3_01e/DATA_INSUFFICIENT.md"
        self.assertTrue(note_path.exists(), note_path)
        text = note_path.read_text(encoding="utf-8")
        self.assertIn("2026-06-18 10:40:00", text)
        self.assertIn("2026-07-12 23:55:00", text)
        self.assertIn("7,072", text)
        self.assertIn("不生成", text)


class AidcDateWindowModelConfigTest(unittest.TestCase):
    METHODS = {
        "lgbm_usmdp.yaml": "univariate-single-multistep-direct-pointwise",
        "lgbm_usmd.yaml": "univariate-single-multistep-direct",
        "lgbm_usmr.yaml": "univariate-single-multistep-recursive",
        "lgbm_usmdr.yaml": "univariate-single-multistep-direct-recursive",
    }
    DATETIME_FEATURES = [
        "hour",
        "minute",
        "day",
        "day_of_week",
        "week_of_year",
        "month",
        "days_in_month",
        "quarter",
        "day_of_year",
        "year",
    ]
    WEATHER_FEATURES = ["rt_ssr", "rt_ws10", "rt_tt2", "cal_rh", "rt_ps", "rt_rain"]

    @staticmethod
    def _config_path(date: str, scene: str, filename: str) -> Path:
        path = CONFIG_ROOT / date / scene / filename
        if date == "2026-07-12" and scene == "A3_01e":
            path = path.with_name(f"{filename}.disabled")
        return path

    @staticmethod
    def _normalize(loaded: dict) -> dict:
        normalized = copy.deepcopy(loaded)
        overrides = normalized["overrides"]
        overrides["runtime"]["now_time"] = "<NOW_TIME>"
        overrides["target_series"]["data_dir"] = "<DATA_DIR>"
        overrides["output"]["scenario_subpath"] = "<SCENARIO>"
        return normalized

    def test_model_config_counts_and_runtime_contract(self):
        active = sorted(
            path
            for date in WINDOWS
            for path in (CONFIG_ROOT / date).glob("**/lgbm_*.yaml")
        )
        disabled = sorted((CONFIG_ROOT / "2026-07-12").glob("**/lgbm_*.yaml.disabled"))
        self.assertEqual(len(active), 44)
        self.assertEqual(len(disabled), 4)

        for date, scenes in {
            "2026-07-12": [scene for scene in SCENES if scene != "A3_01e"],
            "2026-08-14": SCENES,
        }.items():
            for scene in scenes:
                for filename, method in self.METHODS.items():
                    config_path = self._config_path(date, scene, filename)
                    self.assertTrue(config_path.exists(), config_path)
                    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                    self.assertEqual(raw["base_config"], "config.univariate_config")
                    output = raw["overrides"]["output"]
                    self.assertEqual(
                        output["scenario_subpath"],
                        f"aidc_electricity_computility/electricity/{date}/{scene}",
                    )
                    self.assertNotIn("checkpoints_dir", output)
                    self.assertNotIn("test_results_dir", output)
                    self.assertNotIn("pred_results_dir", output)

                    cfg = load_yaml_config(config_path)
                    self.assertEqual(cfg.model_type, "lightgbm")
                    self.assertEqual(cfg.pred_method, method)
                    self.assertTrue(cfg.is_testing)
                    self.assertFalse(cfg.is_forecasting)
                    self.assertEqual(cfg.history_length, 32)
                    self.assertEqual(cfg.data_path, "df_power.csv")
                    self.assertEqual(cfg.target_ts_feat, "time")
                    self.assertEqual(cfg.target, "value")
                    self.assertTrue(Path(cfg.data_dir).joinpath(cfg.data_path).exists())
                    self.assertEqual(cfg.date_history_path, cfg.date_future_path)
                    self.assertEqual(cfg.weather_history_path, cfg.weather_future_path)
                    self.assertTrue(Path(cfg.data_dir).joinpath(cfg.date_history_path).exists())
                    self.assertTrue(Path(cfg.data_dir).joinpath(cfg.weather_history_path).exists())

    def test_all_active_and_disabled_configs_explicitly_define_exogenous_features(self):
        config_paths = sorted(
            path
            for date in WINDOWS
            for path in (CONFIG_ROOT / date).glob("**/lgbm_*.yaml*")
        )
        self.assertEqual(len(config_paths), 48)

        for config_path in config_paths:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(raw["overrides"]["runtime"]["window_length"], 15, config_path)
            exogenous = raw["overrides"]["exogenous_features"]
            self.assertIs(exogenous["enable_date_features"], True, config_path)
            self.assertEqual(exogenous["datetype_features"], ["date_type"], config_path)
            self.assertEqual(
                exogenous["datetype_categorical_features"], ["date_type"], config_path
            )
            self.assertIs(exogenous["enable_weather_features"], True, config_path)
            self.assertEqual(exogenous["weather_features"], self.WEATHER_FEATURES, config_path)
            self.assertEqual(exogenous["weather_categorical_features"], [], config_path)
            self.assertIs(exogenous["enable_datetime_features"], True, config_path)
            self.assertEqual(exogenous["datetime_features"], self.DATETIME_FEATURES, config_path)
            self.assertEqual(exogenous["datetime_categorical_features"], [], config_path)
            self.assertNotIn("weekday", exogenous["datetime_features"], config_path)
            self.assertNotIn("week", exogenous["datetime_features"], config_path)

            cfg = load_yaml_config(config_path)
            self.assertTrue(cfg.enable_date_features, config_path)
            self.assertTrue(cfg.enable_weather_features, config_path)
            self.assertTrue(cfg.enable_datetime_features, config_path)
            self.assertEqual(cfg.datetype_features, ["date_type"], config_path)
            self.assertEqual(cfg.datetype_categorical_features, ["date_type"], config_path)
            self.assertEqual(cfg.weather_features, self.WEATHER_FEATURES, config_path)
            self.assertEqual(cfg.weather_categorical_features, [], config_path)
            self.assertEqual(cfg.datetime_features, self.DATETIME_FEATURES, config_path)
            self.assertEqual(cfg.datetime_categorical_features, [], config_path)

    def test_july_and_august_configs_are_semantically_equal_after_date_normalization(self):
        for scene in SCENES:
            for filename in self.METHODS:
                july_path = self._config_path("2026-07-12", scene, filename)
                august_path = self._config_path("2026-08-14", scene, filename)
                july = yaml.safe_load(july_path.read_text(encoding="utf-8"))
                august = yaml.safe_load(august_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    self._normalize(july),
                    self._normalize(august),
                    f"{scene}/{filename}",
                )


if __name__ == "__main__":
    unittest.main()
