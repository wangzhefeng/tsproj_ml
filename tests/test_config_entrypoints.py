# -*- coding: utf-8 -*-

import subprocess
import sys
import tempfile
import unittest
import datetime
import importlib.util
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from main import Model
from config.config_loader import load_model_config, load_yaml_config
from data_provider.data_loader import DataLoader
from features.FeatureEngineering import FeatureEngineer


ROOT = Path(__file__).resolve().parent.parent
CHECKER_PATH = ROOT / "scripts" / "check_model_configs.py"


def load_config_checker():
    spec = importlib.util.spec_from_file_location("check_model_configs", CHECKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config checker: {CHECKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConfigEntrypointTest(unittest.TestCase):
    def test_model_config_exposes_flat_target_decomposition_fields(self):
        cfg = load_model_config(instantiate=True)

        self.assertEqual(cfg.decomposition_method, "none")
        self.assertFalse(hasattr(cfg, "detrend_target"))
        self.assertEqual(cfg.decomposition_periods, [])
        self.assertEqual(cfg.decomposition_trend_degree, 1)
        self.assertEqual(cfg.decomposition_trend_forecast, "polynomial")
        self.assertAlmostEqual(cfg.decomposition_damping, 0.98)
        self.assertEqual(cfg.decomposition_seasonal_cycles, 4)

    def test_config_checker_rejects_invalid_mstl_periods(self):
        checker = load_config_checker()
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "invalid_mstl.yaml"
            config_path.write_text(
                """base_config: config.univariate_config
overrides:
  runtime:
    history_length: 60
    predict_steps: 96
    window_length: 30
  target_series:
    freq: 15min
  preprocessing:
    decomposition_method: mstl
    decomposition_periods: [96]
""",
                encoding="utf-8",
            )
            _, problems = checker.check_model_yaml(str(config_path))

        self.assertTrue(any("mstl 至少需要两个周期" in problem for problem in problems))

    def test_config_checker_allows_exactly_two_decomposition_cycles(self):
        checker = load_config_checker()
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "stl_two_cycles.yaml"
            config_path.write_text(
                """base_config: config.univariate_config
overrides:
  runtime:
    history_length: 12
    predict_steps: 2
    window_length: 10
  target_series:
    freq: 1D
  preprocessing:
    decomposition_method: stl
    decomposition_periods: [4]
  model_strategy:
    pred_method: univariate-single-multistep-direct-pointwise
""",
                encoding="utf-8",
            )
            _, problems = checker.check_model_yaml(str(config_path))

        self.assertFalse(any("不足两个完整周期" in problem for problem in problems))

    def test_usmdp_rolllag_config_enables_safe_lags_without_target_leakage(self):
        checker = load_config_checker()
        rolllag_path = (
            ROOT
            / "config/aidc_ess_selfuse_load/route_A/tuning/"
            "lgbm_usmdp_prob_mean_rolllag.yaml"
        )
        _, rolllag_problems = checker.check_model_yaml(str(rolllag_path))
        self.assertFalse(any(problem.startswith("提示：") for problem in rolllag_problems))
        self.assertFalse(any("不能依赖目标列 y" in problem for problem in rolllag_problems))

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "usmdp_target_advanced.yaml"
            config_path.write_text(
                """base_config: config.univariate_config
overrides:
  runtime:
    history_length: 60
    predict_steps: 288
    window_length: 30
  advanced_features:
    enable_advanced_features: true
    enable_rolling_features: true
    rolling_columns: [y]
    rolling_windows: [288]
    rolling_stats: [mean]
  model_strategy:
    pred_method: univariate-single-multistep-direct-pointwise
""",
                encoding="utf-8",
            )
            _, target_problems = checker.check_model_yaml(str(config_path))
        self.assertTrue(any("不能依赖目标列 y" in problem for problem in target_problems))

    def test_config_checker_allows_advanced_context_beyond_max_lag_when_history_covers_it(self):
        checker = load_config_checker()
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "usmd_long_context.yaml"
            config_path.write_text(
                """base_config: config.univariate_config
overrides:
  runtime:
    history_length: 60
    predict_steps: 288
    window_length: 30
  time_lag_features:
    lags: [288, 2016]
  advanced_features:
    enable_advanced_features: true
    enable_rolling_features: true
    rolling_columns: [y]
    rolling_windows: [2016, 4032, 8064]
    rolling_stats: [mean]
    enable_diff_features: true
    diff_columns: [y]
    diff_periods: [288, 2016, 8064]
  model_strategy:
    pred_method: univariate-single-multistep-direct
""",
                encoding="utf-8",
            )
            _, problems = checker.check_model_yaml(str(config_path))

        self.assertFalse(any(problem.startswith("advanced 窗口/周期") for problem in problems))

    def test_aidc_multivariate_yaml_configs_use_selected_dataset_contract(self):
        root = ROOT / "config/aidc_electricity_computility/electricity/2026-06-11"
        expected_paths = [
            root / "A1_01a" / "lgbm_msmd.yaml",
            root / "A1_01a" / "lgbm_msmr.yaml",
            root / "A1_01a" / "lgbm_msmdr.yaml",
            root / "A1_201" / "lgbm_msmd.yaml",
            root / "A1_201" / "lgbm_msmr.yaml",
            root / "A1_201" / "lgbm_msmdr.yaml",
            root / "A1_IT" / "lgbm_msmd.yaml",
            root / "A1_IT" / "lgbm_msmr.yaml",
            root / "A1_IT" / "lgbm_msmdr.yaml",
            root / "A3_01e" / "lgbm_msmd.yaml",
            root / "A3_01e" / "lgbm_msmr.yaml",
            root / "A3_01e" / "lgbm_msmdr.yaml",
        ]
        expected_methods = {
            "lgbm_msmd.yaml": "multivariate-single-multistep-direct",
            "lgbm_msmr.yaml": "multivariate-single-multistep-recursive",
            "lgbm_msmdr.yaml": "multivariate-single-multistep-direct-recursive",
        }

        for config_path in expected_paths:
            self.assertTrue(config_path.exists(), config_path)
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["base_config"], "config.multivariate_config", config_path.name)

            cfg = load_yaml_config(config_path)
            self.assertEqual(cfg.data_path, "df_selected.csv", config_path.name)
            self.assertEqual(cfg.target_ts_feat, "count_data_time", config_path.name)
            self.assertEqual(cfg.target, "h_total_use", config_path.name)
            self.assertEqual(cfg.model_type, "lightgbm", config_path.name)
            self.assertEqual(cfg.pred_method, expected_methods[config_path.name], config_path.name)
            self.assertFalse(cfg.enable_feature_selection, config_path.name)
            self.assertTrue(cfg.target_series_numeric_features, config_path.name)
            self.assertNotIn("count_data_time", cfg.target_series_numeric_features, config_path.name)
            self.assertNotIn("h_total_use", cfg.target_series_numeric_features, config_path.name)

    def _run_python(self, code):
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_importing_main_does_not_parse_run_cli_arguments(self):
        code = (
            "import sys; "
            "sys.argv=['run.py','--config-yaml','config/ETT-small/ETTm1/lgbm_usmd.yaml',"
            "'--config-class','ModelConfig','--model-type','lightgbm']; "
            "import main; "
            "print('imported')"
        )

        result = self._run_python(code)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("imported", result.stdout)

    def test_importing_config_loader_does_not_parse_process_arguments(self):
        code = (
            "import sys; "
            "sys.argv=['tool.py','--unexpected-flag']; "
            "import config.config_loader as loader; "
            "print(hasattr(loader, 'load_model_config'))"
        )

        result = self._run_python(code)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("True", result.stdout)

    def test_load_yaml_config_applies_grouped_overrides(self):
        config_path = ROOT / "config/ETT-small/ETTm1/lgbm_msmd.yaml"
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertNotIn("config_class", loaded)
        self.assertEqual(loaded["base_config"], "config.multivariate_config")
        self.assertIn("model_strategy", loaded["overrides"])

        cfg = load_yaml_config(config_path)

        self.assertEqual(cfg.data_path, "ETTm1.csv")
        self.assertEqual(cfg.model_type, "lightgbm")
        self.assertEqual(cfg.pred_method, "multivariate-single-multistep-direct")
        self.assertEqual(cfg.now_time, datetime.datetime(2018, 6, 26, 19, 45, 0))
        self.assertEqual(cfg.date_history_path, "ETTm1_exogenous/df_date.csv")
        self.assertEqual(cfg.weather_history_path, "ETTm1_exogenous/df_weather.csv")
        self.assertEqual(
            cfg.target_series_numeric_features,
            ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
        )
        self.assertFalse(hasattr(cfg, "node_id"))
        self.assertFalse(hasattr(cfg, "out_system_id"))
        self.assertFalse(hasattr(cfg, "model_cfgs"))

    def test_ettm1_yaml_configs_pick_base_config_by_method(self):
        for config_path in sorted((ROOT / "config/ETT-small/ETTm1").glob("*.yaml")):
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertIn("model_strategy", loaded["overrides"], config_path.name)
            expected_base = (
                "config.multivariate_config"
                if "_ms" in config_path.stem
                else "config.univariate_config"
            )
            self.assertEqual(loaded["base_config"], expected_base, config_path.name)

    def test_load_yaml_config_keeps_flat_override_compatibility(self):
        content = (
            "base_config: config.univariate_config\n"
            "overrides:\n"
            "  data_path: ETTm1.csv\n"
            "  now_time: '2018-06-26T19:45:00'\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()

            cfg = load_yaml_config(handle.name)

        self.assertEqual(cfg.data_path, "ETTm1.csv")
        self.assertEqual(cfg.now_time, datetime.datetime(2018, 6, 26, 19, 45, 0))

    def test_load_yaml_config_ignores_deprecated_time_window_overrides(self):
        content = (
            "base_config: config.univariate_config\n"
            "overrides:\n"
            "  start_time: '2018-06-01T00:00:00'\n"
            "  future_time: '2018-06-02T00:00:00'\n"
            "  data_path: ETTm1.csv\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()

            cfg = load_yaml_config(handle.name)

        self.assertEqual(cfg.data_path, "ETTm1.csv")
        self.assertFalse(hasattr(cfg, "start_time"))
        self.assertFalse(hasattr(cfg, "future_time"))

    def test_load_yaml_config_rejects_unknown_grouped_override(self):
        content = (
            "base_config: config.univariate_config\n"
            "overrides:\n"
            "  data:\n"
            "    typo_field: 1\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()

            with self.assertRaisesRegex(AttributeError, "typo_field"):
                load_yaml_config(handle.name)

    def test_run_cli_overrides_yaml_values(self):
        import run

        old_argv = sys.argv[:]
        try:
            sys.argv = [
                "run.py",
                "--config-yaml",
                "config/ETT-small/ETTm1/lgbm_usmd.yaml",
                "--model-type",
                "xgboost",
                "--is-testing",
                "1",
            ]
            args = run.args_parse()
        finally:
            sys.argv = old_argv

        cfg = run._load_config(args.config_yaml)
        cfg = run._apply_overrides(cfg, args)

        self.assertEqual(cfg.model_type, "xgboost")
        self.assertTrue(cfg.is_testing)
        self.assertEqual(cfg.data_path, "ETTm1.csv")

    def test_standard_configs_keep_flat_dataclass_interface(self):
        from config.config_sections import BaseModelConfig, PRED_METHOD_CODE

        expected_fields = {
            "data_path",
            "target",
            "pred_method",
            "lags",
            "model_type",
            "date_history_path",
            "weather_history_path",
            "pred_results_dir",
            "enable_train_outlier_handling",
        }

        for module_name in ["config.univariate_config", "config.multivariate_config"]:
            cfg = load_model_config(module_name, "ModelConfig", instantiate=True)
            field_names = {field.name for field in fields(cfg)}

            self.assertIsInstance(cfg, BaseModelConfig)
            self.assertTrue(is_dataclass(cfg))
            self.assertTrue(expected_fields.issubset(field_names))
            self.assertIsInstance(cfg.pred_method, str)
            self.assertIn(cfg.pred_method, PRED_METHOD_CODE)
            self.assertTrue(hasattr(cfg, "pred_results_dir"))

    def test_generate_configs_builds_grouped_yaml_overrides(self):
        result = self._run_python(
            "import config.generate_configs as g; "
            "params={"
            "'history_length':31,'predict_steps':96,'window_length':15,"
            "'now_time_iso':'2018-06-26T19:45:00',"
            "'data_dir':'./dataset/example/','data_path':'df_power.csv','freq':'5min',"
            "'target_ts_feat':'count_data_time','target':'h_total_use',"
            "'target_series_numeric_features':[],'target_series_categorical_features':[],"
            "'target_series_drop_features':[],"
            "'enable_date_features':False,'enable_weather_features':False,"
            "'enable_datetime_features':True,'enable_lags_features':True,"
            "'model_type':'lightgbm','pred_method':'univariate-single-multistep-direct',"
            "'base_config':'config.univariate_config'"
            "}; "
            "cfg=g.build_yaml_config(params); "
            "print(cfg['base_config']); "
            "print(sorted(cfg['overrides'].keys())); "
            "print(cfg['overrides']['model_strategy']['pred_method']); "
            "print(cfg['overrides']['time_lag_features']['lags'])"
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("config.univariate_config", result.stdout)
        self.assertIn("model_strategy", result.stdout)
        self.assertIn("univariate-single-multistep-direct", result.stdout)
        # 滞后步数由 config_sections.default_lags_for_freq 按频率生成（5min → 288 起步）
        self.assertIn("[288, 576", result.stdout)

    def test_model_lag_validation_message_reports_strict_minimum_window_days(self):
        cfg = load_yaml_config(
            ROOT / "config/aidc_electricity_computility/electricity/2026-06-11/A1_01a/lgbm_usmr.yaml"
        )
        # 当前配置 max(lags)=864、predict_steps=288（5min）：window_length=4 时
        # 滑窗训练行数 = 4*288-288 = 864 <= 864 触发校验，最小 window_length = (864+288)//288+1 = 5
        cfg.window_length = 4

        with self.assertRaisesRegex(ValueError, r"need window_length >= 5\."):
            Model(cfg)


class ConfigRuntimeSemanticsTest(unittest.TestCase):
    def _feature_args(self, **overrides):
        args = SimpleNamespace(
            freq="5min",
            pred_method="univariate-single-multistep-direct",
            enable_date_features=False,
            date_ts_feat=None,
            datetype_features=[],
            datetype_categorical_features=[],
            enable_weather_features=False,
            weather_ts_feat=None,
            weather_features=[],
            weather_categorical_features=[],
            enable_datetime_features=True,
            datetime_features=["hour", "weekday"],
            datetime_categorical_features=[],
            enable_lags_features=True,
            lags=[1, 2],
            enable_advanced_features=False,
            use_horizon_exogenous_for_direct=False,
            enable_global_training=False,
            series_id_feature="series_id",
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_datetime_switch_disables_datetime_features(self):
        args = self._feature_args(enable_datetime_features=False)
        df = pd.DataFrame(
            {
                "time": pd.date_range("2026-06-01 00:00:00", periods=4, freq="5min"),
                "y": [1.0, 2.0, 3.0, 4.0],
            }
        )
        engineer = FeatureEngineer(args, log_prefix="[test]", verbose=False)

        featured, exogenous_features, categorical_features = engineer.create_exogenouse_features(
            df=df,
            df_date_history=None,
            df_date_future=None,
            df_weather_history=None,
            df_weather_future=None,
        )

        self.assertEqual(exogenous_features, [])
        self.assertEqual(categorical_features, [])
        self.assertNotIn("dt_hour", featured.columns)
        self.assertNotIn("dt_weekday", featured.columns)

    def test_lags_switch_disables_lag_features(self):
        args = self._feature_args(enable_lags_features=False)
        df = pd.DataFrame(
            {
                "time": pd.date_range("2026-06-01 00:00:00", periods=4, freq="5min"),
                "y": [1.0, 2.0, 3.0, 4.0],
            }
        )
        engineer = FeatureEngineer(args, log_prefix="[test]", verbose=False)

        featured, endogenous_features, target_output_features = engineer.create_endogenous_basic_features(
            df_series=df,
            endogenous_features_with_target=["y"],
            target_feature="y",
            horizon=2,
        )

        self.assertEqual(endogenous_features, [])
        self.assertNotIn("y_lag_1", featured.columns)
        self.assertNotIn("y_lag_2", featured.columns)
        self.assertEqual(target_output_features, ["y_shift_1", "y_shift_2"])

    def test_target_series_numeric_features_is_explicit_whitelist_when_set(self):
        args = SimpleNamespace(
            freq="5min",
            pred_method="multivariate-single-multistep-direct",
            target_ts_feat="time",
            target="target",
            target_series_numeric_features=["feat_keep"],
            target_series_categorical_features=[],
            target_series_drop_features=[],
            enable_global_training=False,
            series_id_feature="series_id",
            date_ts_feat=None,
            weather_ts_feat=None,
        )
        loader = DataLoader(
            args=args,
            train_start_time=datetime.datetime(2026, 6, 1, 0, 0),
            train_end_time=datetime.datetime(2026, 6, 1, 0, 15),
            forecast_start_time=datetime.datetime(2026, 6, 1, 0, 15),
            forecast_end_time=datetime.datetime(2026, 6, 1, 0, 30),
            log_prefix="[test]",
        )
        df_series = pd.DataFrame(
            {
                "time": pd.date_range("2026-06-01 00:00:00", periods=3, freq="5min"),
                "target": [10.0, 11.0, 12.0],
                "feat_keep": [1.0, 2.0, 3.0],
                "feat_extra": [4.0, 5.0, 6.0],
            }
        )

        (
            df_history,
            _df_date_history,
            _df_weather_history,
            endogenous_features_with_target,
            target_feature,
            _df_custom_history,
        ) = loader.process_history_data(
            {
                "target_series": df_series,
                "date_history": None,
                "weather_history": None,
                "custom_history": None,
            }
        )

        self.assertEqual(target_feature, "y")
        self.assertEqual(endogenous_features_with_target, ["feat_keep", "y"])
        self.assertIn("feat_keep", df_history.columns)
        self.assertNotIn("feat_extra", df_history.columns)

    def test_load_data_merges_deduplicates_and_slices_exogenous_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            pd.DataFrame(
                {
                    "time": pd.date_range("2026-06-01 00:00:00", periods=3, freq="5min"),
                    "target": [10.0, 11.0, 12.0],
                }
            ).to_csv(data_dir / "target.csv", index=False)
            pd.DataFrame(
                {
                    "date": ["2026-06-01", "2026-06-02", "2026-06-03"],
                    "date_type": [1, 1, 1],
                }
            ).to_csv(data_dir / "df_date.csv", index=False)
            pd.DataFrame(
                {
                    "date": ["2026-06-03", "2026-06-04", "2026-06-05"],
                    "date_type": [9, 2, 2],
                }
            ).to_csv(data_dir / "df_date_future.csv", index=False)
            pd.DataFrame(
                {
                    "ts": [
                        "2026-06-01 00:00:00",
                        "2026-06-02 00:00:00",
                        "2026-06-03 00:00:00",
                    ],
                    "rt_ssr": [10.0, 20.0, 30.0],
                }
            ).to_csv(data_dir / "df_weather.csv", index=False)
            pd.DataFrame(
                {
                    "ts": [
                        "2026-06-03 00:00:00",
                        "2026-06-04 00:00:00",
                        "2026-06-05 00:00:00",
                    ],
                    "rt_ssr": [99.0, 40.0, 50.0],
                }
            ).to_csv(data_dir / "df_weather_future.csv", index=False)

            args = SimpleNamespace(
                data_dir=data_dir,
                data_path="target.csv",
                date_history_path="df_date.csv",
                date_future_path="df_date_future.csv",
                date_ts_feat="date",
                weather_history_path="df_weather.csv",
                weather_future_path="df_weather_future.csv",
                weather_ts_feat="ts",
            )
            loader = DataLoader(
                args=args,
                train_start_time=datetime.datetime(2026, 6, 1, 0, 0),
                train_end_time=datetime.datetime(2026, 6, 3, 0, 0),
                forecast_start_time=datetime.datetime(2026, 6, 3, 0, 0),
                forecast_end_time=datetime.datetime(2026, 6, 5, 0, 0),
                log_prefix="[test]",
            )

            input_data = loader.load_data()

        self.assertNotEqual(
            input_data["date_history"]["date"].tolist(),
            input_data["date_future"]["date"].tolist(),
        )
        self.assertEqual(
            input_data["date_history"]["date"].tolist(),
            [
                pd.Timestamp("2026-06-01"),
                pd.Timestamp("2026-06-02"),
                pd.Timestamp("2026-06-03"),
            ],
        )
        self.assertEqual(
            input_data["date_future"]["date"].tolist(),
            [
                pd.Timestamp("2026-06-03"),
                pd.Timestamp("2026-06-04"),
                pd.Timestamp("2026-06-05"),
            ],
        )
        self.assertEqual(
            input_data["weather_history"]["ts"].tolist(),
            [
                pd.Timestamp("2026-06-01 00:00:00"),
                pd.Timestamp("2026-06-02 00:00:00"),
                pd.Timestamp("2026-06-03 00:00:00"),
            ],
        )
        self.assertEqual(
            input_data["weather_future"]["ts"].tolist(),
            [
                pd.Timestamp("2026-06-03 00:00:00"),
                pd.Timestamp("2026-06-04 00:00:00"),
                pd.Timestamp("2026-06-05 00:00:00"),
            ],
        )
        self.assertEqual(
            input_data["weather_future"].loc[
                input_data["weather_future"]["ts"] == pd.Timestamp("2026-06-03 00:00:00"),
                "rt_ssr",
            ].iloc[0],
            99.0,
        )


if __name__ == "__main__":
    unittest.main()
