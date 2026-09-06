# -*- coding: utf-8 -*-

import subprocess
import sys
import tempfile
import unittest
import datetime
import importlib.util
import pickle
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from model_ensemble.contracts import EnsembleRuntimeServices
from model_ensemble.loader import load_ensemble_config
from model_ensemble.runtime import run_ensemble_config
from model_training.estimators.capabilities import EstimatorCapabilities
from model_pipeline.runner import (
    CanonicalBaseModelRunner,
    persist_model_bundle,
    run_canonical_config,
)
from model_performance.resource_planner import plan_ensemble_resources
from forecasting_core.specs import (
    ColumnSpec,
    DataSourceSpec,
    DataSpec,
    EstimatorSpec,
    FeatureSpec,
    ForecastConfigSpec,
    ForecastProblemSpec,
    ForecastStrategySpec,
)
from run import CanonicalModel
from config.config_loader import load_yaml_config
from models.pickle_io import ModelDeployPkl
from model_forecasting.predictor import CanonicalForecaster
from model_training.trainer import CanonicalTrainer


ROOT = Path(__file__).resolve().parent.parent
CHECKER_PATH = ROOT / "scripts" / "check_model_configs.py"
ENSEMBLE_RUNTIME_SERVICES = EnsembleRuntimeServices(
    runner_factory=CanonicalBaseModelRunner,
    persist_bundle=persist_model_bundle,
    plan_resources=plan_ensemble_resources,
)


def load_config_checker():
    spec = importlib.util.spec_from_file_location("check_model_configs", CHECKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config checker: {CHECKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConfigEntrypointTest(unittest.TestCase):
    @staticmethod
    def _source(cfg, name):
        return next(source for source in cfg.data.sources if source.name == name)

    def test_config_checker_rejects_invalid_mstl_periods(self):
        checker = load_config_checker()
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "invalid_mstl.yaml"
            config_path.write_text(
                """schema_version: 2
problem:
  time_col: time
  freq: 15min
  horizon: 96
  targets: [load]
  training_scope: local
  series_id_cols: []
data:
  sources:
    - name: target_history
      source_type: file
      columns:
        - {name: load, role: target, categorical: false}
      history_path: dataset/load.csv
      time_col: time
      series_id_cols: []
      availability: source_time
features:
  target_lags: {load: [96]}
  observed_past_lags: {}
  datetime_features: []
  transformations:
    target:
      calendar_normalization: {method: none}
      decomposition:
        method: mstl
        periods: [96]
      scaling: {method: none, inverse: false}
strategy: {name: direct}
estimator:
  model_type: lightgbm
  target_adapter: independent
  params: {}
probabilistic: {mode: point}
validation:
  history_steps: 60
  train_window_steps: 30
  fold_count: 1
  stride_steps: 1
output: {}
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
                """schema_version: 2
problem:
  time_col: time
  freq: 1D
  horizon: 2
  targets: [load]
  training_scope: local
  series_id_cols: []
data:
  sources:
    - name: target_history
      source_type: file
      columns:
        - {name: load, role: target, categorical: false}
      history_path: dataset/load.csv
      time_col: time
      series_id_cols: []
      availability: source_time
features:
  target_lags: {load: [2]}
  observed_past_lags: {}
  datetime_features: []
  transformations:
    direct:
      layout: independent_models
    target:
      calendar_normalization: {method: none}
      decomposition: {method: stl, periods: [2]}
      scaling: {method: none, inverse: false}
strategy: {name: direct}
estimator:
  model_type: lightgbm
  target_adapter: independent
  params: {}
probabilistic: {mode: point}
validation:
  history_steps: 12
  train_window_steps: 10
  fold_count: 1
  stride_steps: 1
output: {}
""",
                encoding="utf-8",
            )
            _, problems = checker.check_model_yaml(str(config_path))

        self.assertFalse(any("不足两个完整周期" in problem for problem in problems))

    def test_usmdp_rolllag_config_enables_safe_lags_without_target_leakage(self):
        checker = load_config_checker()
        with tempfile.TemporaryDirectory() as tmp_dir:
            rolllag_path = Path(tmp_dir) / "usmdp_safe_rolllag.yaml"
            rolllag_path.write_text(
                """schema_version: 2
problem:
  time_col: time
  freq: 5min
  horizon: 288
  targets: [load]
  training_scope: local
  series_id_cols: []
data:
  sources:
    - name: target_history
      source_type: file
      columns:
        - {name: load, role: target, categorical: false}
      history_path: dataset/load.csv
      time_col: time
      series_id_cols: []
      availability: source_time
features:
  target_lags: {load: [288, 576]}
  observed_past_lags: {}
  datetime_features: []
  transformations:
    direct:
      layout: independent_models
strategy: {name: direct}
estimator:
  model_type: lightgbm
  target_adapter: independent
  params: {}
probabilistic: {mode: point}
validation:
  history_steps: 60
  train_window_steps: 30
  fold_count: 1
  stride_steps: 1
output: {}
""",
                encoding="utf-8",
            )
            _, rolllag_problems = checker.check_model_yaml(str(rolllag_path))
        self.assertFalse(any(problem.startswith("提示：") for problem in rolllag_problems))
        self.assertFalse(any("不能依赖目标列 y" in problem for problem in rolllag_problems))

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "usmdp_target_advanced.yaml"
            config_path.write_text(
                """schema_version: 2
problem:
  time_col: time
  freq: 5min
  horizon: 288
  targets: [load]
  training_scope: local
  series_id_cols: []
data:
  sources:
    - name: target_history
      source_type: file
      columns:
        - {name: load, role: target, categorical: false}
      history_path: dataset/load.csv
      time_col: time
      series_id_cols: []
      availability: source_time
features:
  target_lags: {load: [288]}
  observed_past_lags: {}
  datetime_features: []
  transformations:
    direct:
      layout: independent_models
    advanced:
      rolling:
        columns: [load]
        windows: [288]
        stats: [mean]
strategy: {name: direct}
estimator:
  model_type: lightgbm
  target_adapter: independent
  params: {}
probabilistic: {mode: point}
validation:
  history_steps: 60
  train_window_steps: 30
  fold_count: 1
  stride_steps: 1
output: {}
""",
                encoding="utf-8",
            )
            _, target_problems = checker.check_model_yaml(str(config_path))
        self.assertFalse(
            any("不能依赖目标列 y" in problem for problem in target_problems)
        )

    def test_config_checker_allows_advanced_context_beyond_max_lag_when_history_covers_it(self):
        checker = load_config_checker()
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "usmd_long_context.yaml"
            config_path.write_text(
                """schema_version: 2
problem:
  time_col: time
  freq: 5min
  horizon: 288
  targets: [load]
  training_scope: local
  series_id_cols: []
data:
  sources:
    - name: target_history
      source_type: file
      columns:
        - {name: load, role: target, categorical: false}
      history_path: dataset/load.csv
      time_col: time
      series_id_cols: []
      availability: source_time
features:
  target_lags: {load: [288, 2016]}
  observed_past_lags: {}
  datetime_features: []
  transformations:
    direct:
      layout: independent_models
    advanced:
      rolling:
        columns: [load]
        windows: [2016, 4032, 8064]
        stats: [mean]
      difference:
        columns: [load]
        periods: [288, 2016, 8064]
strategy: {name: direct}
estimator:
  model_type: lightgbm
  target_adapter: independent
  params: {}
probabilistic: {mode: point}
validation:
  history_steps: 60
  train_window_steps: 30
  fold_count: 1
  stride_steps: 1
output: {}
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
        for config_path in expected_paths:
            self.assertTrue(config_path.exists(), config_path)
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["schema_version"], 2, config_path.name)

            cfg = load_yaml_config(config_path)
            target_source = self._source(cfg, "target_history")
            observed_columns = [
                column.name
                for column in target_source.columns
                if column.role.value == "observed_past"
            ]
            self.assertEqual(Path(target_source.history_path).name, "df_selected.csv", config_path.name)
            self.assertEqual(cfg.problem.time_col, "count_data_time", config_path.name)
            self.assertEqual(cfg.problem.targets, ("h_total_use",), config_path.name)
            self.assertEqual(cfg.estimator.model_type, "lightgbm", config_path.name)
            self.assertTrue(observed_columns, config_path.name)
            self.assertNotIn("count_data_time", observed_columns, config_path.name)
            self.assertNotIn("h_total_use", observed_columns, config_path.name)

    def _run_python(self, code):
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_importing_run_does_not_parse_cli_arguments(self):
        code = (
            "import sys; "
            "sys.argv=['run.py','--config-yaml','config/aidc_load_month/route_B/lgbm_usmd_prob_mean.yaml',"
            "'--config-class','ModelConfig','--model-type','lightgbm']; "
            "import run; "
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
        self.assertIn("False", result.stdout)

    def test_load_yaml_config_loads_canonical_groups(self):
        config_path = (
            ROOT
            / "config/aidc_load_15min_daily/route_A/add_exogenous/lgbm_direct_holiday-weather.yaml"
        )
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["schema_version"], 2)
        self.assertEqual(
            set(loaded),
            {
                "schema_version",
                "problem",
                "data",
                "features",
                "strategy",
                "estimator",
                "probabilistic",
                "validation",
                "output",
            },
        )

        cfg = load_yaml_config(config_path)
        self.assertIsInstance(cfg, ForecastConfigSpec)
        assert isinstance(cfg, ForecastConfigSpec)
        self.assertEqual(
            tuple(source.name for source in cfg.data.sources),
            ("target_history", "chinese_holiday", "weather"),
        )
        target_source = self._source(cfg, "target_history")
        weather_source = self._source(cfg, "weather")
        self.assertEqual(
            Path(target_source.history_path).name,
            "A_Loads_15min_mean_20251001_20260731.csv",
        )
        self.assertEqual(cfg.estimator.model_type, "lightgbm")
        self.assertEqual(cfg.validation["forecast_origin"], "2026-07-31T23:45:00")
        self.assertEqual(Path(weather_source.history_path).name, "weather_15min_20250101_20260731.csv")
        self.assertEqual(
            [column.name for column in target_source.columns if column.role.value == "observed_past"],
            [],
        )

class Task27ExecutionMatrixTest(unittest.TestCase):
    STRATEGIES = (
        ("recursive", None),
        ("direct", None),
        ("mimo", None),
        ("recmo", 2),
        ("dirrec", None),
        ("dirmo", 2),
        ("dirrecmo", 2),
    )
    EXPECTED_MATRIX_CASES = 46

    @staticmethod
    def _write_local_data(path):
        times = pd.date_range("2026-01-01", periods=72, freq="1h")
        step = np.arange(len(times), dtype=float)
        pd.DataFrame(
            {
                "time": times,
                "load": 100.0 + step * 0.5 + np.sin(step / 4.0),
                "power": 20.0 + step * 0.2 + np.cos(step / 5.0),
            }
        ).to_csv(path, index=False)

    @staticmethod
    def _write_global_data(path):
        times = pd.date_range("2026-01-01", periods=72, freq="1h")
        rows = []
        for series_index, series_id in enumerate(("A", "B")):
            step = np.arange(len(times), dtype=float)
            rows.extend(
                {
                    "series_id": series_id,
                    "time": timestamp,
                    "load": 100.0 + series_index * 50.0 + step_value * 0.5,
                    "power": 20.0 + series_index * 10.0 + step_value * 0.2,
                }
                for timestamp, step_value in zip(times, step)
            )
        pd.DataFrame(rows).to_csv(path, index=False)

    @staticmethod
    def _target_lags(strategy):
        if strategy in {"recursive", "recmo", "dirrec", "dirrecmo"}:
            return (1, 2, 3)
        return (4, 5, 6)

    def _config(
        self,
        data_path,
        *,
        strategy,
        chunk_length,
        targets,
        training_scope,
        adapter,
        mode,
    ):
        is_global = training_scope == "global"
        target_lags = self._target_lags(strategy)
        if adapter == "native":
            model_type = "rf"
            params = {"n_estimators": 4, "max_depth": 3, "random_state": 0}
        elif mode == "quantile":
            model_type = "qr"
            params = {"alpha": 0.0, "solver": "highs"}
        else:
            model_type = "ridge"
            params = {"alpha": 1e-6}
        columns = [
            ColumnSpec("series_id", "key", categorical=True)
        ] if is_global else []
        columns.extend(ColumnSpec(target, "target") for target in targets)
        return ForecastConfigSpec(
            problem=ForecastProblemSpec(
                time_col="time",
                freq="1h",
                horizon=4,
                targets=targets,
                training_scope=training_scope,
                series_id_cols=("series_id",) if is_global else (),
            ),
            data=DataSpec(
                (
                    DataSourceSpec(
                        name="target_history",
                        source_type="file",
                        columns=tuple(columns),
                        history_path=str(data_path),
                        time_col="time",
                        series_id_cols=("series_id",) if is_global else (),
                        availability="source_time",
                    ),
                )
            ),
            features=FeatureSpec(
                target_lags={target: target_lags for target in targets},
                observed_past_lags={},
                datetime_features=("hour",),
                transformations=(
                    {
                        "feature_scaling": {
                            "method": "none",
                            "grouped": False,
                            "encode_categorical": True,
                        }
                    }
                    if is_global
                    else {}
                ),
            ),
            strategy=ForecastStrategySpec(
                strategy,
                output_chunk_length=chunk_length,
            ),
            estimator=EstimatorSpec(
                model_type=model_type,
                target_adapter=adapter,
                params=params,
            ),
            probabilistic=(
                {"mode": "point"}
                if mode == "point"
                else {
                    "mode": "quantile",
                    "quantiles": [0.1, 0.5, 0.9],
                    "point_quantile": 0.5,
                }
            ),
            validation={
                "forecast_origin": "2026-01-03T23:00:00",
                "history_steps": 10_000,
                "train_window_steps": 9_999,
                "fold_count": 1,
                "stride_steps": 4,
                "training_scope": {
                    "series_order": ["A", "B"] if is_global else [],
                    "incomplete_series_policy": "raise",
                    "unknown_series_policy": "raise",
                },
                **(
                    {"performance": {"model_thread_count": 1}}
                    if adapter == "native"
                    else {}
                ),
            },
            output={"scenario_subpath": "task27"},
        )

    def _ensemble_doc_for_task27(self, data_path, method):
        """Build a reference-based ensemble YAML doc for the TASK27 matrix."""
        import copy

        config = self._config(
            data_path,
            strategy="direct",
            chunk_length=None,
            targets=("load",),
            training_scope="local",
            adapter="independent",
            mode="point",
        )
        payload = config.canonical_payload()
        doc = {
            "schema_version": 2,
            "problem": payload["problem"],
            "data": payload["data"],
            "probabilistic": payload["probabilistic"],
            "ensemble": {
                "members": [
                    {"name": "m_direct", "config_ref": "member_direct.yaml"},
                    {"name": "m_recursive", "config_ref": "member_recursive.yaml"},
                ],
                "oof": {"train_window_steps": 6, "fold_count": 2, "stride_steps": 1},
                "method": {"name": method},
            },
            "validation": payload["validation"],
            "output": payload["output"],
        }
        member_doc = copy.deepcopy(payload)
        member_doc["strategy"] = {"name": "direct"}
        (Path(data_path).parent / "member_direct.yaml").write_text(
            yaml.safe_dump(member_doc), encoding="utf-8"
        )
        member_doc = copy.deepcopy(payload)
        member_doc["strategy"] = {"name": "recursive"}
        (Path(data_path).parent / "member_recursive.yaml").write_text(
            yaml.safe_dump(member_doc), encoding="utf-8"
        )
        return doc

    def _assert_runtime_case(self, config, output_root):
        result = run_canonical_config(config, output_root=output_root)
        prediction = pd.read_csv(result.forecast_dir / "prediction.csv")
        expected_rows = (
            (2 if config.problem.training_scope == "global" else 1)
            * config.problem.horizon
            * len(config.problem.targets)
        )
        self.assertEqual(len(prediction), expected_rows)
        self.assertEqual(result.bundle.schema_version, 2)
        if config.probabilistic.get("mode") == "quantile":
            self.assertEqual(
                [column for column in prediction if column.startswith("predict_q")],
                ["predict_q10", "predict_q50", "predict_q90"],
            )

    def _assert_adapter_case(self, config, data_path):
        frame = pd.read_csv(data_path)
        targets = list(config.problem.targets)
        values = frame[targets].to_numpy(dtype=float)
        horizon = config.problem.horizon
        n_rows = len(values) - horizon
        base_design = np.column_stack(
            (
                np.arange(n_rows, dtype=float),
                values[:n_rows],
            )
        )
        target_values = np.stack(
            [values[step : step + n_rows] for step in range(1, horizon + 1)],
            axis=1,
        )
        resolved = config.strategy.resolve(horizon)
        designs = tuple(
            np.column_stack(
                (base_design, np.full(n_rows, float(call_index)))
            )
            for call_index in range(resolved.n_calls)
        )
        capabilities = EstimatorCapabilities(
            scalar_target=True,
            scalar_quantile=False,
            native_multi_target_point=config.estimator.target_adapter == "native",
            native_multi_target_quantile=False,
            sample_weight=True,
            categorical=False,
            nan_support=False,
        )
        feature_schema = tuple(
            ["row_index"]
            + [f"current_{target}" for target in targets]
            + ["call_index"]
        )
        artifact = CanonicalTrainer(
            config,
            estimator_factory=(
                lambda: RandomForestRegressor(
                    n_estimators=4, max_depth=3, random_state=0, n_jobs=1,
                )
            ) if config.estimator.target_adapter == "native" else (
                lambda: Ridge(alpha=1e-6)
            ),
            capabilities=capabilities,
            feature_schema=feature_schema,
        ).train(designs, target_values, n_series=1)
        predict_base = np.column_stack(
            (
                [float(n_rows)],
                values[n_rows : n_rows + 1],
            )
        )
        predict_designs = tuple(
            np.column_stack(
                (predict_base, np.full(1, float(call_index)))
            )
            for call_index in range(resolved.n_calls)
        )
        prediction = CanonicalForecaster(config, artifact).predict(
            predict_designs[0],
            series_ids=("__local__",),
            forecast_times=pd.date_range("2026-01-04", periods=horizon, freq="1h"),
            feature_provider=lambda call_index, *_: predict_designs[call_index],
        )

        self.assertEqual(prediction.shape, (1, horizon, len(targets)))
        self.assertEqual(artifact.estimator_coupling, config.estimator.target_adapter)
        self.assertTrue(np.isfinite(prediction.values).all())

    def test_task27_full_execution_matrix_reports_exact_counts(self):
        case_count = 0
        passed_count = 0
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            local_path = root / "local.csv"
            local_k1_path = root / "local_k1.csv"
            global_path = root / "global.csv"
            self._write_local_data(local_path)
            pd.read_csv(local_path)[["time", "load"]].to_csv(local_k1_path, index=False)
            self._write_global_data(global_path)

            for strategy, chunk_length in self.STRATEGIES:
                case_count += 1
                with self.subTest(group="local-k1-point", strategy=strategy):
                    config = self._config(
                        local_k1_path,
                        strategy=strategy,
                        chunk_length=chunk_length,
                        targets=("load",),
                        training_scope="local",
                        adapter="independent",
                        mode="point",
                    )
                    self._assert_runtime_case(
                        config,
                        root / "runs" / f"local-k1-{strategy}",
                    )
                    passed_count += 1

            for strategy, chunk_length in self.STRATEGIES:
                for adapter in ("independent", "regressor_chain", "native"):
                    case_count += 1
                    with self.subTest(
                        group="local-k2-point",
                        strategy=strategy,
                        adapter=adapter,
                    ):
                        config = self._config(
                            local_path,
                            strategy=strategy,
                            chunk_length=chunk_length,
                            targets=("load", "power"),
                            training_scope="local",
                            adapter=adapter,
                            mode="point",
                        )
                        # 七策略的 K2 adapter 组合在训练/推理层验证；
                        # Direct 保留两个真实后端的完整 runtime 接线。
                        if adapter == "regressor_chain" or strategy != "direct":
                            self._assert_adapter_case(config, local_path)
                        else:
                            self._assert_runtime_case(
                                config,
                                root / "runs" / f"local-k2-{strategy}-{adapter}",
                            )
                        passed_count += 1

            for strategy, chunk_length in self.STRATEGIES:
                case_count += 1
                with self.subTest(group="local-k2-quantile", strategy=strategy):
                    config = self._config(
                        local_path,
                        strategy=strategy,
                        chunk_length=chunk_length,
                        targets=("load", "power"),
                        training_scope="local",
                        adapter="independent",
                        mode="quantile",
                    )
                    self._assert_runtime_case(
                        config,
                        root / "runs" / f"local-k2-quantile-{strategy}",
                    )
                    passed_count += 1

            for strategy, chunk_length in self.STRATEGIES:
                case_count += 1
                with self.subTest(group="global-n2-k2-point", strategy=strategy):
                    config = self._config(
                        global_path,
                        strategy=strategy,
                        chunk_length=chunk_length,
                        targets=("load", "power"),
                        training_scope="global",
                        adapter="independent",
                        mode="point",
                    )
                    self._assert_runtime_case(
                        config,
                        root / "runs" / f"global-{strategy}",
                    )
                    passed_count += 1

            for method in ("averaging", "linear_blending"):
                case_count += 1
                with self.subTest(group="ensemble", method=method):
                    doc = self._ensemble_doc_for_task27(
                        local_k1_path, method
                    )
                    ens_path = root / f"ens_{method}.yaml"
                    ens_path.write_text(
                        yaml.safe_dump(doc), encoding="utf-8"
                    )
                    config = load_ensemble_config(ens_path)
                    result = run_ensemble_config(
                        config,
                        output_root=root / "runs" / f"ensemble-{method}",
                        base_dir=root,
                        services=ENSEMBLE_RUNTIME_SERVICES,
                    )
                    combined = result["combined_values"]
                    self.assertTrue(np.isfinite(combined).all())
                    passed_count += 1

            case_count += 1
            with self.subTest(group="non-canonical", kind="config"):
                legacy_path = root / "legacy.yaml"
                legacy_path.write_text(
                    """base_config: config.univariate_config\noverrides:\n  runtime:\n    predict_steps: 4\n  model_strategy:\n    pred_method: univariate-single-multistep-direct\n""",
                    encoding="utf-8",
                )
                before = legacy_path.read_bytes()
                with self.assertRaises(ValueError):
                    load_yaml_config(legacy_path)
                self.assertEqual(legacy_path.read_bytes(), before)
                passed_count += 1

            case_count += 1
            with self.subTest(group="non-canonical", kind="artifact"):
                artifact_path = root / "legacy.pkl"
                legacy_artifact = {
                    "bundle_type": "blend_direct_recursive",
                    "direct": "direct-model",
                    "recursive": "recursive-model",
                }
                artifact_path.write_bytes(pickle.dumps(legacy_artifact, protocol=2))
                loaded = ModelDeployPkl(artifact_path).load_model()
                self.assertEqual(loaded, legacy_artifact)
                passed_count += 1

        print(
            f"TASK27_MATRIX cases={case_count} passed={passed_count}",
            flush=True,
        )
        self.assertEqual(case_count, self.EXPECTED_MATRIX_CASES)
        self.assertEqual(passed_count, self.EXPECTED_MATRIX_CASES)

    def test_task27_checker_validates_all_model_configs_read_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "check_model_configs.log"
            error_path = Path(temp_dir) / "check_model_configs.err"
            with log_path.open("w", encoding="utf-8") as stdout, error_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                result = subprocess.run(
                    [sys.executable, str(CHECKER_PATH)],
                    cwd=ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    check=False,
                    timeout=120,
                )
            output = log_path.read_text(encoding="utf-8")
            errors = error_path.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, errors or output[-4000:])
        import re as _re

        checked = _re.search(r"checked=(\d+) passed=\1 hard_failures=0", output)
        self.assertIsNotNone(checked, output[-4000:])
        assert checked is not None
        self.assertEqual(int(checked.group(1)), 5150)
        self.assertNotIn("硬校验失败", output)


if __name__ == "__main__":
    unittest.main()
