# -*- coding: utf-8 -*-
"""Canonical runtime real model_training/backtest/forecast smoke tests."""

import tempfile
import unittest
import pickle
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from data_loading import SourceRegistry
from model_forecasting.design import minimum_history_rows
from model_forecasting.runtime import _RegistryDesignBuilder, run_canonical_config
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
from model_training.strategies import (
    AdapterPredictor,
    CanonicalStrategyArtifact,
    StrategyModelGroupArtifact,
    StrategyTargetPlan,
)
from forecasting_core.tensors import PointForecastTensor
from model_forecasting.forecaster import CanonicalMarginalQuantileForecaster
from probabilistic.training import CanonicalMarginalQuantileArtifact
from forecasting_core.artifacts import ForecastModelBundle


class _LagOffsetAdapter:
    def __init__(self, lag_index, offset):
        self.lag_index = lag_index
        self.offset = offset

    def predict(self, X):
        values = np.asarray(X, dtype=float)[:, self.lag_index] + self.offset
        return values.reshape(-1, 1, 1)


class CanonicalRuntimeSmokeTest(unittest.TestCase):
    def build_config(
        self,
        data_path,
        *,
        mode,
        strategy="recursive",
        horizon=2,
        output_chunk_length=None,
        align_to_target=None,
    ):
        target_lags = (
            (1, 2, 3)
            if strategy in {"recursive", "recmo", "dirrec", "dirrecmo"}
            else tuple(range(horizon, horizon + 3))
        )
        return ForecastConfigSpec(
            problem=ForecastProblemSpec(
                time_col="time",
                freq="1h",
                horizon=horizon,
                targets=("load",),
                information_mode="forecast",
                training_scope="local",
                series_id_cols=(),
            ),
            data=DataSpec(
                (
                    DataSourceSpec(
                        name="target_history",
                        source_type="file",
                        columns=(ColumnSpec("load", "target"),),
                        history_path=str(data_path),
                        time_col="time",
                        availability="source_time",
                    ),
                )
            ),
            features=FeatureSpec(
                target_lags={"load": target_lags},
                observed_past_lags={},
                datetime_features=("hour",),
                transformations=(
                    {
                        "direct": {
                            "layout": "independent_models",
                            "align_to_target": align_to_target,
                        }
                    }
                    if align_to_target is not None
                    else {}
                ),
            ),
            strategy=ForecastStrategySpec(
                strategy,
                output_chunk_length=output_chunk_length,
            ),
            estimator=EstimatorSpec(
                model_type=("ridge" if mode == "point" else "qr"),
                target_adapter="independent",
                params=({"alpha": 1e-6} if mode == "point" else {}),
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
                "forecast_origin": "2026-01-02T23:00:00",
                "history_steps": 10_000,
                "train_window_steps": 9_999,
                "fold_count": 1,
                "stride_steps": horizon,
            },
            output={"scenario_subpath": "smoke", "setting_suffix": ""},
        )

    def test_origin_frozen_direct_lags_require_origin_plus_history_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "load.csv"
            pd.DataFrame(
                {
                    "time": pd.date_range("2026-01-01", periods=12, freq="1h"),
                    "load": np.arange(12, dtype=float),
                }
            ).to_csv(data_path, index=False)
            config = self.build_config(
                data_path,
                mode="point",
                strategy="direct",
                horizon=2,
                align_to_target=False,
            )
            builder = _RegistryDesignBuilder(config, SourceRegistry(config.data, root))

            self.assertEqual(minimum_history_rows(config), 5)

    def build_transformed_k2_config(self, data_path, *, mode):
        targets = ("load", "power")
        return ForecastConfigSpec(
            problem=ForecastProblemSpec(
                time_col="time",
                freq="1h",
                horizon=2,
                targets=targets,
                information_mode="forecast",
                training_scope="local",
                series_id_cols=(),
            ),
            data=DataSpec(
                (
                    DataSourceSpec(
                        name="target_history",
                        source_type="file",
                        columns=tuple(ColumnSpec(target, "target") for target in targets),
                        history_path=str(data_path),
                        time_col="time",
                        availability="source_time",
                    ),
                )
            ),
            features=FeatureSpec(
                target_lags={"load": (2, 3, 4), "power": (2, 3, 4)},
                observed_past_lags={},
                datetime_features=("hour",),
                transformations={
                    "feature_scaling": {
                        "method": "standard",
                        "grouped": False,
                        "encode_categorical": False,
                    },
                    "target": {
                        "calendar_normalization": {"method": "none"},
                        "decomposition": {"method": "linear"},
                        "scaling": {"method": "standard", "inverse": True},
                    },
                },
            ),
            strategy=ForecastStrategySpec("direct"),
            estimator=EstimatorSpec(
                model_type=("ridge" if mode == "point" else "qr"),
                target_adapter="independent",
                params=({"alpha": 1e-9} if mode == "point" else {}),
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
                "forecast_origin": "2026-01-04T11:00:00",
                "history_steps": 10_000,
                "train_window_steps": 9_999,
                "fold_count": 1,
                "stride_steps": 2,
            },
            output={"scenario_subpath": f"transformed-{mode}"},
        )

    def test_point_runtime_writes_real_long_results_and_bundle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            times = pd.date_range("2026-01-01", periods=48, freq="1h")
            values = 100.0 + np.arange(48, dtype=float) * 2.0
            data_path = root / "load.csv"
            pd.DataFrame({"time": times, "load": values}).to_csv(data_path, index=False)
            config = self.build_config(data_path, mode="point")

            result = run_canonical_config(config, output_root=root / "results")

            prediction = pd.read_csv(result.forecast_dir / "prediction.csv")
            scores = pd.read_csv(result.test_dir / "test_scores_df.csv")
            self.assertEqual(
                list(prediction.columns),
                ["series_id", "time", "target", "predict_value"],
            )
            self.assertEqual(len(prediction), 2)
            # K=1、H=2、1 窗：汇总 target + aggregate；horizon 明细独立文件
            self.assertEqual(scores["scope"].tolist(), ["target", "aggregate"])
            horizon_scores = pd.read_csv(
                result.test_dir / "test_scores_horizon_df.csv"
            )
            self.assertEqual(
                set(horizon_scores["scope"]),
                {"horizon", "aggregate_horizon"},
            )
            # 点模式不产出概率分数文件（2026-08-30 接线：仅 quantile 模式）
            self.assertFalse(
                (result.test_dir / "test_scores_probabilistic_df.csv").exists()
            )
            self.assertTrue((result.model_dir / "model.pkl").exists())
            self.assertTrue((result.model_dir / "resolved_model.json").exists())
            self.assertTrue((result.forecast_dir / "resolved_config.json").exists())
            self.assertIsInstance(result.bundle, ForecastModelBundle)
            self.assertEqual(result.bundle.schema_version, 2)
            with (result.model_dir / "model.pkl").open("rb") as handle:
                restored = pickle.load(handle)
            self.assertIsInstance(restored, ForecastModelBundle)
            self.assertEqual(restored.config_fingerprint, config.fingerprint())
            self.assertEqual(
                tuple(item["feature"] for item in restored.feature_lineage),
                restored.selected_features,
            )
            self.assertEqual(restored.source_lineage[0]["source"], "target_history")
            self.assertEqual(
                restored.input_schema["availability_summary"],
                {
                    "known_future": ["dt_hour"],
                    "source_time": ["load__lag_1", "load__lag_2", "load__lag_3"],
                },
            )
            resolved = json.loads(
                (result.forecast_dir / "resolved_config.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                resolved["runtime"]["capability_probe"]["resolved"],
                restored.estimator_spec["capabilities"],
            )

    def test_local_k2_rolling_backtest_writes_multiple_long_windows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            times = pd.date_range("2026-01-01", periods=96, freq="1h")
            data_path = root / "targets.csv"
            pd.DataFrame(
                {
                    "time": times,
                    "load": 100.0 + np.arange(len(times), dtype=float),
                    "power": 500.0 + 4.0 * np.arange(len(times), dtype=float),
                }
            ).to_csv(data_path, index=False)
            base = self.build_transformed_k2_config(data_path, mode="point")
            config = ForecastConfigSpec(
                problem=base.problem,
                data=base.data,
                features=base.features,
                strategy=base.strategy,
                estimator=base.estimator,
                probabilistic=base.probabilistic,
                validation={
                    "forecast_origin": "2026-01-04T23:00:00",
                    "history_steps": 48,
                    "train_window_steps": 24,
                    "fold_count": 3,
                    "stride_steps": 4,
                },
                output=base.output,
            )

            result = run_canonical_config(config, output_root=root / "results")

            cv = pd.read_csv(result.test_dir / "cv_plot_df.csv")
            scores = pd.read_csv(result.test_dir / "test_scores_df.csv")
            self.assertEqual(sorted(cv["window"].unique().tolist()), [1, 2, 3])
            self.assertEqual(set(cv["target"]), {"load", "power"})
            self.assertEqual(len(cv), 3 * 2 * 2)
            # K=2、H=2、3 窗：汇总每窗 2 target + 1 aggregate = 3
            self.assertEqual(len(scores), 3 * 3)
            self.assertEqual(set(scores["scope"]), {"target", "aggregate"})
            horizon_scores = pd.read_csv(
                result.test_dir / "test_scores_horizon_df.csv"
            )
            self.assertEqual(len(horizon_scores), 3 * 6)
            self.assertEqual(
                set(horizon_scores["scope"]),
                {"horizon", "aggregate_horizon"},
            )
            self.assertTrue((result.test_dir / "target_plots" / "load.png").exists())
            self.assertTrue((result.test_dir / "target_plots" / "power.png").exists())
            metadata = json.loads(
                (result.test_dir / "result_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["aggregate_weighting"], {"load": 0.5, "power": 0.5})
            for window in metadata["backtest"]["windows"]:
                self.assertLess(
                    pd.Timestamp(window["training_label_end_max"]),
                    pd.Timestamp(window["label_start"]),
                )
                self.assertLessEqual(window["training_sample_count"], 24)

    def test_quantile_runtime_writes_n_h_k_q_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            times = pd.date_range("2026-01-01", periods=48, freq="1h")
            values = 50.0 + np.sin(np.arange(48) / 4.0) * 5.0 + np.arange(48) * 0.2
            data_path = root / "load.csv"
            pd.DataFrame({"time": times, "load": values}).to_csv(data_path, index=False)
            config = self.build_config(data_path, mode="quantile", strategy="direct")

            result = run_canonical_config(config, output_root=root / "results")

            prediction = pd.read_csv(result.forecast_dir / "prediction.csv")
            self.assertEqual(
                [column for column in prediction if column.startswith("predict_q")],
                ["predict_q10", "predict_q50", "predict_q90"],
            )
            np.testing.assert_allclose(
                prediction["predict_value"],
                prediction["predict_q50"],
            )
            # 概率评估接线（2026-08-30）：quantile 模式产出 tidy long 概率分数文件
            prob_path = result.test_dir / "test_scores_probabilistic_df.csv"
            self.assertTrue(prob_path.exists())
            prob_scores = pd.read_csv(prob_path)
            self.assertEqual(
                set(prob_scores["metric"]),
                {
                    "mae",
                    "bias",
                    "pinball",
                    "interval_coverage",
                    "interval_width",
                    "interval_winkler",
                    "coverage_gap",
                    "calibration_error",
                },
            )
            # 概率分数与点分数同拆分口径（2026-09-03 方案 B）：汇总文件只含
            # target/aggregate；per-horizon 拆到 horizon 明细文件
            self.assertEqual(set(prob_scores["scope"]), {"target", "aggregate"})
            prob_horizon = pd.read_csv(
                result.test_dir / "test_scores_probabilistic_horizon_df.csv"
            )
            self.assertEqual(
                set(prob_horizon["scope"]),
                {"horizon", "aggregate_horizon"},
            )
            self.assertEqual(
                set(prob_scores["interval_name"].dropna()), {"central80"}
            )
            target_pinball = prob_scores[
                (prob_scores["metric"] == "pinball")
                & (prob_scores["scope"] == "target")
            ]
            self.assertEqual(
                sorted(target_pinball["quantile"].tolist()), [0.1, 0.5, 0.9]
            )
            # pinball@0.5 = MAE / 2（同一点预测）：与点分数文件交叉验证
            point_scores = pd.read_csv(result.test_dir / "test_scores_df.csv")
            point_mae = point_scores[
                (point_scores["scope"] == "target")
                & (point_scores["target"] == "load")
            ]["MAE"].iloc[0]
            pin50 = target_pinball[target_pinball["quantile"] == 0.5]["value"].iloc[0]
            self.assertAlmostEqual(pin50, point_mae / 2.0, places=9)

    def test_local_k2_point_and_quantile_restore_target_space_and_pickle_transforms(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            times = pd.date_range("2026-01-01", periods=96, freq="1h")
            step = np.arange(96, dtype=float)
            frame = pd.DataFrame(
                {
                    "time": times,
                    "load": 10.0 + step * 2.0,
                    "power": 100_000.0 + step * 5_000.0,
                }
            )
            data_path = root / "targets.csv"
            frame.to_csv(data_path, index=False)

            for mode in ("point", "quantile"):
                with self.subTest(mode=mode):
                    config = self.build_transformed_k2_config(data_path, mode=mode)
                    result = run_canonical_config(config, output_root=root / "results")
                    prediction = pd.read_csv(result.forecast_dir / "prediction.csv")
                    expected = frame.iloc[84:86].set_index("time")

                    for target in ("load", "power"):
                        actual = prediction.loc[
                            prediction["target"] == target,
                            "predict_value",
                        ].to_numpy()
                        np.testing.assert_allclose(
                            actual,
                            expected[target].to_numpy(),
                            atol=1e-5,
                        )
                    if mode == "quantile":
                        np.testing.assert_allclose(
                            prediction["predict_value"],
                            prediction["predict_q50"],
                            atol=1e-10,
                        )

                    self.assertIsNotNone(result.bundle.target_transform)
                    self.assertIsNotNone(result.bundle.feature_scaler)
                    with (result.model_dir / "model.pkl").open("rb") as handle:
                        restored_bundle = pickle.load(handle)
                    self.assertEqual(
                        set(restored_bundle.target_transform.fitted_keys),
                        {("__local__", "load"), ("__local__", "power")},
                    )
                    self.assertEqual(
                        restored_bundle.feature_scaler.method,
                        "standard",
                    )

    def test_holdout_training_labels_end_before_holdout_labels_start(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            times = pd.date_range("2026-01-01", periods=72, freq="1h")
            data_path = root / "load.csv"
            pd.DataFrame(
                {"time": times, "load": 100.0 + np.arange(72, dtype=float)}
            ).to_csv(data_path, index=False)
            config = self.build_config(data_path, mode="point", horizon=4)

            result = run_canonical_config(config, output_root=root / "results")

            resolved = json.loads(
                (result.forecast_dir / "resolved_config.json").read_text(
                    encoding="utf-8"
                )
            )
            holdout = resolved["runtime"]["holdout"]
            self.assertLess(
                pd.Timestamp(holdout["training_label_end_max"]),
                pd.Timestamp(holdout["label_start"]),
            )
            self.assertEqual(holdout["excluded_overlapping_samples"], 3)

    def test_runtime_materializes_role_sources_and_persists_visibility_audit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            times = pd.date_range("2026-01-01", periods=72, freq="1h")
            origin = times[60]
            pd.DataFrame(
                {
                    "time": times,
                    "load": 100.0 + np.arange(72, dtype=float),
                    "power": 20.0 + np.arange(72, dtype=float) * 0.5,
                }
            ).to_csv(root / "targets.csv", index=False)
            pd.DataFrame(
                {
                    "time": times,
                    "humidity": 40.0 + np.arange(72, dtype=float) * 0.1,
                }
            ).to_csv(root / "observed.csv", index=False)
            weather_rows = [
                {
                    "time": timestamp,
                    "issued_at": times[0],
                    "temperature": 10.0 + index,
                }
                for index, timestamp in enumerate(times[1:])
            ]
            weather_rows.append(
                {
                    "time": times[61],
                    "issued_at": origin + pd.Timedelta(minutes=1),
                    "temperature": 999.0,
                }
            )
            pd.DataFrame(weather_rows).to_csv(root / "weather.csv", index=False)
            pd.DataFrame([{"site_id": "only", "capacity": 500.0}]).to_csv(
                root / "static.csv",
                index=False,
            )

            config = ForecastConfigSpec(
                problem=ForecastProblemSpec(
                    time_col="time",
                    freq="1h",
                    horizon=2,
                    targets=("load", "power"),
                    information_mode="forecast",
                    training_scope="local",
                    series_id_cols=(),
                ),
                data=DataSpec(
                    (
                        DataSourceSpec(
                            name="targets",
                            source_type="file",
                            columns=(
                                ColumnSpec("load", "target"),
                                ColumnSpec("power", "target"),
                            ),
                            history_path=str(root / "targets.csv"),
                            time_col="time",
                            availability="source_time",
                        ),
                        DataSourceSpec(
                            name="observed",
                            source_type="file",
                            columns=(ColumnSpec("humidity", "observed_past"),),
                            history_path=str(root / "observed.csv"),
                            time_col="time",
                            availability="source_time",
                            provider="persistence",
                        ),
                        DataSourceSpec(
                            name="weather",
                            source_type="file",
                            columns=(ColumnSpec("temperature", "known_future"),),
                            future_path=str(root / "weather.csv"),
                            time_col="time",
                            availability="column",
                            available_at_col="issued_at",
                        ),
                        DataSourceSpec(
                            name="site",
                            source_type="file",
                            columns=(
                                ColumnSpec("site_id", "key", categorical=True),
                                ColumnSpec("capacity", "static"),
                            ),
                            history_path=str(root / "static.csv"),
                            series_id_cols=("site_id",),
                        ),
                        DataSourceSpec(
                            name="calendar",
                            source_type="generated",
                            columns=(ColumnSpec("holiday", "known_future"),),
                            time_col="time",
                            availability="generator_defined",
                            generator="calendar",
                        ),
                    )
                ),
                features=FeatureSpec(
                    target_lags={"load": (2, 3), "power": (2, 3)},
                    observed_past_lags={"humidity": (1,)},
                    datetime_features=("hour", "year", "is_month_start"),
                    transformations={
                        "feature_scaling": {
                            "method": "none",
                            "grouped": False,
                            "encode_categorical": False,
                        },
                        "target": {
                            "calendar_normalization": {"method": "none"},
                            "decomposition": {"method": "none"},
                            "scaling": {"method": "none", "inverse": False},
                        },
                    },
                ),
                strategy=ForecastStrategySpec("direct"),
                estimator=EstimatorSpec(
                    model_type="ridge",
                    target_adapter="independent",
                    params={"alpha": 1e-6},
                ),
                probabilistic={"mode": "point"},
                validation={
                    "forecast_origin": origin.isoformat(),
                    "history_steps": 10_000,
                    "train_window_steps": 9_999,
                    "fold_count": 1,
                    "stride_steps": 2,
                },
                output={
                    "identity": {
                        "scenario_subpath": "canonical/roles",
                        "setting_suffix": "",
                    },
                    "directories": {
                        "checkpoints": str(root / "models"),
                        "tests": str(root / "tests"),
                        "forecast": str(root / "forecast"),
                    },
                },
            )

            def calendar_generator(_source, request):
                return pd.DataFrame(
                    {
                        "time": request.forecast_times,
                        "available_at": [request.forecast_origin] * request.H,
                        "holiday": [0.0] * request.H,
                    }
                )

            result = run_canonical_config(
                config,
                generators={"calendar": calendar_generator},
            )

            self.assertTrue(str(result.model_dir).startswith(str(root / "models")))
            self.assertTrue(str(result.test_dir).startswith(str(root / "tests")))
            self.assertTrue(str(result.forecast_dir).startswith(str(root / "forecast")))
            self.assertIn("temperature", result.bundle.selected_features)
            self.assertIn("capacity", result.bundle.selected_features)
            self.assertIn("holiday", result.bundle.selected_features)
            self.assertIn("humidity__lag_1", result.bundle.selected_features)
            self.assertTrue(result.bundle.source_lineage)
            self.assertTrue(result.bundle.input_schema["visibility_proof"])
            resolved = json.loads(
                (result.forecast_dir / "resolved_config.json").read_text(encoding="utf-8")
            )
            self.assertTrue(resolved["runtime"]["source_lineage"])
            self.assertTrue(resolved["runtime"]["visibility_proof"])
            weather_proof = next(
                item
                for item in resolved["runtime"]["visibility_proof"]
                if item["feature_name"] == "temperature"
            )
            self.assertLessEqual(
                pd.Timestamp(weather_proof["available_at"]),
                pd.Timestamp(weather_proof["forecast_origin"]),
            )

    def test_recursive_designs_roll_target_lags_with_fixed_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            times = pd.date_range("2026-01-01", periods=24, freq="1h")
            data_path = root / "load.csv"
            pd.DataFrame(
                {"time": times, "load": np.arange(24, dtype=float)}
            ).to_csv(data_path, index=False)

            cases = (
                ("recursive", None, [[8, 7, 6], [9, 8, 7], [10, 9, 8], [11, 10, 9]]),
                ("recmo", 2, [[8, 7, 6], [10, 9, 8]]),
                ("dirrec", None, [[8, 7, 6], [9, 8, 7], [10, 9, 8], [11, 10, 9]]),
                ("dirrecmo", 2, [[8, 7, 6], [10, 9, 8]]),
            )
            for strategy, chunk_length, expected_lags in cases:
                with self.subTest(strategy=strategy):
                    config = self.build_config(
                        data_path,
                        mode="point",
                        strategy=strategy,
                        horizon=4,
                        output_chunk_length=chunk_length,
                    )
                    builder = _RegistryDesignBuilder(
                        config,
                        SourceRegistry(config.data, root),
                    )

                    designs, _ = builder.training_row(pd.Timestamp("2026-01-01T08:00:00"))

                    self.assertEqual(
                        builder.feature_schema,
                        ("load__lag_1", "load__lag_2", "load__lag_3", "dt_hour"),
                    )
                    np.testing.assert_allclose(
                        np.vstack(designs)[:, :3],
                        np.asarray(expected_lags, dtype=float),
                    )

    def test_training_target_provider_exposes_only_each_calls_dependencies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            times = pd.date_range("2026-01-01", periods=24, freq="1h")
            data_path = root / "load.csv"
            pd.DataFrame(
                {"time": times, "load": np.arange(24, dtype=float)}
            ).to_csv(data_path, index=False)

            for strategy in ("direct", "recursive"):
                with self.subTest(strategy=strategy):
                    config = self.build_config(
                        data_path,
                        mode="point",
                        strategy=strategy,
                        horizon=4,
                    )
                    builder = _RegistryDesignBuilder(
                        config,
                        SourceRegistry(config.data, root),
                    )
                    original_compiler = builder.compiler
                    probed_calls = []

                    class ProbingCompiler:
                        def compile(probe_self, *args, **kwargs):
                            call_index = len(probed_calls)
                            provider = kwargs["target_future_providers"][()]
                            dependencies = builder.plan.dependencies[call_index]
                            for coordinate in dependencies:
                                self.assertEqual(
                                    provider.value_at(
                                        coordinate.target,
                                        coordinate.horizon_step - 1,
                                    ),
                                    float(coordinate.horizon_step + 8),
                                )
                            current = builder.plan.call_coordinates[call_index][0]
                            with self.assertRaisesRegex(
                                ValueError,
                                "not an allowed prior dependency",
                            ):
                                provider.value_at(
                                    current.target,
                                    current.horizon_step - 1,
                                )
                            later_step = min(
                                config.problem.horizon,
                                current.horizon_step + 1,
                            )
                            if later_step != current.horizon_step:
                                with self.assertRaisesRegex(
                                    ValueError,
                                    "not an allowed prior dependency",
                                ):
                                    provider.value_at("load", later_step - 1)
                            probed_calls.append(call_index)
                            return original_compiler.compile(*args, **kwargs)

                    builder.compiler = ProbingCompiler()
                    _, labels = builder.training_row(
                        pd.Timestamp("2026-01-01T08:00:00")
                    )

                    self.assertEqual(
                        probed_calls,
                        list(range(len(builder.plan.call_coordinates))),
                    )
                    np.testing.assert_allclose(labels[:, 0], [9.0, 10.0, 11.0, 12.0])

    def test_quantile_recursive_forecast_uses_median_path_for_all_levels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            times = pd.date_range("2026-01-01", periods=12, freq="1h")
            data_path = root / "load.csv"
            pd.DataFrame(
                {"time": times, "load": 100.0 + np.arange(12, dtype=float)}
            ).to_csv(data_path, index=False)
            config = self.build_config(
                data_path,
                mode="quantile",
                strategy="recursive",
                horizon=3,
            )
            builder = _RegistryDesignBuilder(
                config,
                SourceRegistry(config.data, root),
            )
            designs, provider = builder.forecast_designs(
                pd.Timestamp("2026-01-01T11:00:00")
            )
            plan = StrategyTargetPlan.from_spec(
                config.strategy,
                config.problem.targets,
                config.problem.horizon,
            )
            artifacts = {}
            for level, offset in ((0.1, -20.0), (0.5, 10.0), (0.9, 20.0)):
                artifacts[level] = CanonicalStrategyArtifact(
                    target_plan=plan,
                    model_groups=(
                        StrategyModelGroupArtifact(
                            model_index=0,
                            coordinate_groups=plan.call_coordinates,
                            predictor=AdapterPredictor(_LagOffsetAdapter(0, offset)),
                        ),
                    ),
                    feature_schema=builder.feature_schema,
                    estimator_coupling="independent",
                    estimator_capabilities=MappingProxyType({}),
                    N=1,
                    H=3,
                    K=1,
                )
            artifact = CanonicalMarginalQuantileArtifact(
                levels=(0.1, 0.5, 0.9),
                point_level=0.5,
                artifacts_by_level=MappingProxyType(artifacts),
            )

            forecast = CanonicalMarginalQuantileForecaster(config, artifact).predict(
                designs[0],
                series_ids=("__local__",),
                forecast_times=pd.date_range(
                    "2026-01-01T12:00:00", periods=3, freq="1h"
                ),
                feature_provider=provider,
            )

            self.assertIsInstance(forecast.point, PointForecastTensor)
            np.testing.assert_allclose(
                forecast.quantiles.values[0, :, 0, :],
                np.asarray(
                    [
                        [91.0, 121.0, 131.0],
                        [101.0, 131.0, 141.0],
                        [111.0, 141.0, 151.0],
                    ]
                ),
            )

    def test_all_seven_point_strategies_support_two_targets(self):
        strategies = {
            "recursive": (None, 1, [0, 2, 4, 6]),
            "direct": (None, 4, [0, 0, 0, 0]),
            "mimo": (None, 1, [0]),
            "recmo": (2, 1, [0, 4]),
            "dirrec": (None, 4, [0, 2, 4, 6]),
            "dirmo": (2, 2, [0, 0]),
            "dirrecmo": (2, 2, [0, 4]),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            times = pd.date_range("2026-01-01", periods=72, freq="1h")
            data_path = root / "multi.csv"
            pd.DataFrame(
                {
                    "time": times,
                    "load": 100.0 + np.arange(72, dtype=float),
                    "power": 20.0 + np.arange(72, dtype=float) * 0.25,
                }
            ).to_csv(data_path, index=False)

            for strategy, (
                chunk_length,
                expected_model_count,
                expected_dependency_widths,
            ) in strategies.items():
                with self.subTest(strategy=strategy):
                    target_lags = (
                        (1, 2, 3)
                        if strategy in {"recursive", "recmo", "dirrec", "dirrecmo"}
                        else (4, 5, 6)
                    )
                    config = ForecastConfigSpec(
                        problem=ForecastProblemSpec(
                            time_col="time",
                            freq="1h",
                            horizon=4,
                            targets=("load", "power"),
                            information_mode="forecast",
                            training_scope="local",
                            series_id_cols=(),
                        ),
                        data=DataSpec(
                            (
                                DataSourceSpec(
                                    name="target_history",
                                    source_type="file",
                                    columns=(
                                        ColumnSpec("load", "target"),
                                        ColumnSpec("power", "target"),
                                    ),
                                    history_path=str(data_path),
                                    time_col="time",
                                    availability="source_time",
                                ),
                            )
                        ),
                        features=FeatureSpec(
                            target_lags={"load": target_lags, "power": target_lags},
                            observed_past_lags={},
                            datetime_features=("hour",),
                            transformations={},
                        ),
                        strategy=ForecastStrategySpec(
                            strategy,
                            output_chunk_length=chunk_length,
                        ),
                        estimator=EstimatorSpec(
                            model_type="ridge",
                            target_adapter="independent",
                            params={"alpha": 1e-6},
                        ),
                        probabilistic={"mode": "point"},
                        validation={
                            "forecast_origin": "2026-01-03T23:00:00",
                            "history_steps": 10_000,
                            "train_window_steps": 9_999,
                            "fold_count": 1,
                            "stride_steps": 4,
                        },
                        output={
                            "scenario_subpath": f"seven/{strategy}",
                            "setting_suffix": "",
                        },
                    )

                    result = run_canonical_config(
                        config,
                        output_root=root / "results",
                    )

                    prediction = pd.read_csv(result.forecast_dir / "prediction.csv")
                    scores = pd.read_csv(result.test_dir / "test_scores_df.csv")
                    self.assertEqual(len(prediction), 8)
                    self.assertEqual(set(prediction["target"]), {"load", "power"})
                    # K=2、H=4、1 窗：汇总 2 target + 1 aggregate；horizon 明细独立文件
                    self.assertEqual(
                        scores["scope"].tolist(),
                        ["target", "target", "aggregate"],
                    )
                    horizon_scores = pd.read_csv(
                        result.test_dir / "test_scores_horizon_df.csv"
                    )
                    self.assertEqual(
                        set(horizon_scores["scope"]),
                        {"horizon", "aggregate_horizon"},
                    )
                    self.assertEqual((horizon_scores["scope"] == "horizon").sum(), 2 * 4)
                    self.assertEqual(result.run_dir.name, config.result_identity())
                    self.assertEqual(
                        result.bundle.model.model_count,
                        expected_model_count,
                    )
                    self.assertEqual(
                        [
                            len(dependencies)
                            for dependencies in result.bundle.model.target_plan.dependencies
                        ],
                        expected_dependency_widths,
                    )


if __name__ == "__main__":
    unittest.main()
