# -*- coding: utf-8 -*-

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch
import tempfile
import textwrap
import unittest

import numpy as np
import pandas as pd

from main import Model, load_yaml_config
from config.univariate_config import ModelConfig
from data_provider.data_loader import DataLoader
from features.FeatureScalering import FeatureScaler
from models.ModelForecasting import Forecaster
from models.ModelSaveLoad import ModelDeployPkl
from models.multistep.artifacts import BlendArtifact
from models.multistep.panel import (
    execute_panel,
    materialize_panel_future,
    materialize_panel_history,
    split_panel_window,
)
from models.multistep.resolve import resolve_strategy
from models.multistep.weights import BlendWeights
from probabilistic.spec import resolve_probabilistic_spec
from probabilistic.types import ForecastModelBundle


class GlobalPanelDataContractTest(unittest.TestCase):
    def setUp(self):
        self.times = pd.date_range("2026-01-01", periods=6, freq="1h")
        self.source = pd.DataFrame(
            {
                "series_id": ["A"] * 6 + ["B"] * 6,
                "ts": list(self.times) * 2,
                "load": list(range(6)) + list(range(100, 106)),
            }
        )

    def test_same_timestamps_remain_unique_by_composite_key(self):
        history = materialize_panel_history(
            self.source,
            self.times,
            series_id_col="series_id",
            source_time_col="ts",
            target_col="load",
        )
        self.assertEqual(len(history), 12)
        self.assertFalse(history.duplicated(["series_id", "time"]).any())
        self.assertEqual(history.groupby("series_id", sort=False)["y"].first().to_dict(), {"A": 0, "B": 100})

    def test_data_loader_deduplicates_by_composite_key(self):
        loader = DataLoader.__new__(DataLoader)
        loader.args = SimpleNamespace(
            enable_global_training=True,
            series_id_feature="series_id",
        )
        loader.log_prefix = "[panel-test]"
        processed = loader._DataLoader__process_df_timestamp(self.source, "ts")
        self.assertEqual(len(processed), 12)
        self.assertFalse(processed.duplicated(["series_id", "ts"]).any())

    def test_incomplete_series_policy_is_explicit(self):
        incomplete = self.source.drop(self.source.index[-1])
        with self.assertRaisesRegex(ValueError, "series 'B' is incomplete"):
            materialize_panel_history(
                incomplete,
                self.times,
                series_id_col="series_id",
                source_time_col="ts",
                target_col="load",
            )
        dropped = materialize_panel_history(
            incomplete,
            self.times,
            series_id_col="series_id",
            source_time_col="ts",
            target_col="load",
            incomplete_policy="drop",
        )
        self.assertEqual(tuple(pd.unique(dropped["series_id"])), ("A",))

    def test_window_boundaries_are_computed_per_series(self):
        history = materialize_panel_history(
            self.source,
            self.times,
            series_id_col="series_id",
            source_time_col="ts",
            target_col="load",
        )
        train, test = split_panel_window(
            history,
            series_id_col="series_id",
            window=1,
            horizon=2,
            window_len=6,
        )
        self.assertEqual(train.groupby("series_id", sort=False).size().to_dict(), {"A": 4, "B": 4})
        self.assertEqual(test.groupby("series_id", sort=False).size().to_dict(), {"A": 2, "B": 2})
        self.assertEqual(test.groupby("series_id", sort=False)["y"].first().to_dict(), {"A": 4, "B": 104})

    def test_execute_panel_isolates_state_and_preserves_order(self):
        history = materialize_panel_history(
            self.source,
            self.times,
            series_id_col="series_id",
            source_time_col="ts",
            target_col="load",
        )
        future = materialize_panel_future(
            ("A", "B"),
            pd.date_range("2026-01-01 06:00", periods=2, freq="1h"),
            series_id_col="series_id",
        )
        original = history.copy(deep=True)

        def execute_one(series_slice):
            series_slice.history.loc[:, "y"] = -1
            value = 1.0 if series_slice.series_id == "A" else 2.0
            output = series_slice.future.copy()
            output["predict_value"] = value
            return output

        result = execute_panel(
            history,
            future,
            series_id_col="series_id",
            horizon=2,
            execute_one=execute_one,
        )
        pd.testing.assert_frame_equal(history, original)
        self.assertEqual(tuple(pd.unique(result["series_id"])), ("A", "B"))
        self.assertEqual(result.groupby("series_id", sort=False)["predict_value"].first().to_dict(), {"A": 1.0, "B": 2.0})

    def test_feature_scaler_rejects_unknown_series_id(self):
        args = SimpleNamespace(
            scale_features=False,
            encode_categorical_features=True,
            use_grouped_scaling=False,
            enable_global_training=True,
            series_id_feature="series_id",
            global_unknown_series_policy="raise",
        )
        scaler = FeatureScaler(args)
        scaler.fit_transform(
            pd.DataFrame({"series_id": ["A", "B"], "x": [1.0, 2.0]}),
            ["series_id"],
        )
        with self.assertRaisesRegex(ValueError, "unknown series IDs.*C"):
            scaler.transform(
                pd.DataFrame({"series_id": ["C"], "x": [3.0]}),
                ["series_id"],
            )


class GlobalPanelForecastDispatchTest(unittest.TestCase):
    class DummyForecaster:
        def __init__(self, **kwargs):
            self.df_history = kwargs["df_history"]
            self.df_future = kwargs["df_future"]
            self.blend_direct_pred = None
            self.blend_recursive_pred = None
            self.quantile_outputs = None

        def _predict_by_method(self):
            return np.full(
                len(self.df_future),
                float(self.df_history["y"].iloc[-1]) + 1.0,
            )

    def test_all_five_rollout_families_dispatch_each_series_independently(self):
        methods = (
            "univariate-single-multistep-direct-pointwise",
            "univariate-single-multistep-direct",
            "univariate-single-multistep-recursive",
            "univariate-single-multistep-direct-recursive",
            "univariate-single-multistep-blend-direct-recursive",
        )
        history = pd.DataFrame(
            {
                "series_id": ["A", "A", "B", "B"],
                "time": pd.to_datetime(
                    ["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"]
                ),
                "y": [1.0, 2.0, 10.0, 20.0],
            }
        )
        future = materialize_panel_future(
            ("A", "B"),
            pd.date_range("2026-01-03", periods=2, freq="1D"),
            series_id_col="series_id",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for method in methods:
                with self.subTest(method=method):
                    args = SimpleNamespace(
                        pred_method=method,
                        lags=[1],
                        direct_strategy="multioutput",
                        block_size=1 if "direct-recursive" in method and "blend" not in method else 0,
                        blend_weight_strategy="fixed",
                        blend_weights=[0.5, 0.5],
                        enable_global_training=True,
                        series_id_feature="series_id",
                        global_incomplete_series_policy="raise",
                        target_calendar_normalization="none",
                        decomposition_method="none",
                        endogenous_backfill_strategy="persistence",
                        horizon_mode="fixed_steps",
                        quantile_monotone=False,
                        test_results_dir=Path(temp_dir),
                    )
                    resolve_strategy(args, horizon=2)
                    model = Model.__new__(Model)
                    model.args = args
                    model.horizon = 2
                    model.n_per_day = 1
                    model.log_prefix = "[panel-test]"
                    model.probabilistic_spec = resolve_probabilistic_spec(args)
                    model.target_decomposer = None
                    model.target_transform = None
                    with patch("main.Forecaster", self.DummyForecaster):
                        result = model.forecast(
                            model=object(),
                            scaler_forecasting=None,
                            target_scaler_forecasting=None,
                            df_history=history,
                            df_future=future,
                            df_date_future=None,
                            df_weather_future=None,
                            df_custom_future=None,
                            endogenous_features_with_target=["y"],
                            target_feature="y",
                            target_output_features=["y_shift_1", "y_shift_2"],
                            categorical_features=["series_id"],
                            _save_results=False,
                        )
                    self.assertEqual(len(result), 4)
                    self.assertEqual(
                        result.groupby("series_id", sort=False)["predict_value"].first().to_dict(),
                        {"A": 3.0, "B": 21.0},
                    )


class GlobalPanelStrategyExecutorTest(unittest.TestCase):
    class ShapeModel:
        def __init__(self, width: int, pointwise: bool = False):
            self.width = width
            self.pointwise = pointwise

        def predict(self, X):
            if self.pointwise:
                return np.arange(1, len(X) + 1, dtype=float)
            if self.width == 1:
                return np.asarray([1.0])
            return np.arange(1, self.width + 1, dtype=float).reshape(1, -1)

    @staticmethod
    def _args(method: str) -> ModelConfig:
        args = ModelConfig()
        args.pred_method = method
        args.predict_type = "point"
        args.predict_steps = 2
        args.lags = [1]
        args.enable_lags_features = True
        args.enable_advanced_features = False
        args.enable_global_training = True
        args.series_id_feature = "series_id"
        args.global_incomplete_series_policy = "raise"
        args.global_unknown_series_policy = "raise"
        args.target_calendar_normalization = "none"
        args.decomposition_method = "none"
        args.endogenous_backfill_strategy = "persistence"
        args.horizon_mode = "fixed_steps"
        args.direct_strategy = "multioutput"
        args.blend_weight_strategy = "fixed"
        args.blend_weights = [0.5, 0.5]
        args.block_size = 1 if method == "usmdr" else 0
        return args

    def test_all_nine_strategies_execute_real_forecaster_per_series(self):
        cases = {
            "usmdp": self.ShapeModel(1, pointwise=True),
            "usmd": self.ShapeModel(2),
            "usmr": self.ShapeModel(1),
            "usmdr": self.ShapeModel(1),
            "usbr": BlendArtifact(
                direct_model=self.ShapeModel(2),
                recursive_model=self.ShapeModel(1),
                weights=BlendWeights(0.5, 0.5, strategy="fixed"),
            ),
            "msmd": self.ShapeModel(2),
            "msmr": self.ShapeModel(1),
            "msmdr": self.ShapeModel(1),
            "msbr": BlendArtifact(
                direct_model=self.ShapeModel(2),
                recursive_model=self.ShapeModel(1),
                weights=BlendWeights(0.5, 0.5, strategy="fixed"),
            ),
        }
        history = pd.DataFrame(
            {
                "series_id": ["A"] * 4 + ["B"] * 4,
                "time": list(pd.date_range("2026-01-01", periods=4, freq="1D")) * 2,
                "x": [5.0, 6.0, 7.0, 8.0, 50.0, 60.0, 70.0, 80.0],
                "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            }
        )
        future = materialize_panel_future(
            ("A", "B"),
            list(pd.date_range("2026-01-05", periods=2, freq="1D")),
            series_id_col="series_id",
        )

        for method, model in cases.items():
            with self.subTest(method=method):
                args = self._args(method)

                def execute_one(series_slice):
                    predictor = Forecaster(
                        args=args,
                        horizon=2,
                        model=model,
                        feature_scaler=None,
                        target_scaler=None,
                        df_history=series_slice.history,
                        df_future=series_slice.future,
                        df_date_future=None,
                        df_weather_future=None,
                        df_custom_future=None,
                        endogenous_features=(
                            ["x", "y"] if method.startswith("ms") else ["y"]
                        ),
                        target_feature="y",
                        target_output_features=(
                            ["y_shift_1", "y_shift_2"]
                            if method in {"usmd", "usbr", "msmd", "msbr"}
                            else ["y_shift_1"]
                            if method in {"usmdr", "msmdr"}
                            else ["y_shift_0"]
                        ),
                        categorical_features=["series_id"],
                        selected_features=None,
                        log_prefix=f"[panel-{method}]",
                    )
                    values = np.asarray(predictor._predict_by_method()).reshape(-1)
                    output = series_slice.future.copy()
                    output["predict_value"] = values
                    return output

                result = execute_panel(
                    history,
                    future,
                    series_id_col="series_id",
                    horizon=2,
                    execute_one=execute_one,
                )
                self.assertEqual(len(result), 4)
                self.assertTrue(np.isfinite(result["predict_value"]).all())


class GlobalPanelPipelineSmokeTest(unittest.TestCase):
    def test_yaml_driven_two_series_pipeline_trains_persists_and_forecasts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            times = pd.date_range("2026-01-01", periods=20, freq="1D")
            source = pd.DataFrame(
                {
                    "series_id": ["A"] * len(times) + ["B"] * len(times),
                    "time": list(times) * 2,
                    "value": (
                        list(np.linspace(10.0, 29.0, len(times)))
                        + list(np.linspace(100.0, 138.0, len(times)))
                    ),
                }
            )
            source.to_csv(root / "panel.csv", index=False)
            config_path = root / "panel_smoke.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    base_config: config.univariate_config
                    overrides:
                      output:
                        scenario_subpath: panel-smoke
                        checkpoints_dir: {root / 'models'}
                        test_results_dir: {root / 'test'}
                        pred_results_dir: {root / 'forecast'}
                      runtime:
                        schedule_mode: daily
                        horizon_mode: fixed_steps
                        is_testing: false
                        is_forecasting: true
                        history_length: 20
                        predict_steps: 2
                        window_length: 10
                        now_time: '2026-01-20T00:00:00'
                      target_series:
                        data_dir: {root}
                        data_path: panel.csv
                        freq: 1D
                        target_ts_feat: time
                        target: value
                        target_series_categorical_features: [series_id]
                      exogenous_features:
                        enable_date_features: false
                        enable_weather_features: false
                        enable_datetime_features: false
                        datetime_features: []
                        datetime_categorical_features: []
                      time_lag_features:
                        enable_lags_features: true
                        lags: [1, 2]
                        enable_advanced_features: false
                      preprocessing:
                        scale_features: true
                        feature_scaler_type: standard
                        scale_target: false
                        target_calendar_normalization: none
                        decomposition_method: none
                        encode_categorical_features: true
                      model_strategy:
                        model_type: ridge
                        model_params:
                          alpha: 1.0
                        pred_method: univariate-single-multistep-recursive
                        predict_type: point
                        enable_global_training: true
                        series_id_feature: series_id
                        global_incomplete_series_policy: raise
                        global_unknown_series_policy: raise
                      performance:
                        window_parallel_workers: 1
                        multi_output_n_jobs: 1
                        model_thread_count: 1
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            args = load_yaml_config(config_path)
            pipeline = Model(args)
            pipeline.run()

            model_path = pipeline.args.checkpoints_dir / "model.pkl"
            prediction_path = pipeline.args.pred_results_dir / "prediction.csv"
            self.assertTrue(model_path.exists())
            self.assertTrue(prediction_path.exists())

            bundle = ModelDeployPkl(model_path).load_model()
            self.assertIsInstance(bundle, ForecastModelBundle)
            bundle = cast(ForecastModelBundle, bundle)
            self.assertEqual(
                bundle.input_schema["panel"]["known_series_ids"],
                ["A", "B"],
            )

            prediction = pd.read_csv(prediction_path)
            self.assertEqual(
                prediction.groupby("series_id", sort=False).size().to_dict(),
                {"A": 2, "B": 2},
            )
            self.assertTrue(np.isfinite(prediction["predict_value"]).all())


if __name__ == "__main__":
    unittest.main()
