# -*- coding: utf-8 -*-
"""Visibility-driven canonical feature compiler tests."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_loading import (
    InformationSetRequest,
    ProvidedScenarioProvider,
    SourceRegistry,
)
from feature_engineering import FeatureCompiler
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


class FeatureVisibilityCompilerTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_csv(self, name, rows):
        pd.DataFrame(rows).to_csv(self.base_dir / name, index=False)

    def build_config(
        self,
        *,
        strategy="direct",
        target_lags=None,
        observed_lags=None,
        global_scope=False,
        datetime_features=("hour",),
        transformations=None,
    ):
        series_id_cols = ("site_id",) if global_scope else ()
        key_column = (ColumnSpec("site_id", "key", categorical=True),) if global_scope else ()
        sources = [
                DataSourceSpec(
                    name="targets",
                    source_type="file",
                    columns=(*key_column, ColumnSpec("load", "target"), ColumnSpec("power", "target")),
                    history_path="targets.csv",
                    time_col="ts",
                    series_id_cols=series_id_cols,
                    availability="source_time",
                ),
                DataSourceSpec(
                    name="sensors",
                    source_type="file",
                    columns=(*key_column, ColumnSpec("humidity", "observed_past")),
                    history_path="sensors.csv",
                    time_col="ts",
                    series_id_cols=series_id_cols,
                    availability="source_time",
                    provider="provided_scenario",
                ),
                DataSourceSpec(
                    name="weather",
                    source_type="file",
                    columns=(*key_column, ColumnSpec("temperature", "known_future")),
                    backtest_path="weather_backtest.csv",
                    future_path="weather_future.csv",
                    time_col="ts",
                    series_id_cols=series_id_cols,
                    availability="column",
                    available_at_col="issued_at",
                ),
        ]
        if global_scope:
            sources.append(
                DataSourceSpec(
                    name="sites",
                    source_type="file",
                    columns=(*key_column, ColumnSpec("capacity", "static")),
                    history_path="sites.csv",
                    series_id_cols=series_id_cols,
                ),
            )
        data = DataSpec(tuple(sources))
        return ForecastConfigSpec(
            problem=ForecastProblemSpec(
                time_col="ts",
                freq="1h",
                horizon=2,
                targets=("load", "power"),
                information_mode="forecast",
                training_scope="global" if global_scope else "local",
                series_id_cols=series_id_cols,
            ),
            data=data,
            features=FeatureSpec(
                target_lags=target_lags or {"load": (2,), "power": (2,)},
                observed_past_lags=observed_lags or {"humidity": (2,)},
                datetime_features=datetime_features,
                transformations=transformations or {},
            ),
            strategy=ForecastStrategySpec(strategy),
            estimator=EstimatorSpec(model_type="ridge", target_adapter="independent"),
            probabilistic={},
            validation={},
            output={},
        )

    def write_fixture(self, *, global_scope=False):
        identities = (("A", 0.0), ("B", 1000.0)) if global_scope else ((None, 0.0),)
        targets = []
        sensors = []
        weather = []
        sites = []
        for site_id, offset in identities:
            identity = {"site_id": site_id} if global_scope else {}
            for step, hour in enumerate((1, 2, 3), start=1):
                targets.append(
                    {
                        **identity,
                        "ts": f"2026-01-01 0{hour}:00",
                        "load": offset + step,
                        "power": offset + step * 10,
                    }
                )
                sensors.append(
                    {
                        **identity,
                        "ts": f"2026-01-01 0{hour}:00",
                        "humidity": offset + step * 100,
                    }
                )
            for hour, value in ((4, 40.0), (5, 50.0)):
                weather.append(
                    {
                        **identity,
                        "ts": f"2026-01-01 0{hour}:00",
                        "issued_at": "2026-01-01 03:00",
                        "temperature": offset + value,
                    }
                )
            sites.append({**identity, "capacity": offset + 500.0})
        self.write_csv("targets.csv", targets)
        self.write_csv("sensors.csv", sensors)
        self.write_csv(
            "weather_backtest.csv",
            [
                {
                    **row,
                    "issued_at": "2026-01-01 02:00",
                    "temperature": row["temperature"] - 1.0,
                }
                for row in weather
            ],
        )
        self.write_csv("weather_future.csv", weather)
        self.write_csv("sites.csv", sites)

    def request(self, *, global_scope=False):
        return InformationSetRequest(
            forecast_origin="2026-01-01 03:00",
            forecast_times=pd.date_range("2026-01-01 04:00", periods=2, freq="1h"),
            series_ids=("A", "B") if global_scope else (),
            information_mode="forecast",
        )

    def materialize(self, config, request):
        return SourceRegistry(config.data, self.base_dir).materialize(request)

    def test_direct_uses_only_visible_lags_and_target_time_known_future(self):
        self.write_fixture()
        config = self.build_config()
        request = self.request()

        compiled = FeatureCompiler(config).compile(
            self.materialize(config, request),
            request,
        )

        self.assertEqual(compiled.frame["load__lag_2"].tolist(), [2.0, 3.0])
        self.assertEqual(compiled.frame["power__lag_2"].tolist(), [20.0, 30.0])
        self.assertEqual(compiled.frame["humidity__lag_2"].tolist(), [200.0, 300.0])
        self.assertEqual(compiled.frame["temperature"].tolist(), [40.0, 50.0])
        self.assertEqual(compiled.frame["dt_hour"].tolist(), [4, 5])
        self.assertEqual(compiled.frame["horizon_step"].tolist(), [1, 2])
        self.assertEqual(compiled.schema.feature_names, tuple(compiled.frame.columns[2:]))
        lag_proofs = [
            proof
            for proof in compiled.visibility_proof
            if proof.role in {"target", "observed_past"}
        ]
        self.assertTrue(lag_proofs)
        self.assertTrue(
            all(proof.source_time <= request.forecast_origin for proof in lag_proofs)
        )
        self.assertTrue(
            all(
                proof.available_at is not None
                and proof.available_at <= request.forecast_origin
                for proof in compiled.visibility_proof
            )
        )

    def test_one_source_can_project_target_and_observed_past_lags(self):
        self.write_csv(
            "mixed.csv",
            [
                {"ts": "2026-01-01 01:00", "load": 1.0, "humidity": 10.0},
                {"ts": "2026-01-01 02:00", "load": 2.0, "humidity": 20.0},
                {"ts": "2026-01-01 03:00", "load": 3.0, "humidity": 30.0},
            ],
        )
        config = ForecastConfigSpec(
            problem=ForecastProblemSpec(
                time_col="ts",
                freq="1h",
                horizon=2,
                targets=("load",),
                information_mode="forecast",
                training_scope="local",
                series_id_cols=(),
            ),
            data=DataSpec(
                (
                    DataSourceSpec(
                        name="mixed",
                        source_type="file",
                        columns=(
                            ColumnSpec("load", "target"),
                            ColumnSpec("humidity", "observed_past"),
                        ),
                        history_path="mixed.csv",
                        time_col="ts",
                        series_id_cols=(),
                        availability="source_time",
                        provider="persistence",
                    ),
                )
            ),
            features=FeatureSpec(
                target_lags={"load": (2,)},
                observed_past_lags={"humidity": (2,)},
                datetime_features=(),
                transformations={},
            ),
            strategy=ForecastStrategySpec("direct"),
            estimator=EstimatorSpec(model_type="ridge", target_adapter="independent"),
            probabilistic={},
            validation={},
            output={},
        )
        request = self.request()

        compiled = FeatureCompiler(config).compile(
            self.materialize(config, request),
            request,
        )

        self.assertEqual(compiled.frame["load__lag_2"].tolist(), [2.0, 3.0])
        self.assertEqual(compiled.frame["humidity__lag_2"].tolist(), [20.0, 30.0])

    def test_compile_resets_derived_cache_between_information_sets(self):
        self.write_fixture()
        config = self.build_config()
        request = self.request()
        compiler = FeatureCompiler(config)
        compiler._compile_scope_aux["stale"] = object()

        compiler.compile(self.materialize(config, request), request)

        self.assertNotIn("stale", compiler._compile_scope_aux)

    def test_direct_rejects_target_lag_that_would_consume_a_future_prediction(self):
        self.write_fixture()
        config = self.build_config(target_lags={"load": (1,), "power": (1,)})
        request = self.request()
        provider = ProvidedScenarioProvider(
            horizon=2,
            trajectories={"load": (4.0, 5.0), "power": (40.0, 50.0)},
        )

        with self.assertRaisesRegex(ValueError, "direct.*future target.*load"):
            FeatureCompiler(config).compile(
                self.materialize(config, request),
                request,
                target_future_providers={(): provider},
            )

    def test_safe_lag_at_least_horizon_never_invokes_providers(self):
        """safe-lag 合同：min(lags) >= horizon 时全程真实历史、零 provider。

        目标日对齐（默认 align_to_target=true）下 lag_3 对 h=1,2 的
        source_time（h−3）全部 <= origin，walk 全走 history 分支——
        即使误传 provider 也不得被消费（M5 forecast-date-aligned 模式，
        仓库主力语义，见 USMDP target_lags [31,35,42,49,56]）。
        """
        self.write_fixture()
        config = self.build_config(target_lags={"load": (3,), "power": (3,)})
        request = self.request()
        # 误传的 provider：若实现错误地走到 provider 路径，值会与历史真值不同。
        decoy_provider = ProvidedScenarioProvider(
            horizon=2,
            trajectories={"load": (999.0, 999.0), "power": (999.0, 999.0)},
        )

        compiled = FeatureCompiler(config).compile(
            self.materialize(config, request),
            request,
            target_future_providers={(): decoy_provider},
        )

        # h=1(04:00): lag_3 -> 01:00 -> load=1, power=10
        # h=2(05:00): lag_3 -> 02:00 -> load=2, power=20
        self.assertEqual(compiled.frame["load__lag_3"].tolist(), [1.0, 2.0])
        self.assertEqual(compiled.frame["power__lag_3"].tolist(), [10.0, 20.0])
        safe_proofs = [
            proof
            for proof in compiled.visibility_proof
            if proof.feature_name.endswith("__lag_3")
        ]
        # load 与 power 各 2 步，共 4 条 proof。
        self.assertEqual(len(safe_proofs), 4)
        self.assertTrue(all(proof.provider is None for proof in safe_proofs))
        self.assertTrue(
            all(
                proof.source_time is not None
                and proof.source_time <= request.forecast_origin
                for proof in safe_proofs
            )
        )
        self.assertEqual(
            sorted(str(proof.source_time) for proof in safe_proofs),
            sorted(
                [
                    str(pd.Timestamp("2026-01-01 01:00")),
                    str(pd.Timestamp("2026-01-01 02:00")),
                ]
                * 2
            ),
        )

    def test_ewm_and_ramp_statistics(self):
        """F1+F3：ewm 半衰期统计与爬坡统计（值按 pandas 定义手算）。"""
        self.write_fixture()
        config = self.build_config(
            transformations={
                "advanced": {
                    "ewm": {
                        "columns": ["load"],
                        "halflives": [2],
                        "stats": ["mean", "std"],
                    },
                    "rolling": {
                        "columns": ["load"],
                        "windows": [3],
                        "stats": ["max_diff", "min_diff"],
                    },
                }
            }
        )
        request = self.request()

        compiled = FeatureCompiler(config).compile(
            self.materialize(config, request),
            request,
        )

        # 可见历史 load = [1,2,3]（origin 03:00 前 3 点）。
        history = pd.Series([1.0, 2.0, 3.0])
        ewm = history.ewm(halflife=2, adjust=True)
        expected_mean = float(ewm.mean().iloc[-1])
        expected_std = float(ewm.std().iloc[-1])
        self.assertAlmostEqual(
            compiled.frame["load_ewm_mean_2.0"].tolist()[0], expected_mean, places=10
        )
        self.assertAlmostEqual(
            compiled.frame["load_ewm_std_2.0"].tolist()[0], expected_std, places=10
        )
        # rolling_3 窗 = [1,2,3]，相邻差分 [1,1]。
        self.assertEqual(compiled.frame["load_rolling_max_diff_3"].tolist(), [1.0, 1.0])
        self.assertEqual(compiled.frame["load_rolling_min_diff_3"].tolist(), [1.0, 1.0])

    def test_direct_can_freeze_history_lags_at_forecast_origin(self):
        """显式 align_to_target=false 时，Direct 各 horizon 共用原点历史。"""
        self.write_fixture()
        config = self.build_config(
            target_lags={"load": (1,), "power": (1,)},
            observed_lags={"humidity": (1,)},
            transformations={
                "direct": {
                    "layout": "independent_models",
                    "use_horizon_exogenous": True,
                    "align_to_target": False,
                }
            },
        )
        request = self.request()

        compiled = FeatureCompiler(config).compile(
            self.materialize(config, request),
            request,
        )

        self.assertEqual(compiled.frame["load__lag_1"].tolist(), [2.0, 2.0])
        self.assertEqual(compiled.frame["power__lag_1"].tolist(), [20.0, 20.0])
        self.assertEqual(compiled.frame["humidity__lag_1"].tolist(), [200.0, 200.0])
        self.assertEqual(compiled.frame["temperature"].tolist(), [40.0, 50.0])
        self.assertTrue(
            all(
                proof.source_time is not None
                and proof.source_time <= request.forecast_origin
                for proof in compiled.visibility_proof
                if proof.role in {"target", "observed_past"}
            )
        )

    def test_recursive_requires_explicit_future_target_and_observed_providers(self):
        self.write_fixture()
        config = self.build_config(
            strategy="recursive",
            target_lags={"load": (1,), "power": (1,)},
            observed_lags={"humidity": (1,)},
        )
        request = self.request()
        information_set = self.materialize(config, request)

        with self.assertRaisesRegex(ValueError, "future target provider"):
            FeatureCompiler(config).compile(information_set, request)

        target_provider = ProvidedScenarioProvider(
            horizon=2,
            trajectories={"load": (4.0, 5.0), "power": (40.0, 50.0)},
        )
        with self.assertRaisesRegex(ValueError, "observed_past provider"):
            FeatureCompiler(config).compile(
                information_set,
                request,
                target_future_providers={(): target_provider},
            )

        observed_provider = ProvidedScenarioProvider(
            horizon=2,
            trajectories={"humidity": (400.0, 500.0)},
        )
        compiled = FeatureCompiler(config).compile(
            information_set,
            request,
            target_future_providers={(): target_provider},
            observed_future_providers={(): observed_provider},
        )
        self.assertEqual(compiled.frame["load__lag_1"].tolist(), [3.0, 4.0])
        self.assertEqual(compiled.frame["power__lag_1"].tolist(), [30.0, 40.0])
        self.assertEqual(compiled.frame["humidity__lag_1"].tolist(), [300.0, 400.0])
        provider_proofs = [
            proof for proof in compiled.visibility_proof if proof.provider is not None
        ]
        self.assertTrue(provider_proofs)
        self.assertTrue(
            all(
                proof.available_at is not None
                and proof.available_at <= request.forecast_origin
                for proof in provider_proofs
            )
        )

    def test_provider_visibility_rejects_values_available_after_origin(self):
        self.write_fixture()
        config = self.build_config(
            strategy="recursive",
            target_lags={"load": (1,), "power": (1,)},
            observed_lags={"humidity": (2,)},
        )
        request = self.request()

        class LateProvider(ProvidedScenarioProvider):
            def available_at(self, _name, _step):
                return request.forecast_origin + pd.Timedelta(hours=1)

        provider = LateProvider(
            horizon=2,
            trajectories={"load": (4.0, 5.0), "power": (40.0, 50.0)},
        )

        with self.assertRaisesRegex(
            ValueError,
            "provider value.*available after forecast_origin",
        ):
            FeatureCompiler(config).compile(
                self.materialize(config, request),
                request,
                target_future_providers={(): provider},
            )

    def test_global_k2_compilation_is_isolated_by_series(self):
        self.write_fixture(global_scope=True)
        config = self.build_config(global_scope=True)
        request = self.request(global_scope=True)

        compiled = FeatureCompiler(config).compile(
            self.materialize(config, request),
            request,
        )

        self.assertEqual(compiled.frame["site_id"].tolist(), ["A", "A", "B", "B"])
        self.assertEqual(
            compiled.frame["load__lag_2"].tolist(),
            [2.0, 3.0, 1002.0, 1003.0],
        )
        self.assertEqual(
            compiled.frame["temperature"].tolist(),
            [40.0, 50.0, 1040.0, 1050.0],
        )
        self.assertEqual(
            compiled.frame["capacity"].tolist(),
            [500.0, 500.0, 1500.0, 1500.0],
        )

    def test_migrated_datetime_and_advanced_transformations_compile_from_visible_history(self):
        self.write_fixture()
        config = self.build_config(
            datetime_features=(
                "hour",
                "year",
                "days_in_month",
                "month",
                "quarter",
                "is_month_start",
                "is_month_end",
                "is_quarter_start",
                "is_quarter_end",
            ),
            transformations={
                "direct": {
                    "layout": "independent_models",
                    "use_horizon_exogenous": True,
                    "align_to_target": False,
                    "horizon_feature": {
                        "name": "forecast_horizon_idx",
                        "cyclical": True,
                    },
                },
                "advanced": {
                    "rolling": {
                        "columns": ["load"],
                        "windows": [2],
                        "stats": ["mean"],
                    },
                    "expanding": {"columns": ["load"], "stats": ["max"]},
                    "difference": {"columns": ["load"], "periods": [1]},
                    "percent_change": {"columns": ["load"], "periods": [1]},
                    "time_since": {"columns": ["load"], "events": ["peak"]},
                    "cyclical": {"columns": ["hour"], "period": 24},
                    "interaction": {
                        "column_pairs": [["load__lag_2", "humidity__lag_2"]],
                        "operations": ["add", "multiply"],
                    },
                    "polynomial": {"columns": ["load__lag_2"], "degree": 2},
                },
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
                "datetime_categorical": ["hour", "month"],
            },
        )
        request = self.request()

        compiled = FeatureCompiler(config).compile(
            self.materialize(config, request),
            request,
        )

        expected = {
            "dt_year",
            "dt_days_in_month",
            "dt_is_month_start",
            "dt_is_month_end",
            "dt_is_quarter_start",
            "dt_is_quarter_end",
            "load_rolling_mean_2",
            "load_expanding_max",
            "load_diff_1",
            "load_pct_change_1",
            "load_time_since_peak",
            "dt_hour_sin",
            "dt_hour_cos",
            "load__lag_2_add_humidity__lag_2",
            "load__lag_2_multiply_humidity__lag_2",
            "load__lag_2_pow_2",
        }
        self.assertTrue(expected.issubset(compiled.frame.columns))
        self.assertEqual(compiled.frame["load_rolling_mean_2"].tolist(), [2.5, 2.5])
        self.assertEqual(compiled.frame["load_diff_1"].tolist(), [1.0, 1.0])
        self.assertEqual(compiled.schema.categorical_names, ("dt_hour", "dt_month"))

    def test_runtime_transformations_are_schema_validated_but_not_compiled(self):
        self.write_fixture()
        valid = self.build_config(
            transformations={
                "feature_scaling": {"method": "standard"},
                "target": {
                    "calendar_normalization": {"method": "none"},
                    "decomposition": {"method": "linear"},
                    "scaling": {"method": "robust"},
                },
            }
        )
        self.assertIsInstance(FeatureCompiler(valid), FeatureCompiler)

        invalid = self.build_config(
            transformations={
                "target": {
                    "calendar_normalization": {"method": "ratio"},
                    "decomposition": {"method": "none"},
                    "scaling": {"method": "none"},
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "calendar normalization method"):
            FeatureCompiler(invalid)

    def test_legacy_datetime_aliases_are_rejected_for_canonical_configs(self):
        self.write_fixture()
        canonical = self.build_config(datetime_features=("weekday", "week"))
        with self.assertRaisesRegex(ValueError, "unsupported datetime features"):
            FeatureCompiler(canonical)


if __name__ == "__main__":
    unittest.main()
