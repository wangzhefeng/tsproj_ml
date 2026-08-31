# -*- coding: utf-8 -*-
"""Canonical source registry, information-set, and provider contract tests."""

import ast
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd

from data_loading import (
    AuxiliaryProvider,
    InformationSetRequest,
    MaterializedInformationSet,
    PersistenceProvider,
    ProvidedScenarioProvider,
    SourceRegistry,
    create_endogenous_future_provider,
)
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


def target_source(path="target.csv", *, columns=("load",), series=False, availability="source_time"):
    declared = []
    series_id_cols = ()
    if series:
        declared.append(ColumnSpec("site_id", "key", categorical=True))
        series_id_cols = ("site_id",)
    declared.extend(ColumnSpec(name, "target") for name in columns)
    kwargs = {}
    if availability == "column":
        kwargs["available_at_col"] = "issued_at"
    return DataSourceSpec(
        name="target",
        source_type="file",
        columns=tuple(declared),
        history_path=path,
        time_col="ts",
        series_id_cols=series_id_cols,
        availability=availability,
        **kwargs,
    )


def known_future_source(
    *,
    name="weather",
    history_path=None,
    future_path="weather_future.csv",
    backtest_path="weather_backtest.csv",
    columns=(ColumnSpec("temperature", "known_future"),),
    series=False,
):
    declared = []
    series_id_cols = ()
    if series:
        declared.append(ColumnSpec("site_id", "key", categorical=True))
        series_id_cols = ("site_id",)
    declared.extend(columns)
    return DataSourceSpec(
        name=name,
        source_type="file",
        columns=tuple(declared),
        history_path=history_path,
        backtest_path=backtest_path,
        future_path=future_path,
        time_col="ts",
        series_id_cols=series_id_cols,
        availability="column",
        available_at_col="issued_at",
    )


class InformationSetRequestTest(unittest.TestCase):
    def test_request_is_immutable_and_exposes_n_h_identity(self):
        times = pd.date_range("2026-08-02", periods=3, freq="1h")
        series_ids = ["A", "B"]
        request = InformationSetRequest(
            forecast_origin="2026-08-01 23:00",
            forecast_times=times,
            series_ids=series_ids,
            information_mode="forecast",
        )

        times.values[0] = pd.Timestamp("2030-01-01").value
        series_ids.append("C")

        self.assertEqual(request.forecast_origin, pd.Timestamp("2026-08-01 23:00"))
        self.assertEqual(request.forecast_times[0], pd.Timestamp("2026-08-02 00:00"))
        self.assertEqual(request.series_ids, ("A", "B"))
        self.assertEqual(request.N, 2)
        self.assertEqual(request.H, 3)
        with self.assertRaises(FrozenInstanceError):
            request.information_mode = "oracle"

    def test_local_request_has_one_series_identity(self):
        request = InformationSetRequest(
            forecast_origin="2026-08-01",
            forecast_times=pd.date_range("2026-08-02", periods=2, freq="1D"),
            series_ids=(),
            information_mode="nowcast",
        )
        self.assertEqual(request.series_ids, ())
        self.assertEqual((request.N, request.H), (1, 2))

    def test_multicolumn_series_identities_are_canonical_immutable_tuples(self):
        series_ids = [["north", "A"], ["south", "B"]]
        request = InformationSetRequest(
            forecast_origin="2026-08-01",
            forecast_times=pd.date_range("2026-08-02", periods=2, freq="1D"),
            series_ids=series_ids,
            information_mode="forecast",
        )
        series_ids[0][0] = "mutated"
        self.assertEqual(request.series_ids, (("north", "A"), ("south", "B")))

    def test_request_rejects_invalid_times_series_and_mode(self):
        valid = {
            "forecast_origin": "2026-08-01",
            "forecast_times": pd.date_range("2026-08-02", periods=2, freq="1D"),
            "series_ids": (),
            "information_mode": "forecast",
        }
        invalid_times = (
            pd.DatetimeIndex([]),
            pd.DatetimeIndex(["2026-08-02", pd.NaT]),
            pd.DatetimeIndex(["2026-08-03", "2026-08-02"]),
            pd.DatetimeIndex(["2026-08-02", "2026-08-02"]),
        )
        for forecast_times in invalid_times:
            with self.subTest(forecast_times=forecast_times):
                with self.assertRaises((TypeError, ValueError)):
                    InformationSetRequest(**{**valid, "forecast_times": forecast_times})

        for forecast_origin in (pd.NaT, None, "not-a-time"):
            with self.subTest(forecast_origin=forecast_origin):
                with self.assertRaises((TypeError, ValueError)):
                    InformationSetRequest(**{**valid, "forecast_origin": forecast_origin})

        for series_ids in (("A", "A"), (None,), (["A"], ["A"])):
            with self.subTest(series_ids=series_ids):
                with self.assertRaises((TypeError, ValueError)):
                    InformationSetRequest(**{**valid, "series_ids": series_ids})

        with self.assertRaises(ValueError):
            InformationSetRequest(**{**valid, "information_mode": "realtime"})


class SourceRegistryTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_csv(self, name, rows):
        pd.DataFrame(rows).to_csv(self.base_dir / name, index=False)

    def request(self, *, series_ids=(), mode="forecast", origin="2026-08-01 23:00", periods=2):
        return InformationSetRequest(
            forecast_origin=origin,
            forecast_times=pd.date_range("2026-08-02 00:00", periods=periods, freq="1h"),
            series_ids=series_ids,
            information_mode=mode,
        )

    def test_local_multitarget_materialization_and_source_qualified_lineage(self):
        self.write_csv(
            "target.csv",
            [
                {"ts": "2026-08-01 21:00", "load": 10.0, "power": 20.0},
                {"ts": "2026-08-01 22:00", "load": 11.0, "power": 21.0},
                {"ts": "2026-08-02 00:00", "load": 999.0, "power": 999.0},
            ],
        )
        self.write_csv(
            "weather_actual.csv",
            [
                {"ts": "2026-08-02 00:00", "issued_at": "2026-08-01 18:00", "temperature": 24.0},
                {"ts": "2026-08-02 01:00", "issued_at": "2026-08-01 18:00", "temperature": 25.0},
            ],
        )
        self.write_csv(
            "weather_backtest.csv",
            [
                {"ts": "2026-08-02 00:00", "issued_at": "2026-08-01 20:00", "temperature": 25.0},
                {"ts": "2026-08-02 01:00", "issued_at": "2026-08-01 20:00", "temperature": 26.0},
            ],
        )
        self.write_csv(
            "weather_future.csv",
            [
                {"ts": "2026-08-02 00:00", "issued_at": "2026-08-01 22:00", "temperature": 27.0},
                {"ts": "2026-08-02 01:00", "issued_at": "2026-08-01 22:00", "temperature": 28.0},
            ],
        )
        spec = DataSpec(
            (
                target_source(columns=("load", "power")),
                known_future_source(history_path="weather_actual.csv"),
            )
        )

        result = SourceRegistry(spec, self.base_dir).materialize(self.request())

        self.assertEqual(set(result.target_history), {"target"})
        self.assertEqual(set(result.known_future), {"weather"})
        self.assertEqual(result.target_history["target"]["load"].tolist(), [10.0, 11.0])
        self.assertEqual(result.known_future["weather"]["temperature"].tolist(), [27.0, 28.0])
        self.assertEqual(
            {(item.source_name, item.path_version) for item in result.lineage},
            {
                ("target", "history"),
                ("weather", "history"),
                ("weather", "backtest"),
                ("weather", "future"),
            },
        )
        self.assertTrue(all(item.availability_policy in {"source_time", "column"} for item in result.lineage))
        self.assertTrue(all(not item.oracle for item in result.lineage))

    def test_known_future_history_derives_availability_and_allows_optional_ignored_metadata(self):
        self.write_csv(
            "target.csv",
            [
                {"ts": "2026-08-01 21:00", "load": 10.0},
                {"ts": "2026-08-01 22:00", "load": 11.0},
            ],
        )
        self.write_csv(
            "weather_actual.csv",
            [
                {
                    "ts": "2026-07-31 23:00",
                    "temperature": 24.0,
                    "diagnostic": 1.0,
                }
            ],
        )
        self.write_csv(
            "weather_backtest.csv",
            [
                {
                    "ts": "2026-08-02 00:00",
                    "issued_at": "2026-08-01 20:00",
                    "temperature": 25.0,
                    "diagnostic": 2.0,
                    "source_label": "proxy",
                },
                {
                    "ts": "2026-08-02 01:00",
                    "issued_at": "2026-08-01 20:00",
                    "temperature": 26.0,
                    "diagnostic": 3.0,
                    "source_label": "proxy",
                },
            ],
        )
        self.write_csv(
            "weather_future.csv",
            [
                {
                    "ts": "2026-08-02 00:00",
                    "issued_at": "2026-08-01 22:00",
                    "temperature": 27.0,
                    "diagnostic": 4.0,
                    "source_label": "forecast",
                },
                {
                    "ts": "2026-08-02 01:00",
                    "issued_at": "2026-08-01 22:00",
                    "temperature": 28.0,
                    "diagnostic": 5.0,
                    "source_label": "forecast",
                },
            ],
        )
        spec = DataSpec(
            (
                target_source(),
                known_future_source(
                    history_path="weather_actual.csv",
                    columns=(
                        ColumnSpec("temperature", "known_future"),
                        ColumnSpec("diagnostic", "ignored"),
                        ColumnSpec("source_label", "ignored", categorical=True),
                    ),
                ),
            )
        )

        result = SourceRegistry(spec, self.base_dir).materialize(self.request())

        self.assertEqual(
            result.known_future["weather"]["temperature"].tolist(),
            [27.0, 28.0],
        )

    def test_file_source_projects_declared_columns_and_drops_physical_extras(self):
        self.write_csv(
            "target.csv",
            [
                {
                    "ts": "2026-08-01 21:00",
                    "load": 10.0,
                    "raw_metadata": "not-a-model-input",
                },
                {
                    "ts": "2026-08-01 22:00",
                    "load": 11.0,
                    "raw_metadata": "not-a-model-input",
                },
            ],
        )

        result = SourceRegistry(
            DataSpec((target_source(),)),
            self.base_dir,
        ).materialize(self.request())

        self.assertEqual(
            result.target_history["target"].columns.tolist(),
            ["ts", "load"],
        )

    def test_global_n2_materializes_target_observed_known_future_and_static(self):
        target_rows = []
        observed_rows = []
        future_rows = []
        static_rows = []
        for site_id, offset in (("A", 0), ("B", 100)):
            target_rows.extend(
                [
                    {"site_id": site_id, "ts": "2026-08-01 21:00", "load": 10.0 + offset},
                    {"site_id": site_id, "ts": "2026-08-01 22:00", "load": 11.0 + offset},
                ]
            )
            observed_rows.append(
                {"site_id": site_id, "ts": "2026-08-01 22:00", "humidity": 40.0 + offset}
            )
            future_rows.extend(
                [
                    {"site_id": site_id, "ts": "2026-08-02 00:00", "issued_at": "2026-08-01 20:00", "temperature": 25.0 + offset},
                    {"site_id": site_id, "ts": "2026-08-02 01:00", "issued_at": "2026-08-01 20:00", "temperature": 26.0 + offset},
                ]
            )
            static_rows.append({"site_id": site_id, "region": f"R-{site_id}", "capacity": 50.0 + offset})
        self.write_csv("target.csv", target_rows)
        self.write_csv("observed.csv", observed_rows)
        self.write_csv("weather_future.csv", future_rows)
        self.write_csv("static.csv", static_rows)
        self.write_csv(
            "weather_backtest.csv",
            [
                {
                    **row,
                    "issued_at": "2026-08-01 18:00",
                    "temperature": row["temperature"] - 1.0,
                }
                for row in future_rows
            ],
        )
        observed = DataSourceSpec(
            name="meter",
            source_type="file",
            columns=(
                ColumnSpec("site_id", "key", categorical=True),
                ColumnSpec("humidity", "observed_past"),
            ),
            history_path="observed.csv",
            time_col="ts",
            series_id_cols=("site_id",),
            availability="source_time",
            provider="persistence",
        )
        static = DataSourceSpec(
            name="sites",
            source_type="file",
            columns=(
                ColumnSpec("site_id", "key", categorical=True),
                ColumnSpec("region", "static", categorical=True),
                ColumnSpec("capacity", "static"),
            ),
            history_path="static.csv",
            series_id_cols=("site_id",),
        )
        spec = DataSpec(
            (
                target_source(series=True),
                observed,
                known_future_source(series=True),
                static,
            )
        )

        result = SourceRegistry(spec, self.base_dir).materialize(
            self.request(series_ids=("A", "B"))
        )

        self.assertEqual(len(result.target_history["target"]), 4)
        self.assertEqual(len(result.observed_past["meter"]), 2)
        self.assertEqual(len(result.known_future["weather"]), 4)
        self.assertEqual(len(result.static["sites"]), 2)
        self.assertEqual(result.known_future["weather"]["site_id"].tolist(), ["A", "A", "B", "B"])

    def test_column_availability_selects_latest_as_of_vintage(self):
        self.write_csv("target.csv", [{"ts": "2026-08-01 22:00", "load": 1.0}])
        rows = [
            {"ts": "2026-08-02 00:00", "issued_at": "2026-08-01 18:00", "temperature": 20.0},
            {"ts": "2026-08-02 00:00", "issued_at": "2026-08-01 22:00", "temperature": 21.0},
            {"ts": "2026-08-02 00:00", "issued_at": "2026-08-02 00:00", "temperature": 99.0},
            {"ts": "2026-08-02 01:00", "issued_at": "2026-08-01 22:00", "temperature": 22.0},
        ]
        self.write_csv("weather_backtest.csv", rows[:1])
        self.write_csv("weather_future.csv", rows[1:])
        spec = DataSpec((target_source(), known_future_source()))

        result = SourceRegistry(spec, self.base_dir).materialize(self.request())

        self.assertEqual(result.known_future["weather"]["temperature"].tolist(), [21.0, 22.0])

    def test_generator_defined_uses_reserved_available_at_and_exact_grid(self):
        self.write_csv("target.csv", [{"ts": "2026-08-01 22:00", "load": 1.0}])
        generated = DataSourceSpec(
            name="calendar",
            source_type="generated",
            columns=(ColumnSpec("holiday", "known_future", categorical=True),),
            time_col="ts",
            availability="generator_defined",
            generator="calendar_generator",
        )
        calls = []

        def calendar_generator(source, request):
            calls.append((source, request))
            return pd.DataFrame(
                {
                    "ts": request.forecast_times,
                    "available_at": [request.forecast_origin] * request.H,
                    "holiday": ["no", "yes"],
                }
            )

        result = SourceRegistry(
            DataSpec((target_source(), generated)),
            self.base_dir,
            generators={"calendar_generator": calendar_generator},
        ).materialize(self.request())

        self.assertEqual(result.known_future["calendar"]["holiday"].tolist(), ["no", "yes"])
        self.assertEqual(calls[0][0], generated)
        self.assertEqual(calls[0][1].H, 2)
        self.assertEqual(result.lineage[-1].path_version, "generated")

    def test_forecast_nowcast_block_target_future_and_oracle_is_explicit(self):
        self.write_csv(
            "target.csv",
            [
                {"ts": "2026-08-01 22:00", "load": 1.0},
                {"ts": "2026-08-02 00:00", "load": 2.0},
                {"ts": "2026-08-02 01:00", "load": 3.0},
                {"ts": "2026-08-02 02:00", "load": 4.0},
            ],
        )
        spec = DataSpec((target_source(),))
        registry = SourceRegistry(spec, self.base_dir)

        for mode in ("forecast", "nowcast"):
            with self.subTest(mode=mode):
                result = registry.materialize(self.request(mode=mode))
                self.assertEqual(result.target_history["target"]["load"].tolist(), [1.0])
                self.assertTrue(all(not item.oracle for item in result.lineage))

        oracle = registry.materialize(self.request(mode="oracle"))
        self.assertEqual(oracle.target_history["target"]["load"].tolist(), [1.0, 2.0, 3.0])
        self.assertTrue(all(item.oracle for item in oracle.lineage))

    def test_oracle_explicitly_allows_column_available_target_future(self):
        self.write_csv(
            "target.csv",
            [
                {"ts": "2026-08-01 22:00", "issued_at": "2026-08-01 22:00", "load": 1.0},
                {"ts": "2026-08-02 00:00", "issued_at": "2026-08-02 00:00", "load": 2.0},
                {"ts": "2026-08-02 01:00", "issued_at": "2026-08-02 01:00", "load": 3.0},
            ],
        )
        registry = SourceRegistry(
            DataSpec((target_source(availability="column"),)),
            self.base_dir,
        )

        forecast = registry.materialize(self.request(mode="forecast"))
        oracle = registry.materialize(self.request(mode="oracle"))

        self.assertEqual(forecast.target_history["target"]["load"].tolist(), [1.0])
        self.assertEqual(oracle.target_history["target"]["load"].tolist(), [1.0, 2.0, 3.0])

    def test_reader_cache_and_copy_isolation(self):
        self.write_csv("target.csv", [{"ts": "2026-08-01 22:00", "load": 1.0}])
        calls = []

        def reader(path):
            calls.append(Path(path))
            return pd.read_csv(path)

        registry = SourceRegistry(DataSpec((target_source(),)), self.base_dir, reader=reader)
        first = registry.materialize(self.request())
        first_frame = first.target_history["target"]
        first_frame.loc[:, "load"] = 999.0
        second = registry.materialize(self.request())

        self.assertEqual(len(calls), 1)
        self.assertEqual(second.target_history["target"]["load"].tolist(), [1.0])
        isolated = second.target_history
        isolated["target"].loc[:, "load"] = 888.0
        self.assertEqual(second.target_history["target"]["load"].tolist(), [1.0])

    def test_validation_rejects_missing_duplicates_nan_and_nonfinite(self):
        valid_target = [{"ts": "2026-08-01 22:00", "load": 1.0}]
        cases = {
            "missing": [{"ts": "2026-08-01 22:00"}],
            "duplicate": valid_target * 2,
            "nan": [{"ts": "2026-08-01 22:00", "load": np.nan}],
            "infinite": [{"ts": "2026-08-01 22:00", "load": np.inf}],
        }
        for name, rows in cases.items():
            with self.subTest(name=name):
                self.write_csv("target.csv", rows)
                with self.assertRaises(ValueError):
                    SourceRegistry(DataSpec((target_source(),)), self.base_dir).materialize(self.request())

    def test_validation_rejects_bad_available_at_and_unresolved_vintage_duplicate(self):
        self.write_csv("target.csv", [{"ts": "2026-08-01 22:00", "load": 1.0}])
        invalid_cases = (
            [
                {"ts": "2026-08-02 00:00", "issued_at": "bad", "temperature": 20.0},
                {"ts": "2026-08-02 01:00", "issued_at": "2026-08-01", "temperature": 21.0},
            ],
            [
                {"ts": "2026-08-02 00:00", "issued_at": "2026-08-01", "temperature": 20.0},
                {"ts": "2026-08-02 00:00", "issued_at": "2026-08-01", "temperature": 21.0},
                {"ts": "2026-08-02 01:00", "issued_at": "2026-08-01", "temperature": 22.0},
            ],
        )
        for index, rows in enumerate(invalid_cases):
            with self.subTest(index=index):
                self.write_csv("weather_backtest.csv", rows)
                self.write_csv("weather_future.csv", rows)
                with self.assertRaises(ValueError):
                    SourceRegistry(
                        DataSpec((target_source(), known_future_source())),
                        self.base_dir,
                    ).materialize(self.request())

    def test_known_future_requires_exact_coverage(self):
        self.write_csv("target.csv", [{"ts": "2026-08-01 22:00", "load": 1.0}])
        cases = {
            "missing": [
                {"ts": "2026-08-02 00:00", "issued_at": "2026-08-01", "temperature": 20.0}
            ],
            "extra_series": [
                {"site_id": "A", "ts": "2026-08-02 00:00", "issued_at": "2026-08-01", "temperature": 20.0},
                {"site_id": "A", "ts": "2026-08-02 01:00", "issued_at": "2026-08-01", "temperature": 21.0},
                {"site_id": "B", "ts": "2026-08-02 00:00", "issued_at": "2026-08-01", "temperature": 22.0},
                {"site_id": "B", "ts": "2026-08-02 01:00", "issued_at": "2026-08-01", "temperature": 23.0},
                {"site_id": "C", "ts": "2026-08-02 00:00", "issued_at": "2026-08-01", "temperature": 24.0},
            ],
        }
        self.write_csv("weather_backtest.csv", cases["missing"])
        self.write_csv("weather_future.csv", cases["missing"])
        with self.assertRaises(ValueError):
            SourceRegistry(
                DataSpec((target_source(), known_future_source())),
                self.base_dir,
            ).materialize(self.request())

        self.write_csv("target.csv", [
            {"site_id": "A", "ts": "2026-08-01 22:00", "load": 1.0},
            {"site_id": "B", "ts": "2026-08-01 22:00", "load": 2.0},
        ])
        self.write_csv("weather_backtest.csv", cases["extra_series"])
        self.write_csv("weather_future.csv", cases["extra_series"])
        with self.assertRaises(ValueError):
            SourceRegistry(
                DataSpec((target_source(series=True), known_future_source(series=True))),
                self.base_dir,
            ).materialize(self.request(series_ids=("A", "B")))

    def test_histories_require_each_series_and_static_is_exactly_one_per_series(self):
        self.write_csv("target.csv", [{"site_id": "A", "ts": "2026-08-01 22:00", "load": 1.0}])
        spec = DataSpec((target_source(series=True),))
        with self.assertRaises(ValueError):
            SourceRegistry(spec, self.base_dir).materialize(self.request(series_ids=("A", "B")))

        self.write_csv("target.csv", [
            {"site_id": "A", "ts": "2026-08-01 22:00", "load": 1.0},
            {"site_id": "B", "ts": "2026-08-01 22:00", "load": 2.0},
        ])
        self.write_csv("static.csv", [
            {"site_id": "A", "region": "north"},
            {"site_id": "A", "region": "north-copy"},
            {"site_id": "B", "region": "south"},
        ])
        static = DataSourceSpec(
            name="sites",
            source_type="file",
            columns=(
                ColumnSpec("site_id", "key", categorical=True),
                ColumnSpec("region", "static", categorical=True),
            ),
            history_path="static.csv",
            series_id_cols=("site_id",),
        )
        with self.assertRaises(ValueError):
            SourceRegistry(
                DataSpec((target_source(series=True), static)),
                self.base_dir,
            ).materialize(self.request(series_ids=("A", "B")))

    def test_generated_source_requires_dataframe_and_reserved_available_at(self):
        self.write_csv("target.csv", [{"ts": "2026-08-01 22:00", "load": 1.0}])
        generated = DataSourceSpec(
            name="calendar",
            source_type="generated",
            columns=(ColumnSpec("holiday", "known_future", categorical=True),),
            time_col="ts",
            availability="generator_defined",
            generator="calendar_generator",
        )
        invalid_outputs = (
            [1, 2],
            pd.DataFrame({"ts": pd.date_range("2026-08-02", periods=2, freq="1h"), "holiday": [0, 1]}),
            pd.DataFrame({
                "ts": pd.date_range("2026-08-02", periods=2, freq="1h"),
                "available_at": ["2026-08-01", "2026-08-02 01:00"],
                "holiday": [0, 1],
            }),
        )
        for output in invalid_outputs:
            with self.subTest(output_type=type(output).__name__):
                with self.assertRaises((TypeError, ValueError)):
                    SourceRegistry(
                        DataSpec((target_source(), generated)),
                        self.base_dir,
                        generators={"calendar_generator": lambda source, request, value=output: value},
                    ).materialize(self.request())

    def test_registry_rejects_missing_reader_generator_and_series_shape_mismatch(self):
        with self.assertRaises(TypeError):
            SourceRegistry(DataSpec((target_source(),)), self.base_dir, reader="read")

        generated = DataSourceSpec(
            name="calendar",
            source_type="generated",
            columns=(ColumnSpec("holiday", "known_future"),),
            time_col="ts",
            availability="generator_defined",
            generator="missing",
        )
        self.write_csv("target.csv", [{"ts": "2026-08-01 22:00", "load": 1.0}])
        with self.assertRaises(ValueError):
            SourceRegistry(DataSpec((target_source(), generated)), self.base_dir).materialize(self.request())

        self.write_csv("target.csv", [{"site_id": "A", "ts": "2026-08-01 22:00", "load": 1.0}])
        with self.assertRaises(ValueError):
            SourceRegistry(DataSpec((target_source(series=True),)), self.base_dir).materialize(self.request())

    def test_observed_scenario_and_auxiliary_providers_materialize_explicit_backtest_trajectories(self):
        self.write_csv(
            "target.csv",
            [
                {"ts": "2026-08-01 21:00", "load": 1.0},
                {"ts": "2026-08-01 22:00", "load": 2.0},
            ],
        )
        self.write_csv(
            "observed.csv",
            [
                {
                    "ts": "2026-08-01 21:00",
                    "issued_at": "2026-08-01 21:00",
                    "humidity": 40.0,
                },
                {
                    "ts": "2026-08-01 22:00",
                    "issued_at": "2026-08-01 22:00",
                    "humidity": 41.0,
                },
            ],
        )
        self.write_csv(
            "observed_backtest.csv",
            [
                {
                    "ts": "2026-08-02 00:00",
                    "issued_at": "2026-08-01 21:00",
                    "humidity": 49.0,
                },
                {
                    "ts": "2026-08-02 00:00",
                    "issued_at": "2026-08-01 22:00",
                    "humidity": 50.0,
                },
                {
                    "ts": "2026-08-02 00:00",
                    "issued_at": "2026-08-02 00:00",
                    "humidity": 500.0,
                },
                {
                    "ts": "2026-08-02 01:00",
                    "issued_at": "2026-08-01 22:00",
                    "humidity": 51.0,
                },
            ],
        )

        for method in ("provided_scenario", "auxiliary"):
            with self.subTest(method=method):
                observed = DataSourceSpec(
                    name="observed",
                    source_type="file",
                    columns=(ColumnSpec("humidity", "observed_past"),),
                    history_path="observed.csv",
                    backtest_path="observed_backtest.csv",
                    time_col="ts",
                    availability="column",
                    available_at_col="issued_at",
                    provider=method,
                )
                result = SourceRegistry(
                    DataSpec((target_source(), observed)),
                    self.base_dir,
                ).materialize(self.request())
                provider = result.observed_future_providers[()]
                self.assertEqual(
                    [provider.value_at("humidity", step) for step in range(2)],
                    [50.0, 51.0],
                )
                self.assertEqual(provider.provider_name("humidity"), method)
                self.assertLessEqual(
                    provider.available_at("humidity", 0),
                    self.request().forecast_origin,
                )

    def test_source_time_observed_provider_rejects_future_backtest_trajectory(self):
        self.write_csv(
            "target.csv",
            [{"ts": "2026-08-01 22:00", "load": 1.0}],
        )
        self.write_csv(
            "observed.csv",
            [{"ts": "2026-08-01 22:00", "humidity": 41.0}],
        )
        self.write_csv(
            "observed_backtest.csv",
            [
                {"ts": "2026-08-02 00:00", "humidity": 50.0},
                {"ts": "2026-08-02 01:00", "humidity": 51.0},
            ],
        )
        observed = DataSourceSpec(
            name="observed",
            source_type="file",
            columns=(ColumnSpec("humidity", "observed_past"),),
            history_path="observed.csv",
            backtest_path="observed_backtest.csv",
            time_col="ts",
            availability="source_time",
            provider="provided_scenario",
        )

        with self.assertRaisesRegex(
            ValueError,
            "availability=source_time.*future observed provider trajectory",
        ):
            SourceRegistry(
                DataSpec((target_source(), observed)),
                self.base_dir,
            ).materialize(self.request())


class EndogenousFutureProviderTest(unittest.TestCase):
    def test_provided_scenario_and_auxiliary_require_exact_finite_h_trajectories(self):
        provided = ProvidedScenarioProvider(horizon=3, trajectories={"temperature": [1, 2, 3]})
        auxiliary = AuxiliaryProvider(horizon=3, trajectories={"humidity": np.array([4.0, 5.0, 6.0])})

        self.assertEqual([provided.value_at("temperature", step) for step in range(3)], [1.0, 2.0, 3.0])
        self.assertEqual([auxiliary.value_at("humidity", step) for step in range(3)], [4.0, 5.0, 6.0])
        for provider, name in ((provided, "temperature"), (auxiliary, "humidity")):
            with self.assertRaises(IndexError):
                provider.value_at(name, 3)
            with self.assertRaises(KeyError):
                provider.value_at("missing", 0)

        invalid = (
            {"x": [1.0, 2.0]},
            {"x": [1.0, np.nan, 3.0]},
            {"x": [1.0, np.inf, 3.0]},
        )
        for trajectories in invalid:
            with self.subTest(trajectories=trajectories):
                with self.assertRaises(ValueError):
                    ProvidedScenarioProvider(horizon=3, trajectories=trajectories)
                with self.assertRaises(ValueError):
                    AuxiliaryProvider(horizon=3, trajectories=trajectories)

    def test_persistence_is_explicit_and_uses_last_finite_history(self):
        provider = PersistenceProvider(horizon=3, history={"humidity": [1.0, np.nan, 2.5, np.nan]})
        self.assertEqual([provider.value_at("humidity", step) for step in range(3)], [2.5, 2.5, 2.5])
        with self.assertRaises(ValueError):
            PersistenceProvider(horizon=2, history={"humidity": [np.nan, np.inf]})

    def test_factory_requires_explicit_method_and_valid_payload(self):
        provided = create_endogenous_future_provider(
            "provided_scenario",
            horizon=2,
            trajectories={"x": [1.0, 2.0]},
        )
        persistence = create_endogenous_future_provider(
            "persistence",
            horizon=2,
            history={"x": [0.0, 3.0]},
        )
        auxiliary = create_endogenous_future_provider(
            "auxiliary",
            horizon=2,
            trajectories={"x": [4.0, 5.0]},
        )
        self.assertEqual(
            [provider.value_at("x", 1) for provider in (provided, persistence, auxiliary)],
            [2.0, 3.0, 5.0],
        )
        with self.assertRaises(TypeError):
            create_endogenous_future_provider(horizon=2, trajectories={"x": [1.0, 2.0]})
        with self.assertRaises(ValueError):
            create_endogenous_future_provider("default", horizon=2, trajectories={"x": [1.0, 2.0]})
        with self.assertRaises(ValueError):
            create_endogenous_future_provider("persistence", horizon=2, trajectories={"x": [1.0, 2.0]})


class ImportBoundaryTest(unittest.TestCase):
    def test_new_data_core_has_no_legacy_runtime_imports(self):
        root = Path(__file__).parents[1] / "data_loading"
        forbidden_symbols = {"DataLoader", "pred_method"}
        for path in sorted(root.glob("*.py")):
            with self.subTest(path=path.name):
                module = ast.parse(path.read_text(encoding="utf-8"))
                imported_modules = set()
                imported_symbols = set()
                for node in ast.walk(module):
                    if isinstance(node, ast.ImportFrom):
                        imported_modules.add(node.module or "")
                        imported_symbols.update(alias.asname or alias.name for alias in node.names)
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            imported_modules.add(alias.name)
                            imported_symbols.add(alias.asname or alias.name.rsplit(".", 1)[-1])
                self.assertFalse(
                    any(name == "models" or name.startswith("models.") for name in imported_modules)
                )
                self.assertFalse(
                    any(name == "data_provider" or name.startswith("data_provider.") for name in imported_modules)
                )
                self.assertTrue(forbidden_symbols.isdisjoint(imported_symbols))


if __name__ == "__main__":
    unittest.main()
