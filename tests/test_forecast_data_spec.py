# -*- coding: utf-8 -*-
"""显式数据源与列角色规格的独立契约测试。"""
import ast
import json
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import model_forecasting.specs as specs
from model_forecasting.specs import (
    AvailabilityPolicy,
    ColumnRole,
    ColumnSpec,
    DataSourceSpec,
    DataSpec,
)


class ForecastDataSpecValidTest(unittest.TestCase):
    def make_registry(self):
        return DataSpec(
            sources=[
                DataSourceSpec(
                    name="target_history",
                    source_type="file",
                    columns=[
                        ColumnSpec("site_id", ColumnRole.KEY, categorical=True),
                        ColumnSpec("load", ColumnRole.TARGET),
                    ],
                    history_path="dataset/load.csv",
                    time_col="ts",
                    series_id_cols=["site_id"],
                    availability="source_time",
                ),
                DataSourceSpec(
                    name="observed_weather",
                    source_type="file",
                    columns=[
                        ColumnSpec("site_id", ColumnRole.KEY, categorical=True),
                        ColumnSpec("temperature_actual", ColumnRole.OBSERVED_PAST),
                        ColumnSpec("quality_flag", ColumnRole.IGNORED),
                    ],
                    history_path="dataset/weather_actual.csv",
                    backtest_path="dataset/weather_backtest.csv",
                    time_col="ts",
                    series_id_cols=["site_id"],
                    availability=AvailabilityPolicy.SOURCE_TIME,
                    provider="provided_scenario",
                ),
                DataSourceSpec(
                    name="weather_forecast",
                    source_type="file",
                    columns=[
                        ColumnSpec("site_id", ColumnRole.KEY, categorical=True),
                        ColumnSpec("temperature_forecast", ColumnRole.KNOWN_FUTURE),
                    ],
                    history_path="dataset/weather_forecast_history.csv",
                    backtest_path="dataset/weather_forecast_backtest.csv",
                    future_path="dataset/weather_forecast_future.csv",
                    time_col="ts",
                    series_id_cols=["site_id"],
                    availability="column",
                    available_at_col="issued_at",
                ),
                DataSourceSpec(
                    name="site_metadata",
                    source_type="file",
                    columns=[
                        ColumnSpec("site_id", ColumnRole.KEY, categorical=True),
                        ColumnSpec("region", ColumnRole.STATIC, categorical=True),
                        ColumnSpec("capacity_mw", ColumnRole.STATIC),
                    ],
                    history_path="dataset/sites.csv",
                    series_id_cols=["site_id"],
                ),
                DataSourceSpec(
                    name="calendar",
                    source_type="generated",
                    columns=[
                        ColumnSpec("site_id", ColumnRole.KEY, categorical=True),
                        ColumnSpec("holiday", ColumnRole.KNOWN_FUTURE, categorical=True),
                        ColumnSpec("generator_note", ColumnRole.IGNORED),
                    ],
                    series_id_cols=["site_id"],
                    generator="calendar_features",
                    availability="generator_defined",
                ),
            ]
        )

    def test_column_role_has_exact_public_values(self):
        self.assertEqual(
            [role.value for role in ColumnRole],
            ["target", "observed_past", "known_future", "static", "key", "ignored"],
        )

    def test_availability_policy_has_exact_public_values(self):
        self.assertEqual(
            [policy.value for policy in AvailabilityPolicy],
            ["source_time", "column", "generator_defined", "forecast_origin"],
        )

    def test_valid_registry_covers_all_roles(self):
        data_spec = self.make_registry()

        self.assertEqual(data_spec.target_columns, ("load",))
        self.assertEqual(data_spec.role_columns(ColumnRole.OBSERVED_PAST), ("temperature_actual",))
        self.assertEqual(
            data_spec.role_columns("known_future"),
            ("temperature_forecast", "holiday"),
        )
        self.assertEqual(data_spec.role_columns(ColumnRole.STATIC), ("region", "capacity_mw"))
        self.assertEqual(data_spec.role_columns(ColumnRole.KEY), ("site_id",))
        self.assertEqual(data_spec.role_columns(ColumnRole.IGNORED), ("quality_flag", "generator_note"))

    def test_lists_are_copied_to_immutable_tuples(self):
        columns = [ColumnSpec("load", "target")]
        series_ids = []
        source = DataSourceSpec(
            name="target_history",
            source_type="file",
            columns=columns,
            history_path="dataset/load.csv",
            time_col="ts",
            series_id_cols=series_ids,
            availability="source_time",
        )
        sources = [source]
        data_spec = DataSpec(sources=sources)

        columns.append(ColumnSpec("other", "ignored"))
        series_ids.append("site_id")
        sources.append(source)

        self.assertEqual(source.columns, (ColumnSpec("load", ColumnRole.TARGET),))
        self.assertEqual(source.series_id_cols, ())
        self.assertEqual(data_spec.sources, (source,))
        self.assertIsInstance(source.columns, tuple)
        self.assertIsInstance(source.series_id_cols, tuple)
        self.assertIsInstance(data_spec.sources, tuple)

    def test_specs_are_immutable(self):
        column = ColumnSpec("load", ColumnRole.TARGET)
        source = DataSourceSpec(
            name="target_history",
            source_type="file",
            columns=(column,),
            history_path="dataset/load.csv",
            time_col="ts",
            availability="source_time",
        )
        data_spec = DataSpec((source,))

        with self.assertRaises(FrozenInstanceError):
            column.name = "power"
        with self.assertRaises(FrozenInstanceError):
            source.name = "other"
        with self.assertRaises(FrozenInstanceError):
            data_spec.sources = ()

    def test_canonical_payload_is_json_friendly_deterministic_and_isolated(self):
        data_spec = self.make_registry()

        first = data_spec.canonical_payload()
        second = data_spec.canonical_payload()

        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertEqual(json.loads(json.dumps(first)), first)
        self.assertEqual(first["sources"][0]["columns"][1], {
            "name": "load",
            "role": "target",
            "categorical": False,
        })
        self.assertEqual(first["sources"][4]["generator"], "calendar_features")
        self.assertEqual(first["sources"][0]["availability"], "source_time")
        self.assertEqual(first["sources"][2]["availability"], "column")
        self.assertEqual(first["sources"][4]["availability"], "generator_defined")
        self.assertEqual(first["sources"][1]["provider"], "provided_scenario")
        self.assertNotIn("available_at_col", first["sources"][0])
        self.assertEqual(first["sources"][2]["available_at_col"], "issued_at")
        self.assertNotIn("availability", first["sources"][3])
        self.assertNotIn("history_path", first["sources"][4])
        self.assertNotIn("generator", first["sources"][0])

        first["sources"][0]["columns"][1]["name"] = "mutated"
        self.assertEqual(data_spec.target_columns, ("load",))
        self.assertEqual(data_spec.canonical_payload(), second)

    def test_public_exports_preserve_problem_spec(self):
        self.assertIs(specs.AvailabilityPolicy, AvailabilityPolicy)
        self.assertIs(specs.ColumnRole, ColumnRole)
        self.assertIs(specs.ColumnSpec, ColumnSpec)
        self.assertIs(specs.DataSourceSpec, DataSourceSpec)
        self.assertIs(specs.DataSpec, DataSpec)
        self.assertTrue(
            {
                "ForecastProblemSpec",
                "AvailabilityPolicy",
                "ColumnRole",
                "ColumnSpec",
                "DataSourceSpec",
                "DataSpec",
            }.issubset(specs.__all__)
        )

    def test_data_module_has_no_legacy_runtime_imports(self):
        source_path = Path(__file__).parents[1] / "model_forecasting" / "specs" / "data.py"
        module = ast.parse(source_path.read_text(encoding="utf-8"))
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

        for module_name in ("models", "data_provider"):
            with self.subTest(module_name=module_name):
                self.assertFalse(
                    any(
                        imported == module_name or imported.startswith(f"{module_name}.")
                        for imported in imported_modules
                    )
                )
        self.assertTrue({"pred_method", "DataLoader"}.isdisjoint(imported_symbols))


class ColumnSpecInvalidTest(unittest.TestCase):
    def test_rejects_invalid_names(self):
        for name in ("", "  ", " load", "load ", 1, None):
            with self.subTest(name=name):
                expected = TypeError if not isinstance(name, str) else ValueError
                with self.assertRaises(expected):
                    ColumnSpec(name, ColumnRole.TARGET)

    def test_rejects_unknown_role(self):
        for role in ("past", "TARGET", 1, None):
            with self.subTest(role=role):
                with self.assertRaises((TypeError, ValueError)):
                    ColumnSpec("load", role)

    def test_rejects_non_boolean_categorical(self):
        for categorical in (0, 1, "false", None):
            with self.subTest(categorical=categorical):
                with self.assertRaises(TypeError):
                    ColumnSpec("load", ColumnRole.TARGET, categorical=categorical)

    def test_target_cannot_be_categorical(self):
        with self.assertRaises(ValueError):
            ColumnSpec("load", ColumnRole.TARGET, categorical=True)

    def test_static_and_key_columns_may_be_categorical(self):
        self.assertTrue(ColumnSpec("region", ColumnRole.STATIC, categorical=True).categorical)
        self.assertTrue(ColumnSpec("site_id", ColumnRole.KEY, categorical=True).categorical)


class DataSourceSpecInvalidTest(unittest.TestCase):
    def make_source(self, **overrides):
        values = {
            "name": "target_history",
            "source_type": "file",
            "columns": (ColumnSpec("load", ColumnRole.TARGET),),
            "history_path": "dataset/load.csv",
            "time_col": "ts",
            "availability": "source_time",
        }
        values.update(overrides)
        return DataSourceSpec(**values)

    def assert_invalid(self, **overrides):
        with self.assertRaises((TypeError, ValueError)):
            self.make_source(**overrides)

    def test_rejects_invalid_source_name_and_source_type(self):
        for name in ("", " ", " source", "source ", 1):
            with self.subTest(name=name):
                self.assert_invalid(name=name)
        for source_type in ("database", "FILE", 1, None):
            with self.subTest(source_type=source_type):
                self.assert_invalid(source_type=source_type)

    def test_rejects_empty_non_column_and_duplicate_columns(self):
        self.assert_invalid(columns=())
        self.assert_invalid(columns=(ColumnSpec("load", "target"), "temperature"))
        self.assert_invalid(columns=(ColumnSpec("load", "target"), ColumnSpec("load", "ignored")))

    def test_file_source_requires_at_least_one_path(self):
        self.assert_invalid(history_path=None, backtest_path=None, future_path=None)

    def test_generated_source_requires_generator_and_forbids_all_paths(self):
        generated = {
            "source_type": "generated",
            "columns": (ColumnSpec("holiday", "known_future", categorical=True),),
            "history_path": None,
            "time_col": None,
            "availability": "generator_defined",
        }
        self.assert_invalid(**generated, generator=None)
        for path_name in ("history_path", "backtest_path", "future_path"):
            with self.subTest(path_name=path_name):
                overrides = {**generated, path_name: "data.csv", "generator": "calendar"}
                self.assert_invalid(**overrides)

    def test_file_source_forbids_generator(self):
        self.assert_invalid(generator="calendar")

    def test_temporal_file_roles_require_time_col(self):
        for role in (ColumnRole.TARGET, ColumnRole.OBSERVED_PAST, ColumnRole.KNOWN_FUTURE):
            overrides = {
                "columns": (ColumnSpec(f"value_{role.value}", role),),
                "time_col": None,
            }
            if role is ColumnRole.KNOWN_FUTURE:
                overrides.update(history_path=None, future_path="future.csv")
            with self.subTest(role=role):
                self.assert_invalid(**overrides)

    def test_generated_known_future_may_derive_time(self):
        source = DataSourceSpec(
            name="calendar",
            source_type="generated",
            columns=(ColumnSpec("holiday", "known_future", categorical=True),),
            generator="calendar",
            availability="generator_defined",
        )
        self.assertIsNone(source.time_col)

    def test_constructor_normalizes_enum_and_string_availability(self):
        for value in (AvailabilityPolicy.SOURCE_TIME, "source_time"):
            with self.subTest(value=value):
                source = self.make_source(availability=value)
                self.assertIs(source.availability, AvailabilityPolicy.SOURCE_TIME)

    def test_temporal_roles_require_explicit_availability(self):
        cases = (
            {},
            {
                "columns": (ColumnSpec("temperature_actual", ColumnRole.OBSERVED_PAST),),
                "history_path": "weather.csv",
            },
            {
                "columns": (ColumnSpec("temperature_forecast", ColumnRole.KNOWN_FUTURE),),
                "history_path": None,
                "future_path": "forecast.csv",
            },
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                self.assert_invalid(**overrides, availability=None)
        self.assert_invalid(
            source_type="generated",
            columns=(ColumnSpec("holiday", ColumnRole.KNOWN_FUTURE),),
            history_path=None,
            time_col=None,
            generator="calendar",
            availability=None,
        )

    def test_rejects_unknown_or_non_string_availability(self):
        for value in ("future_known", "SOURCE_TIME", 0, True, object()):
            with self.subTest(value=value):
                self.assert_invalid(availability=value)

    def test_source_time_policy_validates_roles_and_metadata(self):
        observed = self.make_source(
            columns=(ColumnSpec("temperature_actual", ColumnRole.OBSERVED_PAST),),
            history_path="weather.csv",
            availability="source_time",
            provider="provided_scenario",
        )
        self.assertIs(observed.availability, AvailabilityPolicy.SOURCE_TIME)
        self.assert_invalid(time_col=None, availability="source_time")
        self.assert_invalid(
            availability="source_time",
            available_at_col="published_at",
        )
        self.assert_invalid(
            columns=(ColumnSpec("temperature_forecast", ColumnRole.KNOWN_FUTURE),),
            history_path=None,
            future_path="forecast.csv",
            availability="source_time",
        )

    def test_column_policy_is_valid_for_all_temporal_roles(self):
        cases = (
            {
                "columns": (ColumnSpec("load", ColumnRole.TARGET),),
                "history_path": "target.csv",
            },
            {
                "columns": (ColumnSpec("temperature_actual", ColumnRole.OBSERVED_PAST),),
                "history_path": "weather.csv",
            },
            {
                "columns": (ColumnSpec("temperature_forecast", ColumnRole.KNOWN_FUTURE),),
                "history_path": None,
                "future_path": "forecast.csv",
            },
        )
        for overrides in cases:
            with self.subTest(role=overrides["columns"][0].role):
                provider = (
                    "provided_scenario"
                    if overrides["columns"][0].role is ColumnRole.OBSERVED_PAST
                    else None
                )
                source = self.make_source(
                    **overrides,
                    availability="column",
                    available_at_col="published_at",
                    provider=provider,
                )
                self.assertIs(source.availability, AvailabilityPolicy.COLUMN)
        self.assert_invalid(availability="column", available_at_col=None)

    def test_forecast_origin_policy_is_known_future_file_only(self):
        source = self.make_source(
            columns=(ColumnSpec("temperature_forecast", ColumnRole.KNOWN_FUTURE),),
            history_path=None,
            future_path="forecast.csv",
            availability="forecast_origin",
        )
        self.assertIs(source.availability, AvailabilityPolicy.FORECAST_ORIGIN)
        self.assertIsNone(source.available_at_col)

        with self.assertRaisesRegex(
            ValueError,
            "availability=forecast_origin is only valid for known_future sources",
        ):
            self.make_source(availability="forecast_origin")
        with self.assertRaisesRegex(
            ValueError,
            "availability=forecast_origin requires time_col",
        ):
            self.make_source(
                columns=(ColumnSpec("temperature_forecast", ColumnRole.KNOWN_FUTURE),),
                history_path=None,
                future_path="forecast.csv",
                time_col=None,
                availability="forecast_origin",
            )
        with self.assertRaisesRegex(
            ValueError,
            "availability=forecast_origin forbids available_at_col",
        ):
            self.make_source(
                columns=(ColumnSpec("temperature_forecast", ColumnRole.KNOWN_FUTURE),),
                history_path=None,
                future_path="forecast.csv",
                availability="forecast_origin",
                available_at_col="issued_at",
            )

    def test_generator_defined_policy_is_generated_known_future_only(self):
        generated = {
            "source_type": "generated",
            "columns": (ColumnSpec("holiday", ColumnRole.KNOWN_FUTURE),),
            "history_path": None,
            "time_col": None,
            "generator": "calendar",
        }
        source = self.make_source(**generated, availability="generator_defined")
        self.assertIs(source.availability, AvailabilityPolicy.GENERATOR_DEFINED)
        self.assert_invalid(availability="generator_defined")
        self.assert_invalid(
            **generated,
            availability="generator_defined",
            available_at_col="issued_at",
        )
        self.assert_invalid(
            source_type="generated",
            columns=(ColumnSpec("note", ColumnRole.IGNORED),),
            history_path=None,
            time_col=None,
            generator="generator",
            availability="generator_defined",
        )

    def test_known_future_policy_depends_on_source_type(self):
        file_source = {
            "columns": (ColumnSpec("temperature_forecast", ColumnRole.KNOWN_FUTURE),),
            "history_path": None,
            "future_path": "forecast.csv",
        }
        self.assert_invalid(**file_source, availability="generator_defined")
        self.assert_invalid(
            **file_source,
            availability="column",
            available_at_col=None,
        )

        generated_source = {
            "source_type": "generated",
            "columns": (ColumnSpec("holiday", ColumnRole.KNOWN_FUTURE),),
            "history_path": None,
            "generator": "calendar",
        }
        self.assert_invalid(
            **generated_source,
            time_col="ts",
            availability="column",
            available_at_col="issued_at",
        )

    def test_target_may_share_source_time_history_with_observed_past(self):
        source = self.make_source(
            columns=(
                ColumnSpec("load", ColumnRole.TARGET),
                ColumnSpec("temperature_actual", ColumnRole.OBSERVED_PAST),
            ),
            provider="provided_scenario",
        )

        self.assertEqual(
            tuple(column.role for column in source.columns),
            (ColumnRole.TARGET, ColumnRole.OBSERVED_PAST),
        )
        self.assertIs(source.availability, AvailabilityPolicy.SOURCE_TIME)
        self.assertEqual(source.provider, "provided_scenario")

    def test_target_cannot_be_mixed_with_known_future(self):
        with self.assertRaisesRegex(
            ValueError,
            "target cannot be mixed with known_future in one source",
        ):
            self.make_source(
                columns=(
                    ColumnSpec("load", ColumnRole.TARGET),
                    ColumnSpec("temperature_forecast", ColumnRole.KNOWN_FUTURE),
                ),
            )

    def test_static_source_contract(self):
        valid_static = {
            "columns": (
                ColumnSpec("site_id", "key", categorical=True),
                ColumnSpec("region", "static", categorical=True),
            ),
            "series_id_cols": ("site_id",),
            "time_col": None,
            "availability": None,
            "available_at_col": None,
        }
        self.assertIsInstance(self.make_source(**valid_static), DataSourceSpec)
        invalid_cases = (
            {**valid_static, "columns": valid_static["columns"] + (ColumnSpec("load", "target"),)},
            {**valid_static, "time_col": "ts"},
            {**valid_static, "availability": "source_time"},
            {
                **valid_static,
                "availability": "column",
                "available_at_col": "published_at",
            },
            {**valid_static, "backtest_path": "backtest.csv"},
            {**valid_static, "future_path": "future.csv"},
            {**valid_static, "history_path": None, "future_path": "sites.csv"},
            {**valid_static, "series_id_cols": ()},
        )
        for overrides in invalid_cases:
            with self.subTest(overrides=overrides):
                self.assert_invalid(**overrides)

    def test_target_source_requires_history_and_forbids_actual_future_paths(self):
        self.assert_invalid(history_path=None, backtest_path="target_backtest.csv")
        self.assert_invalid(backtest_path="target_backtest.csv")
        self.assert_invalid(future_path="target_future.csv")

    def test_observed_past_source_forbids_future_path(self):
        self.assert_invalid(
            columns=(ColumnSpec("temperature_actual", "observed_past"),),
            future_path="weather_future.csv",
        )

    def test_observed_past_file_requires_history_path(self):
        self.assert_invalid(
            columns=(ColumnSpec("temperature_actual", "observed_past"),),
            history_path=None,
            backtest_path="weather_backtest.csv",
            provider="provided_scenario",
        )

    def test_observed_past_requires_explicit_provider(self):
        with self.assertRaisesRegex(
            ValueError,
            "observed_past sources require an explicit provider",
        ):
            self.make_source(
                columns=(ColumnSpec("temperature_actual", "observed_past"),),
                history_path="weather.csv",
            )

        source = self.make_source(
            columns=(ColumnSpec("temperature_actual", "observed_past"),),
            history_path="weather.csv",
            provider="provided_scenario",
        )
        self.assertEqual(source.provider, "provided_scenario")

    def test_known_future_file_requires_future_path_and_availability(self):
        known_future = {
            "columns": (ColumnSpec("temperature_forecast", "known_future"),),
            "history_path": None,
        }
        self.assert_invalid(**known_future, future_path=None)
        self.assert_invalid(
            **known_future,
            future_path="weather_future.csv",
            availability="column",
            available_at_col=None,
        )
        source = self.make_source(
            **known_future,
            future_path="weather_future.csv",
            backtest_path=None,
            availability="column",
            available_at_col="issued_at",
        )
        self.assertIsNone(source.backtest_path)

    def test_metadata_names_are_disjoint_from_columns_and_each_other(self):
        columns = (
            ColumnSpec("site_id", "key"),
            ColumnSpec("note", "ignored"),
            ColumnSpec("load", "known_future"),
        )
        common = {
            "columns": columns,
            "history_path": None,
            "future_path": "future.csv",
            "series_id_cols": ("site_id",),
            "availability": "column",
        }
        invalid_cases = (
            {**common, "time_col": "load", "available_at_col": "available_at"},
            {**common, "time_col": "site_id", "available_at_col": "available_at"},
            {**common, "time_col": "note", "available_at_col": "available_at"},
            {**common, "time_col": "time", "available_at_col": "load"},
            {**common, "time_col": "time", "available_at_col": "site_id"},
            {**common, "time_col": "time", "available_at_col": "note"},
            {**common, "time_col": "time", "available_at_col": "time"},
            {**common, "time_col": "available_at", "available_at_col": "available_at"},
        )
        for overrides in invalid_cases:
            with self.subTest(overrides=overrides):
                self.assert_invalid(**overrides)

    def test_generated_source_role_restrictions(self):
        for role in (ColumnRole.TARGET, ColumnRole.OBSERVED_PAST, ColumnRole.STATIC):
            with self.subTest(role=role):
                self.assert_invalid(
                    source_type="generated",
                    columns=(ColumnSpec(f"value_{role.value}", role),),
                    history_path=None,
                    generator="generator",
                )

    def test_available_at_only_for_dynamic_temporal_sources(self):
        self.assert_invalid(
            columns=(ColumnSpec("note", "ignored"),),
            time_col=None,
            availability="source_time",
        )
        self.assert_invalid(
            columns=(ColumnSpec("note", "ignored"),),
            time_col=None,
            availability=None,
            available_at_col="available_at",
        )
        self.assert_invalid(
            columns=(
                ColumnSpec("site_id", "key", categorical=True),
                ColumnSpec("region", "static", categorical=True),
            ),
            series_id_cols=("site_id",),
            time_col=None,
            availability=None,
            available_at_col="available_at",
        )

    def test_series_ids_must_be_unique_explicit_key_columns(self):
        key_column = ColumnSpec("site_id", "key", categorical=True)
        cases = (
            {"columns": (key_column, ColumnSpec("load", "target")), "series_id_cols": ()},
            {"columns": (ColumnSpec("load", "target"),), "series_id_cols": ("site_id",)},
            {"columns": (key_column, ColumnSpec("load", "target")), "series_id_cols": ("site_id", "site_id")},
            {"columns": (key_column, ColumnSpec("load", "target")), "series_id_cols": (" site_id",)},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                self.assert_invalid(**overrides)

    def test_optional_names_and_paths_reject_blank_or_whitespace(self):
        for field in (
            "history_path",
            "backtest_path",
            "future_path",
            "time_col",
            "available_at_col",
            "generator",
        ):
            for value in ("", "  ", f" {field}", f"{field} "):
                with self.subTest(field=field, value=value):
                    overrides = {field: value}
                    if field == "generator":
                        overrides = {
                            "source_type": "generated",
                            "columns": (ColumnSpec("holiday", "known_future"),),
                            "history_path": None,
                            "time_col": None,
                            "generator": value,
                            "availability": "generator_defined",
                        }
                    self.assert_invalid(**overrides)


class DataSpecInvalidTest(unittest.TestCase):
    def target_source(self, name="target_history", column="load"):
        return DataSourceSpec(
            name=name,
            source_type="file",
            columns=(ColumnSpec(column, "target"),),
            history_path="target.csv",
            time_col="ts",
            availability="source_time",
        )

    def feature_source(self, name, column, role):
        paths = {"history_path": "feature.csv"}
        if role == ColumnRole.KNOWN_FUTURE:
            paths = {
                "future_path": "future.csv",
                "availability": "column",
                "available_at_col": "issued_at",
            }
        elif role == ColumnRole.OBSERVED_PAST:
            paths["availability"] = "source_time"
            paths["provider"] = "provided_scenario"
        return DataSourceSpec(
            name=name,
            source_type="file",
            columns=(ColumnSpec(column, role),),
            time_col="ts" if role != ColumnRole.STATIC else None,
            **paths,
        )

    def test_requires_nonempty_sources_and_at_least_one_target(self):
        for sources in ((), []):
            with self.subTest(sources=sources):
                with self.assertRaises(ValueError):
                    DataSpec(sources)
        with self.assertRaises(ValueError):
            DataSpec((self.feature_source("past", "temperature", ColumnRole.OBSERVED_PAST),))

    def test_rejects_non_source_entries_and_duplicate_source_names(self):
        with self.assertRaises(TypeError):
            DataSpec((self.target_source(), "future"))
        with self.assertRaises(ValueError):
            DataSpec((self.target_source(), self.target_source(column="power")))

    def test_modeled_columns_are_globally_unique_across_roles_and_sources(self):
        duplicate_cases = (
            self.feature_source("past", "load", ColumnRole.OBSERVED_PAST),
            self.feature_source("future", "load", ColumnRole.KNOWN_FUTURE),
        )
        for duplicate in duplicate_cases:
            with self.subTest(duplicate=duplicate):
                with self.assertRaises(ValueError):
                    DataSpec((self.target_source(), duplicate))

        static = DataSourceSpec(
            name="metadata",
            source_type="file",
            columns=(ColumnSpec("site_id", "key"), ColumnSpec("load", "static")),
            history_path="sites.csv",
            series_id_cols=("site_id",),
        )
        with self.assertRaises(ValueError):
            DataSpec((self.target_source(), static))

    def test_key_and_ignored_names_may_repeat_across_sources(self):
        target = DataSourceSpec(
            name="target",
            source_type="file",
            columns=(ColumnSpec("site_id", "key"), ColumnSpec("note", "ignored"), ColumnSpec("load", "target")),
            history_path="target.csv",
            time_col="ts",
            series_id_cols=("site_id",),
            availability="source_time",
        )
        future = DataSourceSpec(
            name="future",
            source_type="file",
            columns=(ColumnSpec("site_id", "key"), ColumnSpec("note", "ignored"), ColumnSpec("temperature", "known_future")),
            future_path="future.csv",
            time_col="ts",
            series_id_cols=("site_id",),
            availability="column",
            available_at_col="issued_at",
        )

        data_spec = DataSpec((target, future))
        self.assertEqual(data_spec.role_columns("key"), ("site_id",))
        self.assertEqual(data_spec.role_columns("ignored"), ("note",))

    def test_column_names_cannot_change_roles_across_sources(self):
        target = self.target_source()
        conflict_cases = (
            DataSourceSpec(
                name="ignored_load",
                source_type="file",
                columns=(ColumnSpec("load", ColumnRole.IGNORED),),
                history_path="ignored.csv",
            ),
            DataSourceSpec(
                name="observed_site_id",
                source_type="file",
                columns=(ColumnSpec("site_id", ColumnRole.OBSERVED_PAST),),
                history_path="observed.csv",
                time_col="ts",
                availability="source_time",
                provider="provided_scenario",
            ),
            DataSourceSpec(
                name="future_note",
                source_type="file",
                columns=(ColumnSpec("note", ColumnRole.KNOWN_FUTURE),),
                future_path="future.csv",
                time_col="ts",
                availability="column",
                available_at_col="issued_at",
            ),
        )
        keyed_target = DataSourceSpec(
            name="keyed_target",
            source_type="file",
            columns=(
                ColumnSpec("site_id", ColumnRole.KEY),
                ColumnSpec("note", ColumnRole.IGNORED),
                ColumnSpec("load", ColumnRole.TARGET),
            ),
            history_path="target.csv",
            time_col="ts",
            series_id_cols=("site_id",),
            availability="source_time",
        )
        for sources in (
            (target, conflict_cases[0]),
            (keyed_target, conflict_cases[1]),
            (keyed_target, conflict_cases[2]),
        ):
            with self.subTest(conflict=sources[1].name):
                with self.assertRaisesRegex(ValueError, "exactly one role"):
                    DataSpec(sources)

    def test_role_columns_rejects_unknown_role(self):
        data_spec = DataSpec((self.target_source(),))
        with self.assertRaises((TypeError, ValueError)):
            data_spec.role_columns("feature")


if __name__ == "__main__":
    unittest.main()
