# -*- coding: utf-8 -*-
"""ForecastProblemSpec 的独立问题定义契约测试。"""
import ast
import json
import unittest
import warnings
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import Literal, get_type_hints

import forecasting_core.specs as specs
from forecasting_core.specs import ForecastProblemSpec
from forecasting_core.specs.problem import Frequency


class ForecastProblemSpecValidTest(unittest.TestCase):
    def test_preserves_project_standard_frequency_spellings(self):
        for freq in ("5min", "15min", "1h", "1D", "1ME", "1MS"):
            with self.subTest(freq=freq):
                spec = ForecastProblemSpec(
                    time_col="ts",
                    freq=freq,
                    horizon=1,
                    targets="load",
                    information_mode="forecast",
                    training_scope="local",
                )

                self.assertEqual(spec.freq, freq)

    def test_canonicalizes_equivalent_supported_frequency_spellings(self):
        aliases = {
            "D": "1D",
            "24h": "1D",
            "1440min": "1D",
            "15T": "15min",
            "60min": "1h",
            "60T": "1h",
            "h": "1h",
            "H": "1h",
        }

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            for freq, expected in aliases.items():
                with self.subTest(freq=freq):
                    spec = ForecastProblemSpec(
                        time_col="ts",
                        freq=freq,
                        horizon=1,
                        targets="load",
                        information_mode="forecast",
                        training_scope="local",
                    )

                    self.assertEqual(spec.freq, expected)

    def test_equivalent_frequencies_have_the_same_canonical_identity(self):
        equivalent_groups = (
            ("15min", "15T"),
            ("1h", "60min", "60T", "H"),
            ("1D", "24h", "1440min", "D"),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            for frequencies in equivalent_groups:
                canonical = ForecastProblemSpec(
                    time_col="ts",
                    freq=frequencies[0],
                    horizon=1,
                    targets="load",
                    information_mode="forecast",
                    training_scope="local",
                )
                for freq in frequencies[1:]:
                    with self.subTest(canonical=frequencies[0], alias=freq):
                        alias = ForecastProblemSpec(
                            time_col="ts",
                            freq=freq,
                            horizon=1,
                            targets="load",
                            information_mode="forecast",
                            training_scope="local",
                        )

                        self.assertIs(alias.freq, canonical.freq)
                        self.assertEqual(alias, canonical)
                        self.assertEqual(hash(alias), hash(canonical))
                        self.assertEqual(alias.canonical_payload(), canonical.canonical_payload())

    def test_single_target_local_problem(self):
        spec = ForecastProblemSpec(
            time_col="ts",
            freq="15min",
            horizon=96,
            targets="load",
            information_mode="forecast",
            training_scope="local",
        )

        self.assertEqual(spec.targets, ("load",))
        self.assertEqual(spec.series_id_cols, ())
        self.assertEqual(spec.n_targets, 1)
        self.assertFalse(spec.is_global)

    def test_two_target_global_problem_normalizes_sequences(self):
        targets = ["load", "temperature"]
        series_id_cols = ["site_id", "meter_id"]
        spec = ForecastProblemSpec(
            time_col="ts",
            freq="1D",
            horizon=7,
            targets=targets,
            series_id_cols=series_id_cols,
            information_mode="nowcast",
            training_scope="global",
        )
        targets.append("humidity")
        series_id_cols.append("region_id")

        self.assertEqual(spec.targets, ("load", "temperature"))
        self.assertEqual(spec.series_id_cols, ("site_id", "meter_id"))
        self.assertIsInstance(spec.targets, tuple)
        self.assertIsInstance(spec.series_id_cols, tuple)
        self.assertEqual(spec.n_targets, 2)
        self.assertTrue(spec.is_global)

    def test_local_problem_may_include_series_ids(self):
        spec = ForecastProblemSpec(
            time_col="ts",
            freq="1D",
            horizon=1,
            targets=("load",),
            series_id_cols="site_id",
            information_mode="oracle",
            training_scope="local",
        )

        self.assertEqual(spec.series_id_cols, ("site_id",))
        self.assertFalse(spec.is_global)

    def test_canonical_payload_is_json_friendly_and_deterministic(self):
        spec = ForecastProblemSpec(
            time_col="ts",
            freq="15min",
            horizon=16,
            targets=["load", "temperature"],
            series_id_cols=["site_id", "meter_id"],
            information_mode="forecast",
            training_scope="global",
        )

        expected = {
            "time_col": "ts",
            "freq": "15min",
            "horizon": 16,
            "targets": ["load", "temperature"],
            "series_id_cols": ["site_id", "meter_id"],
            "information_mode": "forecast",
            "training_scope": "global",
        }
        first = spec.canonical_payload()
        second = spec.canonical_payload()

        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertIsNot(first, second)
        self.assertEqual(json.loads(json.dumps(first)), expected)
        self.assertNotIn("model", first)
        self.assertNotIn("strategy", first)
        self.assertNotIn("covariates", first)

        first["targets"].append("humidity")
        first["series_id_cols"].append("region_id")
        self.assertEqual(spec.targets, ("load", "temperature"))
        self.assertEqual(spec.series_id_cols, ("site_id", "meter_id"))
        self.assertEqual(spec.canonical_payload(), expected)

    def test_public_constructor_typing_matches_normalized_inputs(self):
        init_hints = get_type_hints(ForecastProblemSpec.__init__)
        field_hints = get_type_hints(ForecastProblemSpec)

        self.assertEqual(
            Frequency,
            Literal["5min", "15min", "1h", "1D", "1ME", "1MS"],
        )
        self.assertEqual(init_hints["freq"], str)
        self.assertEqual(field_hints["freq"], Frequency)
        self.assertEqual(init_hints["targets"], str | Sequence[str])
        self.assertEqual(init_hints["series_id_cols"], str | Sequence[str])
        self.assertEqual(
            init_hints["information_mode"],
            Literal["forecast", "nowcast", "oracle"],
        )
        self.assertEqual(init_hints["training_scope"], Literal["local", "global"])
        self.assertEqual(field_hints["targets"], tuple[str, ...])
        self.assertEqual(field_hints["series_id_cols"], tuple[str, ...])
        self.assertEqual(
            field_hints["information_mode"],
            Literal["forecast", "nowcast", "oracle"],
        )
        self.assertEqual(field_hints["training_scope"], Literal["local", "global"])

    def test_dataclass_and_import_surface_only_define_problem_fields(self):
        self.assertEqual(
            [field.name for field in fields(ForecastProblemSpec)],
            [
                "time_col",
                "freq",
                "horizon",
                "targets",
                "information_mode",
                "training_scope",
                "series_id_cols",
            ],
        )
        self.assertIn("ForecastProblemSpec", specs.__all__)
        for name in ("model", "strategy", "covariates"):
            self.assertFalse(hasattr(ForecastProblemSpec, name))

        source_path = Path(__file__).resolve().parents[1] / "forecasting_core/specs/problem.py"
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module or ""
            for node in ast.walk(module)
            if isinstance(node, ast.ImportFrom)
        }
        imported_modules.update(
            alias.name
            for node in ast.walk(module)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        self.assertFalse(any(name == "models" or name.startswith("models.") for name in imported_modules))
        self.assertNotIn("pred_method", source_path.read_text(encoding="utf-8"))

    def test_spec_is_immutable(self):
        spec = ForecastProblemSpec(
            time_col="ts",
            freq="1D",
            horizon=1,
            targets=("load",),
            information_mode="forecast",
            training_scope="local",
        )

        with self.assertRaises(FrozenInstanceError):
            spec.horizon = 2


class ForecastProblemSpecInvalidTest(unittest.TestCase):
    def make_spec(self, **overrides):
        values = {
            "time_col": "ts",
            "freq": "15min",
            "horizon": 96,
            "targets": ("load",),
            "series_id_cols": (),
            "information_mode": "forecast",
            "training_scope": "local",
        }
        values.update(overrides)
        return ForecastProblemSpec(**values)

    def test_rejects_non_positive_horizon(self):
        for horizon in (0, -1):
            with self.subTest(horizon=horizon):
                with self.assertRaises(ValueError):
                    self.make_spec(horizon=horizon)

    def test_rejects_invalid_or_non_positive_frequency(self):
        for freq in ("bananas", "0min", "-1h"):
            with self.subTest(freq=freq):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", FutureWarning)
                    with self.assertRaises(ValueError):
                        self.make_spec(freq=freq)

    def test_rejects_parseable_but_unsupported_frequency(self):
        for freq in ("m", "1W", "1B", "2ME", "2MS", "1s", "1S", "1Y", "1YE"):
            with self.subTest(freq=freq):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", FutureWarning)
                    with self.assertRaises(ValueError):
                        self.make_spec(freq=freq)

    def test_rejects_non_integer_horizon(self):
        for horizon in (1.5, "96", True, False):
            with self.subTest(horizon=horizon):
                with self.assertRaises(TypeError):
                    self.make_spec(horizon=horizon)

    def test_rejects_blank_time_or_frequency(self):
        for field, value in (("time_col", ""), ("time_col", "  "), ("freq", ""), ("freq", "  ")):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    self.make_spec(**{field: value})

    def test_rejects_whitespace_bearing_semantic_names(self):
        cases = (
            {"time_col": " ts"},
            {"time_col": "ts "},
            {"freq": " 15min"},
            {"freq": "15min "},
            {"targets": " load "},
            {"targets": ["load", "temperature "]},
            {"series_id_cols": " site_id"},
            {"series_id_cols": ["site_id", "meter_id "]},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self.make_spec(**overrides)

    def test_duplicate_after_whitespace_is_rejected_without_normalization(self):
        cases = (
            {"targets": ("load", " load")},
            {"targets": ("load", "load ")},
            {"series_id_cols": ("site_id", " site_id")},
            {"series_id_cols": ("site_id", "site_id ")},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self.make_spec(**overrides)

    def test_rejects_empty_or_blank_targets(self):
        for targets in ((), [], "", "  ", ["load", ""]):
            with self.subTest(targets=targets):
                with self.assertRaises(ValueError):
                    self.make_spec(targets=targets)

    def test_rejects_blank_series_id_names(self):
        for series_id_cols in ("", "  ", ["site_id", ""]):
            with self.subTest(series_id_cols=series_id_cols):
                with self.assertRaises(ValueError):
                    self.make_spec(series_id_cols=series_id_cols)

    def test_rejects_duplicate_targets_or_series_ids(self):
        with self.assertRaises(ValueError):
            self.make_spec(targets=("load", "load"))
        with self.assertRaises(ValueError):
            self.make_spec(series_id_cols=("site_id", "site_id"))

    def test_rejects_name_overlap(self):
        cases = (
            {"targets": ("ts",)},
            {"series_id_cols": ("ts",)},
            {"targets": ("load",), "series_id_cols": ("load",)},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self.make_spec(**overrides)

    def test_rejects_unknown_information_mode(self):
        with self.assertRaises(ValueError):
            self.make_spec(information_mode="hindcast")

    def test_rejects_unknown_training_scope(self):
        with self.assertRaises(ValueError):
            self.make_spec(training_scope="panel")

    def test_global_scope_requires_series_id(self):
        with self.assertRaises(ValueError):
            self.make_spec(training_scope="global", series_id_cols=())

    def test_rejects_non_string_names_and_invalid_sequence_inputs(self):
        cases = (
            {"time_col": 1},
            {"freq": 15},
            {"targets": ("load", 1)},
            {"series_id_cols": ("site_id", 1)},
            {"targets": None},
            {"series_id_cols": None},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(TypeError):
                    self.make_spec(**overrides)


if __name__ == "__main__":
    unittest.main()
