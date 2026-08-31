# -*- coding: utf-8 -*-
"""七种 canonical forecasting strategy 的独立契约测试。"""
import ast
import json
import unittest
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import forecasting_core.specs as specs
import forecasting_core.specs.strategy as strategy_module
from forecasting_core.specs import ForecastStrategySpec, StrategyName


class CanonicalStrategySpecValidTest(unittest.TestCase):
    def test_strategy_name_has_exact_canonical_values(self):
        self.assertEqual(
            [strategy.value for strategy in StrategyName],
            [
                "recursive",
                "direct",
                "mimo",
                "recmo",
                "dirrec",
                "dirmo",
                "dirrecmo",
            ],
        )
        self.assertTrue(issubclass(StrategyName, str))

    def test_all_seven_strategies_have_canonical_plan_facts(self):
        cases = (
            ("recursive", None, 1, 1, True, 12),
            ("direct", None, 12, 1, False, 12),
            ("mimo", None, 1, 12, False, 1),
            ("recmo", 3, 1, 3, True, 4),
            ("dirrec", None, 12, 1, True, 12),
            ("dirmo", 3, 4, 3, False, 4),
            ("dirrecmo", 3, 4, 3, True, 4),
        )

        for name, chunk, model_count, steps_per_call, consumes_previous, n_calls in cases:
            with self.subTest(name=name):
                spec = ForecastStrategySpec(name=name, output_chunk_length=chunk)
                plan = spec.resolve(12)

                self.assertIs(spec.name, StrategyName(name))
                self.assertIsInstance(plan, strategy_module._ResolvedStrategyPlan)
                self.assertEqual(plan.horizon, 12)
                self.assertEqual(plan.model_count, model_count)
                self.assertEqual(plan.steps_per_call, steps_per_call)
                self.assertEqual(plan.consumes_previous, consumes_previous)
                self.assertEqual(plan.n_calls, n_calls)

    def test_accepts_enum_and_exact_lowercase_string_but_stores_enum(self):
        from_enum = ForecastStrategySpec(StrategyName.DIRECT)
        from_string = ForecastStrategySpec("direct")

        self.assertIs(from_enum.name, StrategyName.DIRECT)
        self.assertIs(from_string.name, StrategyName.DIRECT)
        self.assertEqual(from_enum, from_string)

    def test_problem_horizon_is_not_strategy_spec_state_or_payload(self):
        plain = ForecastStrategySpec("recursive")
        chunked = ForecastStrategySpec("dirmo", output_chunk_length=2)

        self.assertEqual(
            [field.name for field in fields(ForecastStrategySpec)],
            ["name", "output_chunk_length"],
        )
        self.assertFalse(hasattr(plain, "horizon"))
        self.assertEqual(plain.canonical_payload(), {"name": "recursive"})
        self.assertEqual(
            chunked.canonical_payload(),
            {"name": "dirmo", "output_chunk_length": 2},
        )

    def test_canonical_payload_is_deterministic_json_friendly_and_isolated(self):
        spec = ForecastStrategySpec("dirmo", output_chunk_length=2)
        expected = {"name": "dirmo", "output_chunk_length": 2}

        first = spec.canonical_payload()
        second = spec.canonical_payload()

        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertIsNot(first, second)
        self.assertEqual(json.loads(json.dumps(first)), expected)

        first["name"] = "mutated"
        first["output_chunk_length"] = 999
        self.assertEqual(spec.canonical_payload(), expected)

    def test_resolve_different_horizons_without_mutating_spec(self):
        spec = ForecastStrategySpec("direct")

        short_plan = spec.resolve(4)
        long_plan = spec.resolve(12)

        self.assertEqual(spec.canonical_payload(), {"name": "direct"})
        self.assertEqual(short_plan.horizon, 4)
        self.assertEqual(short_plan.model_count, 4)
        self.assertEqual(short_plan.n_calls, 4)
        self.assertEqual(long_plan.horizon, 12)
        self.assertEqual(long_plan.model_count, 12)
        self.assertEqual(long_plan.n_calls, 12)
        self.assertIsNot(short_plan, long_plan)

    def test_specs_and_resolved_plans_are_immutable(self):
        spec = ForecastStrategySpec("recursive")
        plan = spec.resolve(8)

        with self.assertRaises(FrozenInstanceError):
            spec.name = StrategyName.DIRECT
        with self.assertRaises(FrozenInstanceError):
            plan.horizon = 4

    def test_rule_outcomes_are_stable_across_public_resolution(self):
        first = ForecastStrategySpec("dirrecmo", output_chunk_length=3).resolve(12)
        second = ForecastStrategySpec("dirrecmo", output_chunk_length=3).resolve(12)
        unrelated = ForecastStrategySpec("recursive").resolve(12)
        third = ForecastStrategySpec("dirrecmo", output_chunk_length=3).resolve(12)

        expected = (4, 3, True, 4)
        self.assertEqual(
            (first.model_count, first.steps_per_call, first.consumes_previous, first.n_calls),
            expected,
        )
        self.assertEqual(second, first)
        self.assertEqual(third, first)
        self.assertEqual(
            (
                unrelated.model_count,
                unrelated.steps_per_call,
                unrelated.consumes_previous,
                unrelated.n_calls,
            ),
            (1, 1, True, 12),
        )

    def test_dataclass_and_import_surface_are_task_six_only(self):
        self.assertIn("StrategyName", specs.__all__)
        self.assertIn("ForecastStrategySpec", specs.__all__)
        self.assertNotIn("ResolvedStrategyPlan", specs.__all__)
        self.assertIs(specs.StrategyName, StrategyName)
        self.assertIs(specs.ForecastStrategySpec, ForecastStrategySpec)
        self.assertFalse(hasattr(specs, "ResolvedStrategyPlan"))

        with self.assertRaises(ImportError):
            exec("from forecasting_core.specs.strategy import ResolvedStrategyPlan", {})
        with self.assertRaises(TypeError):
            strategy_module._ResolvedStrategyPlan()
        with self.assertRaises(TypeError):
            strategy_module._ResolvedStrategyPlan(
                name=StrategyName.RECURSIVE,
                horizon=12,
            )

        for existing_export in (
            "ForecastProblemSpec",
            "ColumnRole",
            "ColumnSpec",
            "DataSourceSpec",
            "DataSpec",
        ):
            self.assertIn(existing_export, specs.__all__)
            self.assertTrue(hasattr(specs, existing_export))

        source_path = Path(__file__).resolve().parents[1] / "forecasting_core/specs/strategy.py"
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

        self.assertEqual(
            imported_modules - {"dataclasses", "enum", "types"},
            set(),
        )

        forbidden_names = {
            "usmd",
            "usmdp",
            "usmdr",
            "usmr",
            "msmd",
            "msmdr",
            "msmr",
            "usbr",
            "msbr",
            "ensemble",
            "global",
            "probabilistic",
        }
        identifier_surface = {
            node.id
            for node in ast.walk(module)
            if isinstance(node, ast.Name)
        }
        identifier_surface.update(
            node.attr
            for node in ast.walk(module)
            if isinstance(node, ast.Attribute)
        )
        self.assertEqual(identifier_surface & forbidden_names, set())


class CanonicalStrategySpecInvalidTest(unittest.TestCase):
    def test_rejects_invalid_names(self):
        invalid_names = (
            "Recursive",
            "DIRECT",
            " recursive",
            "recursive ",
            "usmr",
            "usmd",
            "blend",
            "pointwise",
            "unknown",
            "",
            None,
            1,
        )

        for name in invalid_names:
            with self.subTest(name=name):
                with self.assertRaises((TypeError, ValueError)):
                    ForecastStrategySpec(name=name)

    def test_non_chunk_strategies_forbid_chunk_at_construction(self):
        for name in ("recursive", "direct", "mimo", "dirrec"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    ForecastStrategySpec(name, output_chunk_length=2)

    def test_chunk_strategies_require_integer_non_bool_chunk_at_construction(self):
        invalid_chunks = (None, True, False, 2.0, "2")

        for name in ("recmo", "dirmo", "dirrecmo"):
            for chunk in invalid_chunks:
                with self.subTest(name=name, chunk=chunk):
                    expected_error = ValueError if chunk is None else TypeError
                    with self.assertRaises(expected_error):
                        ForecastStrategySpec(name, output_chunk_length=chunk)

    def test_resolve_rejects_invalid_horizons_for_every_strategy(self):
        invalid_horizons = (0, -1, True, False, 1.5, "8", None)

        for name, chunk in (
            ("recursive", None),
            ("direct", None),
            ("mimo", None),
            ("recmo", 2),
            ("dirrec", None),
            ("dirmo", 2),
            ("dirrecmo", 2),
        ):
            spec = ForecastStrategySpec(name, output_chunk_length=chunk)
            for horizon in invalid_horizons:
                with self.subTest(name=name, horizon=horizon):
                    expected_error = (
                        TypeError
                        if isinstance(horizon, bool) or not isinstance(horizon, int)
                        else ValueError
                    )
                    with self.assertRaises(expected_error):
                        spec.resolve(horizon)

    def test_all_chunk_strategies_reject_invalid_chunks_when_resolved(self):
        cases = (
            (8, -1),
            (8, 0),
            (8, 1),
            (8, 3),
            (8, 5),
            (8, 7),
            (8, 8),
            (8, 9),
            (2, 2),
            (2, 3),
            (3, 2),
            (5, 2),
            (5, 3),
            (7, 2),
            (7, 3),
        )

        for name in ("recmo", "dirmo", "dirrecmo"):
            for horizon, chunk in cases:
                with self.subTest(name=name, horizon=horizon, chunk=chunk):
                    spec = ForecastStrategySpec(name, output_chunk_length=chunk)
                    with self.assertRaises(ValueError):
                        spec.resolve(horizon)


if __name__ == "__main__":
    unittest.main()
