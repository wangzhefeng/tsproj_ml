# -*- coding: utf-8 -*-
"""监督特征编译产物缓存的命中与失效测试。"""
from __future__ import annotations

import ast
import copy
import importlib.util
import multiprocessing
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from time import perf_counter, sleep
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_loading import SourceRegistry
from feature_engineering.cache import (
    COMPILED_CACHE_DIR_NAME,
    compute_raw_design_fingerprint,
    raw_design_cache_lock,
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
    parse_model_config,
)
from model_forecasting.runtime import CanonicalBaseModelRunner


def _generated_feature_a(*_args, **_kwargs):
    return "a"


def _generated_feature_b(*_args, **_kwargs):
    return "b"


def _hold_raw_design_cache_lock(
    cache_root: str,
    attempting,
    entered,
    release,
) -> None:
    attempting.set()
    with raw_design_cache_lock(cache_root, "shared-key"):
        entered.set()
        if not release.wait(timeout=10.0):
            raise TimeoutError("test process did not receive lock release")


class CompiledFeatureCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_path = self.root / "load.csv"
        self.times = pd.date_range("2026-01-01", periods=64, freq="1h")
        self._write_values(np.arange(len(self.times), dtype=float))
        self.origin = cast(pd.Timestamp, pd.Timestamp(self.times.to_numpy()[-4]))
        self.config = self._config()
        self.cache_root = self.root / "results"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_values(self, values: np.ndarray) -> None:
        pd.DataFrame(
            {
                "time": self.times,
                "load": 100.0 + values,
            }
        ).to_csv(self.data_path, index=False)

    def _config(
        self,
        performance: dict[str, int] | None = None,
    ) -> ForecastConfigSpec:
        validation = {
            "forecast_origin": str(self.origin),
            "history_steps": 32,
            "train_window_steps": 16,
            "fold_count": 1,
            "stride_steps": 2,
        }
        if performance is not None:
            validation["performance"] = performance
        return ForecastConfigSpec(
            problem=ForecastProblemSpec(
                time_col="time",
                freq="1h",
                horizon=2,
                targets=("load",),
                training_scope="local",
                series_id_cols=(),
            ),
            data=DataSpec(
                (
                    DataSourceSpec(
                        name="target_history",
                        source_type="file",
                        columns=(ColumnSpec("load", "target"),),
                        history_path=str(self.data_path),
                        time_col="time",
                        availability="source_time",
                    ),
                )
            ),
            features=FeatureSpec(
                target_lags={"load": (2, 3, 4)},
                observed_past_lags={},
                datetime_features=("hour",),
                transformations={},
            ),
            strategy=ForecastStrategySpec("direct"),
            estimator=EstimatorSpec(
                model_type="ridge",
                target_adapter="independent",
                params={"alpha": 1e-6},
            ),
            probabilistic={"mode": "point"},
            validation=validation,
            output={"scenario_subpath": "compiled-cache"},
        )

    def _runner(self) -> CanonicalBaseModelRunner:
        return CanonicalBaseModelRunner(
            self.config,
            SourceRegistry(self.config.data, self.root),
            self.origin,
            compiled_cache_root=self.cache_root,
        )

    def test_compiler_source_change_invalidates_raw_design(self) -> None:
        from feature_engineering import cache

        def fingerprint():
            return compute_raw_design_fingerprint(
                self.config, base_dir=self.root, origin=self.origin, generators={}
            )

        original = cache.file_sha256
        before = fingerprint()

        def changed(path):
            if Path(path).name == "compiler.py":
                return "changed-compiler-implementation"
            return original(path)

        with patch.object(cache, "file_sha256", side_effect=changed):
            self.assertNotEqual(before, fingerprint())
        self.assertEqual(before, fingerprint())

    def test_cache_hit_is_fast_and_source_change_invalidates(self) -> None:
        first = self._runner()
        self.assertFalse(first.compiled_cache_hit)
        first_resources = first.runtime_resources_payload()
        first_compile = first_resources["training_compile"]
        self.assertEqual(first_compile["mode"], "batch")
        self.assertEqual(first_compile["reason_codes"], [])
        self.assertGreater(first_compile["origin_count"], 0)
        self.assertGreater(first_compile["estimated_origin_call_count"], 0)
        self.assertEqual(
            set(first_compile["stage_wall_seconds"]),
            {
                "materialize",
                "labels",
                "compile_batch",
                "compile_row",
                "batch_prepare",
                "batch_feature_columns",
                "finish_and_proof_validation",
                "split_concatenate",
            },
        )
        self.assertTrue(
            all(
                value >= 0.0
                for value in first_compile["stage_wall_seconds"].values()
            )
        )
        self.assertEqual(
            set(first_resources["cache"]["stage_wall_seconds"]),
            {"load", "compile", "write"},
        )
        self.assertGreater(
            first_resources["cache"]["stage_wall_seconds"]["compile"],
            0.0,
        )
        self.assertGreaterEqual(
            first_resources["cache"]["stage_wall_seconds"]["write"],
            0.0,
        )
        cache_parent = self.cache_root / COMPILED_CACHE_DIR_NAME
        first_entries = tuple(cache_parent.iterdir())
        self.assertEqual(len(first_entries), 1)

        start = perf_counter()
        second = self._runner()
        elapsed = perf_counter() - start
        self.assertTrue(second.compiled_cache_hit)
        self.assertLess(elapsed, 1.0)
        second_resources = second.runtime_resources_payload()
        self.assertEqual(second_resources["training_compile"]["mode"], "cache_hit")
        self.assertGreaterEqual(
            second_resources["cache"]["stage_wall_seconds"]["load"],
            0.0,
        )
        self.assertEqual(
            second_resources["cache"]["stage_wall_seconds"]["compile"],
            0.0,
        )
        self.assertEqual(
            second_resources["cache"]["stage_wall_seconds"]["write"],
            0.0,
        )
        for actual, expected in zip(second.X_all, first.X_all):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(second.Y_all, first.Y_all)
        self.assertEqual(second.feature_schema, first.feature_schema)

        changed = np.arange(len(self.times), dtype=float)
        changed[0] = -999.0
        self._write_values(changed)
        third = self._runner()
        self.assertFalse(third.compiled_cache_hit)
        self.assertEqual(len(tuple(cache_parent.iterdir())), 2)

    def test_performance_controls_share_compiled_design_cache(self) -> None:
        first = self._runner()
        self.assertFalse(first.compiled_cache_hit)
        configured = self._config(
            {
                "window_parallel_workers": 2,
                "total_thread_limit": 4,
            }
        )

        second = CanonicalBaseModelRunner(
            configured,
            SourceRegistry(configured.data, self.root),
            self.origin,
            compiled_cache_root=self.cache_root,
        )

        self.assertTrue(second.compiled_cache_hit)
        cache_parent = self.cache_root / COMPILED_CACHE_DIR_NAME
        self.assertEqual(len(tuple(cache_parent.iterdir())), 1)

    def test_fit_and_output_only_changes_share_raw_design_cache(self) -> None:
        first = self._runner()
        self.assertFalse(first.compiled_cache_hit)
        payload: dict[str, Any] = copy.deepcopy(self.config.canonical_payload())
        payload["estimator"] = {
            "model_type": "lightgbm",
            "target_adapter": "independent",
            "params": {},
        }
        payload["probabilistic"] = {
            "mode": "quantile",
            "quantiles": [0.1, 0.5, 0.9],
            "point_quantile": 0.5,
        }
        payload["output"]["scenario_subpath"] = "other-output"
        payload["validation"]["performance"] = {
            "multi_output_n_jobs": 8,
        }
        payload["validation"]["eval_mask"] = {
            "mode": "range",
            "min_value": 0.0,
        }
        payload["features"]["transformations"]["feature_scaling"] = {
            "method": "minmax",
        }
        payload["features"]["transformations"]["target"] = {
            "scaling": {"method": "standard", "inverse": True},
        }
        payload["features"]["selection"] = {
            "method": "correlation",
            "max_features": 2,
        }
        configured = parse_model_config(payload, source="fit-only-change.yaml")

        second = CanonicalBaseModelRunner(
            configured,
            SourceRegistry(configured.data, self.root),
            self.origin,
            compiled_cache_root=self.cache_root,
        )

        self.assertTrue(second.compiled_cache_hit)
        self.assertNotEqual(self.config.fingerprint(), configured.fingerprint())
        self.assertNotEqual(self.config.result_identity(), configured.result_identity())
        self.assertEqual(first.compiled_cache_fingerprint, second.compiled_cache_fingerprint)
        for actual, expected in zip(second.X_all, first.X_all):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(second.Y_all, first.Y_all)
        cache_parent = self.cache_root / COMPILED_CACHE_DIR_NAME
        self.assertEqual(len(tuple(cache_parent.iterdir())), 1)

    def test_concurrent_same_key_miss_compiles_once(self) -> None:
        compile_calls = 0
        count_lock = Lock()
        compile_supervised = CanonicalBaseModelRunner._compile_supervised_arrays

        def slow_compile(runner: CanonicalBaseModelRunner) -> None:
            nonlocal compile_calls
            with count_lock:
                compile_calls += 1
            sleep(0.1)
            compile_supervised(runner)

        def build_runner(_index: int) -> CanonicalBaseModelRunner:
            return CanonicalBaseModelRunner(
                self.config,
                SourceRegistry(self.config.data, self.root),
                self.origin,
                compiled_cache_root=self.cache_root,
            )

        with patch.object(
            CanonicalBaseModelRunner,
            "_compile_supervised_arrays",
            slow_compile,
        ), ThreadPoolExecutor(max_workers=2) as executor:
            runners = tuple(executor.map(build_runner, range(2)))

        self.assertEqual(compile_calls, 1)
        self.assertEqual(
            sorted(runner.compiled_cache_hit for runner in runners),
            [False, True],
        )
        for actual, expected in zip(runners[1].X_all, runners[0].X_all):
            np.testing.assert_array_equal(actual, expected)

    def test_cache_module_does_not_import_posix_lock_unconditionally(self) -> None:
        cache_module = Path(__file__).parents[1] / "feature_engineering/cache.py"
        module = ast.parse(cache_module.read_text(encoding="utf-8"))

        self.assertFalse(
            any(
                (
                    isinstance(statement, ast.ImportFrom)
                    and statement.module == "fcntl"
                )
                or (
                    isinstance(statement, ast.Import)
                    and any(alias.name == "fcntl" for alias in statement.names)
                )
                for statement in module.body
            ),
            "cache module must remain importable on platforms without fcntl",
        )

    def test_raw_design_cache_lock_serializes_processes(self) -> None:
        context = multiprocessing.get_context("spawn")
        first_attempting = context.Event()
        first_entered = context.Event()
        first_release = context.Event()
        second_attempting = context.Event()
        second_entered = context.Event()
        second_release = context.Event()
        first = context.Process(
            target=_hold_raw_design_cache_lock,
            args=(
                str(self.cache_root),
                first_attempting,
                first_entered,
                first_release,
            ),
        )
        second = context.Process(
            target=_hold_raw_design_cache_lock,
            args=(
                str(self.cache_root),
                second_attempting,
                second_entered,
                second_release,
            ),
        )
        first.start()
        try:
            self.assertTrue(first_entered.wait(timeout=5.0))
            second.start()
            self.assertTrue(second_attempting.wait(timeout=5.0))
            self.assertFalse(second_entered.wait(timeout=0.2))
            first_release.set()
            self.assertTrue(second_entered.wait(timeout=5.0))
            second_release.set()
        finally:
            first_release.set()
            second_release.set()
            first.join(timeout=5.0)
            if first.is_alive():
                first.terminate()
                first.join(timeout=5.0)
            if second.pid is not None:
                second.join(timeout=5.0)
                if second.is_alive():
                    second.terminate()
                    second.join(timeout=5.0)
        self.assertEqual(first.exitcode, 0)
        self.assertEqual(second.exitcode, 0)

    def test_raw_design_inputs_invalidate_fingerprint(self) -> None:
        def fingerprint(
            config: ForecastConfigSpec,
            *,
            origin: pd.Timestamp = self.origin,
            generators: dict[str, Any] | None = None,
        ) -> str:
            return compute_raw_design_fingerprint(
                config,
                base_dir=self.root,
                origin=origin,
                generators=generators or {},
            )

        baseline = fingerprint(self.config)
        payloads: list[dict[str, Any]] = []

        changed_features: dict[str, Any] = copy.deepcopy(
            self.config.canonical_payload()
        )
        changed_features["features"]["target_lags"]["load"] = [2, 3, 5]
        payloads.append(changed_features)

        changed_strategy: dict[str, Any] = copy.deepcopy(
            self.config.canonical_payload()
        )
        changed_strategy["strategy"] = {"name": "mimo"}
        payloads.append(changed_strategy)

        changed_geometry: dict[str, Any] = copy.deepcopy(
            self.config.canonical_payload()
        )
        changed_geometry["validation"]["history_steps"] = 31
        payloads.append(changed_geometry)

        changed_data: dict[str, Any] = copy.deepcopy(self.config.canonical_payload())
        changed_data["data"]["sources"][0]["name"] = "renamed_target_history"
        payloads.append(changed_data)

        for index, payload in enumerate(payloads):
            with self.subTest(index=index):
                changed = parse_model_config(payload, source=f"raw-change-{index}.yaml")
                self.assertNotEqual(baseline, fingerprint(changed))

        self.assertNotEqual(
            baseline,
            fingerprint(
                self.config,
                origin=cast(pd.Timestamp, self.origin - pd.Timedelta(hours=1)),
            ),
        )

    def test_generator_implementation_invalidates_raw_fingerprint(self) -> None:
        payload: dict[str, Any] = copy.deepcopy(self.config.canonical_payload())
        payload["data"]["sources"].append(
            {
                "name": "generated_calendar",
                "source_type": "generated",
                "columns": [
                    {
                        "name": "calendar_value",
                        "role": "known_future",
                        "categorical": False,
                    }
                ],
                "time_col": "time",
                "series_id_cols": [],
                "availability": "generator_defined",
                "generator": "test_generator",
            }
        )
        config = parse_model_config(payload, source="generated.yaml")

        first = compute_raw_design_fingerprint(
            config,
            base_dir=self.root,
            origin=self.origin,
            generators={"test_generator": _generated_feature_a},
        )
        second = compute_raw_design_fingerprint(
            config,
            base_dir=self.root,
            origin=self.origin,
            generators={"test_generator": _generated_feature_b},
        )

        self.assertNotEqual(first, second)

    def test_generator_helper_change_invalidates_raw_fingerprint(self) -> None:
        payload: dict[str, Any] = copy.deepcopy(self.config.canonical_payload())
        payload["data"]["sources"].append(
            {
                "name": "generated_calendar",
                "source_type": "generated",
                "columns": [
                    {
                        "name": "calendar_value",
                        "role": "known_future",
                        "categorical": False,
                    }
                ],
                "time_col": "time",
                "series_id_cols": [],
                "availability": "generator_defined",
                "generator": "test_generator",
            }
        )
        config = parse_model_config(payload, source="generated-helper.yaml")

        def load_generator(filename: str, helper_value: str):
            path = self.root / filename
            path.write_text(
                "def _helper():\n"
                f"    return {helper_value!r}\n\n"
                "def generate(*_args, **_kwargs):\n"
                "    return _helper()\n",
                encoding="utf-8",
            )
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"could not load test generator from {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.generate

        first = compute_raw_design_fingerprint(
            config,
            base_dir=self.root,
            origin=self.origin,
            generators={
                "test_generator": load_generator("generator_a.py", "a")
            },
        )
        second = compute_raw_design_fingerprint(
            config,
            base_dir=self.root,
            origin=self.origin,
            generators={
                "test_generator": load_generator("generator_b.py", "b")
            },
        )

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
