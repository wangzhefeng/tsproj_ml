# -*- coding: utf-8 -*-
"""RawDesignGroup batch runtime integration tests."""
from __future__ import annotations

import tempfile
import threading
import time
import unittest
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from config.config_loader import load_yaml_config
from data_loading import SourceRegistry
from forecasting_core.runtime_resources import RuntimeResourceBudget
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
from model_pipeline.batch_runtime import (
    _SharedBatchRunnerFactory,
    _batch_id,
    _resolve_config_pool_thread_limit,
    run_canonical_batch,
    verify_batch_results,
)
from model_pipeline.runner import CanonicalBaseModelRunner
from model_performance.transform_cache import FoldTransformCache


def _run_same_batch_in_process(
    config_path,
    output_root,
    counter,
    entered_run,
    release_run,
    started,
    result_queue,
):
    original_run = CanonicalBaseModelRunner.run

    def blocked_run(runner, *args, **kwargs):
        with counter.get_lock():
            counter.value += 1
        entered_run.set()
        if not release_run.wait(15):
            raise TimeoutError("test did not release batch model run")
        return original_run(runner, *args, **kwargs)

    started.set()
    try:
        with patch.object(CanonicalBaseModelRunner, "run", blocked_run):
            report = run_canonical_batch(
                (config_path,),
                output_root=output_root,
                resume=True,
            )
        result_queue.put(("ok", report.completed_count))
    except Exception as exc:
        result_queue.put(("error", type(exc).__name__, str(exc)))


class CanonicalBatchRuntimeTest(unittest.TestCase):
    def test_batch_id_is_independent_of_manifest_order(self) -> None:
        output_root = self.root / "ordered-results"
        self.assertEqual(
            _batch_id(self.paths, output_root),
            _batch_id(tuple(reversed(self.paths)), output_root),
        )

    def test_precompiled_payload_rejects_fingerprint_mismatch(self) -> None:
        loaded = load_yaml_config(self.paths[0])
        assert isinstance(loaded, ForecastConfigSpec)
        registry = SourceRegistry(loaded.data, self.root)
        runner = CanonicalBaseModelRunner(loaded, registry, self.origin)

        with self.assertRaisesRegex(ValueError, "precompiled fingerprint mismatch"):
            CanonicalBaseModelRunner(
                loaded,
                registry,
                self.origin,
                precompiled_payload=runner.raw_design_payload(),
                precompiled_fingerprint="0" * 64,
            )

    def test_failed_raw_design_construction_records_zero_group_payload_loads(self) -> None:
        output_root = self.root / "failed-construction-results"

        with patch.object(
            CanonicalBaseModelRunner,
            "_compile_supervised_arrays",
            side_effect=RuntimeError("injected compile failure"),
        ):
            report = run_canonical_batch(self.paths, output_root=output_root)

        self.assertEqual(report.completed_count, 0)
        self.assertEqual(report.failed_count, 2)
        self.assertEqual(report.raw_payload_load_count, 0)
        state = json.loads(report.state_path.read_text(encoding="utf-8"))
        self.assertEqual(len(state["groups"]), 1)
        self.assertEqual(
            next(iter(state["groups"].values()))["raw_payload_load_count"],
            0,
        )

    def test_explicit_profile_above_batch_child_budget_fails_before_groups(self) -> None:
        config = self._config("lgb", "batch/lgb", {})
        ridge_config = self._config("ridge", "batch/lgb-ridge", {"alpha": 1e-6})
        from dataclasses import replace as dataclass_replace

        config = dataclass_replace(
            config,
            problem=dataclass_replace(config.problem, horizon=8),
            features=dataclass_replace(config.features, target_lags={"load": (8, 9, 10)}),
            validation={
                **dict(config.validation),
                "performance": {
                    "window_parallel_workers": 1,
                    "multi_output_n_jobs": 8,
                    "model_thread_count": 1,
                },
            },
        )
        ridge_config = dataclass_replace(
            ridge_config,
            problem=dataclass_replace(ridge_config.problem, horizon=8),
            features=config.features,
        )
        path = self.root / "lgb.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 2,
                    **config.canonical_payload(),
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        ridge_path = self.root / "lgb-ridge.yaml"
        ridge_path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 2,
                    **ridge_config.canonical_payload(),
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        output_root = self.root / "explicit-profile-results"

        with patch.object(
            CanonicalBaseModelRunner,
            "run",
            side_effect=AssertionError("model execution before global preflight"),
        ) as run_mock, patch.object(
            CanonicalBaseModelRunner,
            "run_prelimited",
            side_effect=AssertionError("model execution before global preflight"),
        ) as prelimited_mock:
            with self.assertRaisesRegex(
                ValueError,
                r"batch child budget.*budget_product=8, available=4.*config_workers=2",
            ):
                run_canonical_batch(
                    (self.paths[0], path, ridge_path),
                    output_root=output_root,
                    config_workers=2,
                    parent_budget=RuntimeResourceBudget(
                        physical_cores=8,
                        logical_cores=8,
                        total_thread_limit=8,
                        memory_limit_bytes=1_000_000_000,
                    ),
                )

        run_mock.assert_not_called()
        prelimited_mock.assert_not_called()
        state_paths = list((output_root / "_batch_state").glob("*.json"))
        self.assertEqual(len(state_paths), 1)
        state = json.loads(state_paths[0].read_text())
        self.assertTrue(all(task["status"] == "failed" for task in state["tasks"].values()))
        self.assertTrue(all(task["error"]["phase"] == "preflight" for task in state["tasks"].values()))

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_path = self.root / "load.csv"
        times = pd.date_range("2026-01-01", periods=96, freq="1h")
        pd.DataFrame(
            {
                "time": times,
                "load": 100.0 + np.arange(len(times), dtype=float),
            }
        ).to_csv(self.data_path, index=False)
        self.origin = pd.Timestamp(times[-3])
        self.paths = (
            self._write_config("ridge", "batch/ridge", {"alpha": 1e-6}),
            self._write_config("lasso", "batch/lasso", {"alpha": 1e-6}),
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _config(
        self,
        model_type: str,
        scenario_subpath: str,
        params: dict[str, float],
    ) -> ForecastConfigSpec:
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
                transformations={
                    "feature_scaling": {"method": "standard"},
                    "target": {
                        "calendar_normalization": {"method": "none"},
                        "decomposition": {"method": "none"},
                        "scaling": {"method": "standard", "inverse": True},
                    },
                },
            ),
            strategy=ForecastStrategySpec("direct"),
            estimator=EstimatorSpec(
                model_type=model_type,
                target_adapter="independent",
                params=params,
            ),
            probabilistic={"mode": "point"},
            validation={
                "forecast_origin": self.origin.isoformat(),
                "history_steps": 48,
                "train_window_steps": 16,
                "fold_count": 2,
                "stride_steps": 2,
                "performance": {"total_thread_limit": 2},
            },
            output={"scenario_subpath": scenario_subpath},
        )

    def _write_config(
        self,
        model_type: str,
        scenario_subpath: str,
        params: dict[str, float],
    ) -> Path:
        path = self.root / f"{model_type}.yaml"
        payload = {
            "schema_version": 2,
            **self._config(model_type, scenario_subpath, params).canonical_payload(),
        }
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def test_shared_group_materializes_raw_and_each_transform_branch_once(self) -> None:
        output_root = self.root / "batch-results"
        compile_calls = 0
        transform_calls = 0
        original_compile = CanonicalBaseModelRunner._compile_supervised_arrays
        from model_pipeline import runner as runtime_module

        original_transform = runtime_module._fit_runtime_transforms

        def counted_compile(runner):
            nonlocal compile_calls
            compile_calls += 1
            return original_compile(runner)

        def counted_transform(*args, **kwargs):
            nonlocal transform_calls
            transform_calls += 1
            return original_transform(*args, **kwargs)

        with patch.object(
            CanonicalBaseModelRunner,
            "_compile_supervised_arrays",
            counted_compile,
        ), patch.object(
            runtime_module,
            "_fit_runtime_transforms",
            counted_transform,
        ):
            report = run_canonical_batch(self.paths, output_root=output_root)

        self.assertEqual(report.input_count, 2)
        self.assertEqual(report.completed_count, 2)
        self.assertEqual(report.failed_count, 0)
        self.assertEqual(report.group_count, 1)
        self.assertEqual(compile_calls, 1)
        self.assertEqual(transform_calls, 3)
        self.assertEqual(report.raw_payload_load_count, 1)
        self.assertEqual(report.transform_cache["misses"], 3)
        self.assertEqual(report.transform_cache["hits"], 3)
        verified = verify_batch_results(report.state_path)
        self.assertEqual(verified["completed_count"], 2)
        self.assertEqual(verified["verified_count"], 2)

    def test_failed_task_resumes_without_rerunning_completed_task(self) -> None:
        output_root = self.root / "resume-results"
        original_run = CanonicalBaseModelRunner.run
        first_calls: list[str] = []

        def fail_lasso(runner, *args, **kwargs):
            first_calls.append(runner.config.estimator.model_type)
            if runner.config.estimator.model_type == "lasso":
                raise RuntimeError("injected model failure")
            return original_run(runner, *args, **kwargs)

        with patch.object(CanonicalBaseModelRunner, "run", fail_lasso):
            first = run_canonical_batch(self.paths, output_root=output_root)

        self.assertEqual(first.completed_count, 1)
        self.assertEqual(first.failed_count, 1)
        self.assertEqual(first_calls, ["ridge", "lasso"])
        with self.assertRaisesRegex(ValueError, "is incomplete"):
            verify_batch_results(first.state_path)
        first_state = json.loads(first.state_path.read_text(encoding="utf-8"))
        completed_before = next(
            task
            for task in first_state["tasks"].values()
            if task["status"] == "completed"
        )
        completed_metadata_before = json.loads(
            Path(completed_before["artifacts"]["resolved_config"]).read_text(
                encoding="utf-8"
            )
        )["runtime"]["batch"]

        second_calls: list[str] = []

        def count_resume(runner, *args, **kwargs):
            second_calls.append(runner.config.estimator.model_type)
            return original_run(runner, *args, **kwargs)

        with patch.object(CanonicalBaseModelRunner, "run", count_resume):
            second = run_canonical_batch(
                self.paths,
                output_root=output_root,
                resume=True,
            )

        self.assertEqual(second.completed_count, 2)
        self.assertEqual(second.failed_count, 0)
        self.assertEqual(second_calls, ["lasso"])
        verified = verify_batch_results(second.state_path)
        self.assertEqual(verified["verified_count"], 2)
        second_state = json.loads(second.state_path.read_text(encoding="utf-8"))
        completed_after = second_state["tasks"][completed_before["task_id"]]
        completed_metadata_after = json.loads(
            Path(completed_after["artifacts"]["resolved_config"]).read_text(
                encoding="utf-8"
            )
        )["runtime"]["batch"]
        self.assertEqual(completed_metadata_after, completed_metadata_before)

    def test_batch_checkpoint_reuses_completed_folds_after_final_failure(self) -> None:
        from sklearn.linear_model import Ridge
        fit_count = 0
        original_fit = Ridge.fit

        def counted_fit(model, *args, **kwargs):
            nonlocal fit_count
            fit_count += 1
            return original_fit(model, *args, **kwargs)

        root = self.root / "checkpoint-results"
        with patch.object(Ridge, "fit", counted_fit):
            with patch.object(CanonicalBaseModelRunner, "fit_final", side_effect=RuntimeError("final fault")):
                first = run_canonical_batch(self.paths[:1], output_root=root)
            self.assertEqual(first.failed_count, 1)
            backtest_fits = fit_count
            self.assertGreater(backtest_fits, 0)
            second = run_canonical_batch(self.paths[:1], output_root=root)
            self.assertEqual(second.completed_count, 1)
            resumed_fits = fit_count - backtest_fits
            before_fresh = fit_count
            fresh = run_canonical_batch(self.paths[:1], output_root=self.root / "fresh-checkpoint", resume=False)
            self.assertEqual(fresh.completed_count, 1)
            self.assertEqual(fit_count - before_fresh, backtest_fits + resumed_fits)
        import hashlib
        resumed_state = json.loads(second.state_path.read_text())
        fresh_state = json.loads(fresh.state_path.read_text())
        a = next(iter(resumed_state["tasks"].values()))["artifacts"]
        b = next(iter(fresh_state["tasks"].values()))["artifacts"]
        for name in ("backtest", "scores", "horizon_scores", "forecast"):
            self.assertEqual(hashlib.sha256(Path(a[name]).read_bytes()).hexdigest(),
                             hashlib.sha256(Path(b[name]).read_bytes()).hexdigest())

    def test_shared_dynamic_runner_factory_reuses_raw_payload_and_child_services(self) -> None:
        loaded = load_yaml_config(self.paths[0])
        self.assertIsInstance(loaded, ForecastConfigSpec)
        registry = SourceRegistry(loaded.data, self.root)
        origin_value = loaded.validation.get("forecast_origin")
        self.assertIsNotNone(origin_value)
        origin = pd.Timestamp(str(origin_value))
        budget = RuntimeResourceBudget(
            physical_cores=2,
            logical_cores=2,
            total_thread_limit=2,
            memory_limit_bytes=1_000_000_000,
            parent_concurrency=2,
            source="batch-test",
        )
        transform_cache = FoldTransformCache()
        factory = _SharedBatchRunnerFactory(
            compiled_cache_root=self.root / "dynamic-cache",
            resource_budget=budget,
            fold_transform_cache=transform_cache,
        )
        compile_calls = 0
        original_compile = CanonicalBaseModelRunner._compile_supervised_arrays

        def counted_compile(runner):
            nonlocal compile_calls
            compile_calls += 1
            return original_compile(runner)

        with patch.object(
            CanonicalBaseModelRunner,
            "_compile_supervised_arrays",
            counted_compile,
        ):
            first = factory(loaded, registry, origin, checkpoint_root=self.root / "dynamic-checkpoints")
            second = factory(loaded, registry, origin, checkpoint_root=self.root / "dynamic-checkpoints")

        self.assertEqual(compile_calls, 1)
        self.assertIsNot(first, second)
        self.assertFalse(first.compiled_cache_hit)
        self.assertTrue(second.compiled_cache_hit)
        self.assertIs(first.fold_transform_cache, transform_cache)
        self.assertIs(second.fold_transform_cache, transform_cache)
        self.assertEqual(first.resource_budget.parent_concurrency, 2)
        self.assertEqual(second.resource_budget.parent_concurrency, 2)
        self.assertEqual(first.checkpoint_root, self.root / "dynamic-checkpoints")
        self.assertEqual(second.checkpoint_root, self.root / "dynamic-checkpoints")

    def test_resume_rejects_changed_physical_config(self) -> None:
        output_root = self.root / "changed-config-results"
        first = run_canonical_batch(self.paths, output_root=output_root)
        self.assertEqual(first.completed_count, 2)
        payload = yaml.safe_load(self.paths[1].read_text(encoding="utf-8"))
        payload["estimator"]["params"]["alpha"] = 0.25
        self.paths[1].write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "physical config changed"):
            run_canonical_batch(
                self.paths,
                output_root=output_root,
                resume=True,
            )

    def test_same_batch_is_single_flight_across_spawn_processes(self) -> None:
        context = multiprocessing.get_context("spawn")
        counter = context.Value("i", 0)
        entered_run = context.Event()
        release_run = context.Event()
        first_started = context.Event()
        second_started = context.Event()
        result_queue = context.Queue()
        output_root = self.root / "process-lock-results"
        arguments = (
            str(self.paths[0]),
            str(output_root),
            counter,
            entered_run,
            release_run,
        )
        first = context.Process(
            target=_run_same_batch_in_process,
            args=(*arguments, first_started, result_queue),
        )
        second = context.Process(
            target=_run_same_batch_in_process,
            args=(*arguments, second_started, result_queue),
        )

        first.start()
        self.assertTrue(first_started.wait(10))
        self.assertTrue(entered_run.wait(10))
        second.start()
        self.assertTrue(second_started.wait(10))
        time.sleep(0.5)
        with counter.get_lock():
            self.assertEqual(counter.value, 1)
        release_run.set()
        first.join(20)
        second.join(20)

        self.assertEqual(first.exitcode, 0)
        self.assertEqual(second.exitcode, 0)
        self.assertEqual(
            sorted(result_queue.get(timeout=5) for _ in range(2)),
            [("ok", 1), ("ok", 1)],
        )
        with counter.get_lock():
            self.assertEqual(counter.value, 1)

    def test_config_pool_rejects_heterogeneous_process_thread_limits(self) -> None:
        homogeneous = (
            SimpleNamespace(execution_plan=SimpleNamespace(model_threads=2)),
            SimpleNamespace(execution_plan=SimpleNamespace(model_threads=2)),
        )
        heterogeneous = (
            *homogeneous,
            SimpleNamespace(execution_plan=SimpleNamespace(model_threads=1)),
        )

        self.assertEqual(_resolve_config_pool_thread_limit(homogeneous), 2)
        with self.assertRaisesRegex(ValueError, "heterogeneous model_threads"):
            _resolve_config_pool_thread_limit(heterogeneous)

    def test_parent_config_workers_bound_child_plans_and_run_concurrently(self) -> None:
        output_root = self.root / "parallel-results"
        original = CanonicalBaseModelRunner.run_prelimited
        lock = threading.Lock()
        active = 0
        max_active = 0

        def counted(runner, *args, **kwargs):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.05)
                return original(runner, *args, **kwargs)
            finally:
                with lock:
                    active -= 1

        with patch.object(CanonicalBaseModelRunner, "run_prelimited", counted):
            report = run_canonical_batch(
                self.paths,
                output_root=output_root,
                config_workers=2,
            )

        self.assertEqual(
            report.completed_count,
            2,
            msg=report.state_path.read_text(encoding="utf-8"),
        )
        self.assertGreaterEqual(max_active, 2)
        state = yaml.safe_load(report.state_path.read_text(encoding="utf-8"))
        for task in state["tasks"].values():
            resolved = yaml.safe_load(
                Path(task["artifacts"]["resolved_config"]).read_text(
                    encoding="utf-8"
                )
            )
            resources = resolved["runtime"]["resources"]
            self.assertEqual(resources["budget"]["parent_concurrency"], 2)
            self.assertLessEqual(
                2 * resources["execution_plan"]["budget_product"],
                resources["budget"]["total_thread_limit"],
            )


if __name__ == "__main__":
    unittest.main()
