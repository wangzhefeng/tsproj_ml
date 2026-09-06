"""Recoverable RawDesignGroup batch runtime for canonical model configs."""
from __future__ import annotations

import hashlib
import json
import errno
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from time import perf_counter, sleep
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence

import psutil
from threadpoolctl import threadpool_limits

if os.name == "nt":
    import msvcrt as _process_lock_backend
else:
    import fcntl as _process_lock_backend

from config.config_loader import load_yaml_config
from data_loading import BUILTIN_GENERATORS, SourceRegistry
from feature_engineering.cache import compute_raw_design_fingerprint
from forecasting_core.runtime_resources import RuntimeResourceBudget
from forecasting_core.specs import ForecastConfigSpec

from model_forecasting.batch_artifacts import (
    artifact_paths, artifact_digests, artifacts_complete, validate_artifacts,
)
from model_performance.batch_memory import BoundedPayloadCache, SampledRSS
from forecasting_core.checkpoints import FitCheckpointError
from uuid import uuid4
from model_performance.resource_planner import detect_runtime_budget, plan_runtime_execution
from model_forecasting.runtime import CanonicalBaseModelRunner
from model_performance.transform_cache import FoldTransformCache
from model_testing.primitives import resolve_origin


_BATCH_LOCKS: dict[Path, threading.Lock] = {}
_BATCH_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True, slots=True)
class CanonicalBatchReport:
    input_count: int
    completed_count: int
    failed_count: int
    group_count: int
    raw_payload_load_count: int
    transform_cache: dict[str, int]
    peak_rss_bytes: int
    state_path: Path


@dataclass(frozen=True, slots=True)
class _BatchTask:
    task_id: str
    path: Path
    config: ForecastConfigSpec
    registry: SourceRegistry
    origin: Any
    raw_fingerprint: str
    physical_sha256: str


class _SharedBatchRunnerFactory:
    """Create distinct runners while sharing exact raw payloads and child services."""

    def __init__(
        self,
        *,
        compiled_cache_root: Path,
        resource_budget: RuntimeResourceBudget,
        fold_transform_cache: FoldTransformCache,
        checkpoint_root: Path | None = None,
    ) -> None:
        self._compiled_cache_root = compiled_cache_root
        self._resource_budget = resource_budget
        self._fold_transform_cache = fold_transform_cache
        self._checkpoint_root = checkpoint_root
        self._payloads = BoundedPayloadCache(
            max_bytes=max(1, resource_budget.memory_limit_bytes // 8),
            max_entries=4,
        )

    def __call__(
        self,
        config: ForecastConfigSpec,
        registry: SourceRegistry,
        origin: Any,
        *,
        checkpoint_root: str | Path | None = None,
    ) -> CanonicalBaseModelRunner:
        resolved_checkpoint_root = (
            Path(checkpoint_root) if checkpoint_root is not None else self._checkpoint_root
        )
        fingerprint = compute_raw_design_fingerprint(
            config,
            base_dir=registry._base_dir,
            origin=origin,
            generators=registry._generators,
        )
        created = []

        def materialize():
            runner = CanonicalBaseModelRunner(
                config,
                registry,
                origin,
                compiled_cache_root=self._compiled_cache_root,
                resource_budget=self._resource_budget,
                fold_transform_cache=self._fold_transform_cache,
                checkpoint_root=resolved_checkpoint_root,
            )
            created.append(runner)
            return runner.raw_design_payload()

        payload = self._payloads.get_or_create(fingerprint, materialize)
        if created:
            return created[0]
        return CanonicalBaseModelRunner(
            config, registry, origin,
            compiled_cache_root=self._compiled_cache_root,
            resource_budget=self._resource_budget,
            precompiled_payload=payload,
            precompiled_fingerprint=fingerprint,
            fold_transform_cache=self._fold_transform_cache,
            checkpoint_root=resolved_checkpoint_root,
        )


def _task_id(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _resolve_config_pool_thread_limit(
    runners: Sequence[Any],
) -> int:
    limits = {int(runner.execution_plan.model_threads) for runner in runners}
    if not limits:
        return 1
    if len(limits) != 1:
        raise ValueError(
            "config pool cannot mix heterogeneous model_threads because "
            "BLAS/OpenMP limits are process-global; got "
            f"{sorted(limits)}"
        )
    return next(iter(limits))


def _validate_explicit_plans_within_child_budgets(
    tasks: Sequence[tuple[_BatchTask, CanonicalBaseModelRunner]],
    *,
    config_workers: int,
) -> None:
    """Fail fast when an explicit performance profile cannot fit batch concurrency.

    Group-level failures during model execution are already recoverable, but an
    explicit profile that can never satisfy its declared workers is a planning
    error: surface it before any group starts instead of failing per task.
    """
    for task, runner in tasks:
        if runner.execution_plan.profile_source != "validation.performance":
            continue
        child_available = max(
            1,
            runner.execution_plan.available_threads // config_workers,
        )
        if runner.execution_plan.budget_product > child_available:
            raise ValueError(
                "batch child budget cannot satisfy explicit performance "
                f"profile for {task.path}: "
                f"budget_product={runner.execution_plan.budget_product}, "
                f"available={child_available} "
                f"(config_workers={config_workers})"
            )


def _batch_id(paths: tuple[Path, ...], output_root: Path) -> str:
    payload = {
        "paths": sorted(str(path.resolve()) for path in paths),
        "output_root": str(output_root.resolve()),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _preflight_groups(groups, state, root, budget, config_workers):
    """Validate every group before any fit; retain only metadata between groups.

    Compilation uses the durable RawDesign cache. Execution reloads one group at
    a time rather than holding all preflight designs in memory. A compile failure
    remains a failed config task, not an excuse to skip planning other groups.
    """
    summaries = {}
    for group_key, group_tasks in groups.items():
        pending = [task for task in group_tasks if not (
            state["tasks"][task.task_id].get("status") == "completed"
            and _artifacts_complete(state["tasks"][task.task_id])
        )]
        if not pending:
            continue
        workers = min(config_workers, len(pending))
        budget.for_children(workers)
        raw_payload = None
        runners = []
        for task in pending:
            try:
                runner = CanonicalBaseModelRunner(
                    task.config, task.registry, task.origin,
                    compiled_cache_root=root, resource_budget=budget,
                    precompiled_payload=raw_payload,
                    precompiled_fingerprint=group_key if raw_payload is not None else None,
                )
            except RuntimeError:
                # The execution construction path records config compile failures.
                continue
            if raw_payload is None:
                raw_payload = runner.raw_design_payload()
            runners.append((task, runner))
        if workers > 1:
            _validate_explicit_plans_within_child_budgets(runners, config_workers=workers)
        for task, runner in runners:
            runner.resource_budget = runner.resource_budget.for_children(workers)
            runner.execution_plan = plan_runtime_execution(
                task.config, runner.workload, budget=runner.resource_budget,
                base_dir=task.registry._base_dir, feature_schema=runner.builder.feature_schema,
            )
        if workers > 1:
            _resolve_config_pool_thread_limit(tuple(runner for _, runner in runners))
        # Shared raw, up to four calendar horizons, transient transform/fit arrays,
        # and a bounded retained cache. Native model RSS is measured, not hard capped.
        design_bytes = max((r.workload.design_bytes for _, r in runners), default=0)
        dynamic_factor = 4 if any(
            task.config.validation.get("horizon_mode") == "calendar_month" for task in pending
        ) else 1
        reserve = budget.memory_limit_bytes // 4
        estimated = design_bytes * (dynamic_factor + 4 * workers) + reserve
        limit = min([budget.memory_limit_bytes] + [r.resource_budget.memory_limit_bytes for _, r in runners])
        if estimated > limit:
            raise ValueError(
                "batch group memory admission exceeds budget: "
                f"group={group_key}, estimated_bytes={estimated}, available={limit}"
            )
        summaries[group_key] = {
            "config_workers": workers, "estimated_working_set_bytes": estimated,
            "memory_limit_bytes": limit, "raw_design_bytes": design_bytes,
            "dynamic_horizon_factor": dynamic_factor,
            "native_allocations_hard_capped": False,
        }
        # No raw data references escape a preflight group.
        runners.clear()
        raw_payload = None
        runner = None
    return summaries


def _acquire_process_lock(lock_file: BinaryIO) -> None:
    if os.name != "nt":
        _process_lock_backend.flock(
            lock_file.fileno(),
            _process_lock_backend.LOCK_EX,
        )
        return
    lock_file.seek(0, os.SEEK_END)
    if lock_file.tell() == 0:
        lock_file.write(b"\0")
        lock_file.flush()
    lock_file.seek(0)
    while True:
        try:
            _process_lock_backend.locking(
                lock_file.fileno(),
                _process_lock_backend.LK_NBLCK,
                1,
            )
            return
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EDEADLK}:
                raise
            sleep(0.05)


def _release_process_lock(lock_file: BinaryIO) -> None:
    if os.name != "nt":
        _process_lock_backend.flock(
            lock_file.fileno(),
            _process_lock_backend.LOCK_UN,
        )
        return
    lock_file.seek(0)
    _process_lock_backend.locking(
        lock_file.fileno(),
        _process_lock_backend.LK_UNLCK,
        1,
    )


@contextmanager
def _batch_state_lock(path: Path) -> Iterator[None]:
    """Serialize one batch state machine across threads and processes."""
    lock_path = path.resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _BATCH_LOCKS_GUARD:
        process_lock = _BATCH_LOCKS.setdefault(lock_path, threading.Lock())
    with process_lock, lock_path.open("a+b") as lock_file:
        _acquire_process_lock(lock_file)
        try:
            yield
        finally:
            _release_process_lock(lock_file)


def _write_state(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_tasks(
    paths: tuple[Path, ...],
    *,
    generators: Mapping[str, Any],
) -> tuple[_BatchTask, ...]:
    tasks = []
    for path in paths:
        loaded = load_yaml_config(path)
        if not isinstance(loaded, ForecastConfigSpec):
            raise TypeError(
                "canonical batch runtime accepts ForecastConfigSpec only; "
                f"got {type(loaded).__name__} for {path}"
            )
        if loaded.strategy is None:
            raise ValueError(f"canonical batch config requires strategy: {path}")
        registry = SourceRegistry(loaded.data, Path.cwd(), generators=generators)
        origin = resolve_origin(registry, loaded.validation.get("forecast_origin"))
        raw_fingerprint = compute_raw_design_fingerprint(
            loaded,
            base_dir=registry._base_dir,
            origin=origin,
            generators=registry._generators,
        )
        tasks.append(
            _BatchTask(
                task_id=_task_id(path),
                path=path,
                config=loaded,
                registry=registry,
                origin=origin,
                raw_fingerprint=raw_fingerprint,
                physical_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    return tuple(tasks)


def _core_artifacts(result) -> dict[str, str]:
    return artifact_paths(result)


def _artifacts_complete(task_state: Mapping[str, Any]) -> bool:
    return artifacts_complete(task_state)


def _annotate_batch_metadata(
    task_state: Mapping[str, Any],
    *,
    batch_id: str,
    group_key: str,
    parent_budget: RuntimeResourceBudget,
    transform_cache: FoldTransformCache,
    peak_rss_bytes: int,
) -> None:
    artifacts = task_state.get("artifacts", {})
    resolved_path = Path(str(artifacts.get("resolved_config", "")))
    if not resolved_path.is_file():
        return
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    runtime = payload.setdefault("runtime", {})
    runtime["batch"] = {
        "schema_version": 1,
        "batch_id": batch_id,
        "task_id": task_state["task_id"],
        "raw_design_group": group_key,
        "parent_budget": parent_budget.payload(),
        "transform_cache": transform_cache.payload(),
        "peak_rss_bytes": peak_rss_bytes,
    }
    temporary = resolved_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved_path)


def _run_canonical_batch_locked(
    config_paths: Iterable[str | Path],
    *,
    output_root: str | Path,
    parent_budget: RuntimeResourceBudget | None = None,
    config_workers: int = 1,
    resume: bool = True,
    generators: Mapping[str, Any] | None = None,
    rss_monitor: SampledRSS | None = None,
) -> CanonicalBatchReport:
    """Run physical model YAMLs by RawDesignGroup with atomic task state."""
    paths = tuple(Path(path).resolve() for path in config_paths)
    if not paths:
        raise ValueError("canonical batch requires at least one config path")
    if len(set(paths)) != len(paths):
        raise ValueError("canonical batch config paths must be unique")
    if isinstance(config_workers, bool) or not isinstance(config_workers, int):
        raise TypeError("config_workers must be a positive integer")
    if config_workers <= 0 or config_workers > len(paths):
        raise ValueError("config_workers must be between 1 and the config count")
    root = Path(output_root).resolve()
    merged_generators = {**BUILTIN_GENERATORS, **(generators or {})}
    tasks = _load_tasks(paths, generators=merged_generators)
    budget = parent_budget or detect_runtime_budget()
    batch_id = _batch_id(paths, root)
    state_path = root / "_batch_state" / f"{batch_id}.json"
    expected_paths = sorted(str(path) for path in paths)

    if resume and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_paths") != expected_paths:
            raise ValueError("batch state config manifest does not match requested paths")
        for task in tasks:
            existing = state.get("tasks", {}).get(task.task_id)
            if not isinstance(existing, Mapping):
                raise ValueError(f"batch state is missing task for {task.path}")
            if (
                existing.get("physical_sha256") != task.physical_sha256
                or existing.get("config_fingerprint") != task.config.fingerprint()
                or existing.get("raw_design_group") != task.raw_fingerprint
            ):
                raise ValueError(
                    f"physical config changed since batch state was created: {task.path}"
                )
    else:
        state = {
            "schema_version": 1,
            "batch_id": batch_id,
            "config_paths": expected_paths,
            "tasks": {
                task.task_id: {
                    "task_id": task.task_id,
                    "path": str(task.path),
                    "config_fingerprint": task.config.fingerprint(),
                    "raw_design_group": task.raw_fingerprint,
                    "physical_sha256": task.physical_sha256,
                    "status": "pending",
                    "attempts": 0,
                }
                for task in tasks
            },
        }
        _write_state(state_path, state)

    checkpoint_root = Path(state.setdefault(
        "checkpoint_root", str(root / "_batch_state" / batch_id / "checkpoints" / uuid4().hex)
    ))
    groups: dict[str, list[_BatchTask]] = {}
    for task in tasks:
        groups.setdefault(task.raw_fingerprint, []).append(task)

    try:
        state["preflight"] = _preflight_groups(groups, state, root, budget, config_workers)
    except Exception as exc:
        for task in tasks:
            task_state = state["tasks"][task.task_id]
            if task_state.get("status") != "completed" or not _artifacts_complete(task_state):
                task_state.update({"status": "failed", "error": {
                    "type": type(exc).__name__, "message": str(exc),
                    "phase": "preflight", "config": str(task.path),
                }})
        state["preflight_error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_state(state_path, state)
        raise
    state.pop("preflight_error", None)
    _write_state(state_path, state)

    process = psutil.Process()
    peak_rss_bytes = process.memory_info().rss
    raw_payload_load_count = 0
    aggregate_transform_hits = 0
    aggregate_transform_misses = 0

    for group_key, group_tasks in groups.items():
        group_runnable: list[tuple[_BatchTask, CanonicalBaseModelRunner]] = []
        pending = [
            task
            for task in group_tasks
            if not (
                state["tasks"][task.task_id].get("status") == "completed"
                and _artifacts_complete(state["tasks"][task.task_id])
            )
        ]
        if not pending:
            continue
        transform_cache = FoldTransformCache(max_bytes=max(1, budget.memory_limit_bytes // 8))
        raw_payload = None
        group_started = perf_counter()
        group_peak_rss = process.memory_info().rss
        effective_workers = min(config_workers, len(pending))
        child_budget = budget.for_children(effective_workers)
        dynamic_runner_factory = _SharedBatchRunnerFactory(
            compiled_cache_root=root,
            resource_budget=child_budget,
            fold_transform_cache=transform_cache,
            checkpoint_root=checkpoint_root,
        )
        runnable: list[tuple[_BatchTask, CanonicalBaseModelRunner]] = []

        for task in pending:
            task_state = state["tasks"][task.task_id]
            task_state.update(
                {
                    "status": "running",
                    "attempts": int(task_state.get("attempts", 0)) + 1,
                }
            )
            task_state.pop("error", None)
            _write_state(state_path, state)
            try:
                if raw_payload is None:
                    runner = CanonicalBaseModelRunner(
                        task.config,
                        task.registry,
                        task.origin,
                        compiled_cache_root=root,
                        resource_budget=budget,
                        fold_transform_cache=transform_cache,
                        checkpoint_root=checkpoint_root,
                    )
                    raw_payload = runner.raw_design_payload()
                    raw_payload_load_count += 1
                else:
                    runner = CanonicalBaseModelRunner(
                        task.config,
                        task.registry,
                        task.origin,
                        compiled_cache_root=root,
                        resource_budget=budget,
                        precompiled_payload=raw_payload,
                        precompiled_fingerprint=group_key,
                        fold_transform_cache=transform_cache,
                        checkpoint_root=checkpoint_root,
                    )
                group_peak_rss = max(group_peak_rss, process.memory_info().rss)
                runnable.append((task, runner))
                group_runnable.append((task, runner))
            except Exception as exc:
                task_state.update(
                    {
                        "status": "failed",
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                )
                _write_state(state_path, state)

        def execute(item: tuple[_BatchTask, CanonicalBaseModelRunner]):
            task, runner = item
            try:
                runner.calendar_runner_factory = dynamic_runner_factory
                result = (
                    runner.run(root)
                    if effective_workers == 1
                    else runner.run_prelimited(root)
                )
                return result, None
            except Exception as exc:
                return None, exc

        if effective_workers == 1:
            outcomes = tuple(execute(item) for item in runnable)
        else:
            _validate_explicit_plans_within_child_budgets(
                tuple(group_runnable),
                config_workers=effective_workers,
            )
            for task, runner in runnable:
                runner.resource_budget = runner.resource_budget.for_children(effective_workers)
                runner.execution_plan = plan_runtime_execution(
                    task.config,
                    runner.workload,
                    budget=runner.resource_budget,
                    base_dir=task.registry._base_dir,
                    feature_schema=runner.builder.feature_schema,
                )
            process_thread_limit = _resolve_config_pool_thread_limit(
                tuple(runner for _, runner in runnable)
            )
            with threadpool_limits(limits=process_thread_limit), ThreadPoolExecutor(
                max_workers=effective_workers
            ) as executor:
                outcomes = tuple(executor.map(execute, runnable))

        for (task, _runner), (result, error) in zip(runnable, outcomes):
            task_state = state["tasks"][task.task_id]
            if error is None:
                if result is None:
                    raise RuntimeError("batch task returned neither result nor error")
                task_state.update({
                    "status": "verifying",
                    "artifacts": _core_artifacts(result),
                    "result_identity": result.run_dir.name,
                })
            else:
                task_state.update(
                    {
                        "status": "failed",
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                            "coordinates": error.as_dict() if isinstance(error, FitCheckpointError) else {
                                "config": str(task.path), "fold": None, "model": None,
                            },
                        },
                    }
                )
            _write_state(state_path, state)

        group_peak_rss = max(
            group_peak_rss, process.memory_info().rss,
            rss_monitor.peak if rss_monitor is not None else 0,
        )
        peak_rss_bytes = max(peak_rss_bytes, group_peak_rss)
        aggregate_transform_hits += transform_cache.hits
        aggregate_transform_misses += transform_cache.misses
        group_metadata = {
            "raw_payload_load_count": 1 if raw_payload is not None else 0,
            "config_workers": effective_workers,
            "transform_cache": transform_cache.payload(),
            "wall_seconds": perf_counter() - group_started,
            "peak_rss_bytes": group_peak_rss,
            "rss_measurement": {"method": "sampled_process_rss", "interval_seconds": 0.02,
                                "scope": "batch_cumulative", "hard_limit": False},
            "memory_admission": state["preflight"].get(group_key),
            "dynamic_raw_cache": dynamic_runner_factory._payloads.payload(),
        }
        state.setdefault("groups", {})[group_key] = group_metadata
        for task, _runner in runnable:
            task_state = state["tasks"][task.task_id]
            if task_state.get("status") == "verifying":
                try:
                    _annotate_batch_metadata(
                        task_state,
                        batch_id=batch_id,
                        group_key=group_key,
                        parent_budget=child_budget,
                        transform_cache=transform_cache,
                        peak_rss_bytes=group_peak_rss,
                    )
                    validate_artifacts(task_state, require_digests=False)
                    task_state["artifact_sha256"] = artifact_digests(task_state["artifacts"])
                    task_state["status"] = "completed"
                except Exception as exc:
                    task_state.update({"status": "failed", "error": {
                        "type": type(exc).__name__, "message": str(exc),
                        "phase": "artifact_verification", "config": str(task.path),
                    }})
        _write_state(state_path, state)

        # Drop strong ownership before constructing the next design group.
        runnable.clear()
        group_runnable.clear()
        outcomes = ()
        raw_payload = None
        runner = None
        _runner = None
        result = None

    statuses = [value.get("status") for value in state["tasks"].values()]
    state["summary"] = {
        "input_count": len(tasks),
        "completed_count": statuses.count("completed"),
        "failed_count": statuses.count("failed"),
        "group_count": len(groups),
        "peak_rss_bytes": peak_rss_bytes,
    }
    _write_state(state_path, state)
    return CanonicalBatchReport(
        input_count=len(tasks),
        completed_count=statuses.count("completed"),
        failed_count=statuses.count("failed"),
        group_count=len(groups),
        raw_payload_load_count=raw_payload_load_count,
        transform_cache={
            "hits": aggregate_transform_hits,
            "misses": aggregate_transform_misses,
        },
        peak_rss_bytes=peak_rss_bytes,
        state_path=state_path,
    )


def run_canonical_batch(
    config_paths: Iterable[str | Path],
    *,
    output_root: str | Path,
    parent_budget: RuntimeResourceBudget | None = None,
    config_workers: int = 1,
    resume: bool = True,
    generators: Mapping[str, Any] | None = None,
) -> CanonicalBatchReport:
    """Run one batch ID under an inter-thread and inter-process lock."""
    paths = tuple(Path(path).resolve() for path in config_paths)
    if not paths:
        raise ValueError("canonical batch requires at least one config path")
    root = Path(output_root).resolve()
    batch_id = _batch_id(paths, root)
    lock_path = root / "_batch_state" / f"{batch_id}.lock"
    with _batch_state_lock(lock_path), SampledRSS() as rss_monitor:
        try:
            return _run_canonical_batch_locked(
                paths,
                output_root=root,
                parent_budget=parent_budget,
                config_workers=config_workers,
                resume=resume,
                generators=generators,
                rss_monitor=rss_monitor,
            )
        except BaseException as exc:
            state_path = lock_path.with_suffix(".json")
            if state_path.is_file():
                state = json.loads(state_path.read_text())
                for task in state.get("tasks", {}).values():
                    if task.get("status") in {"running", "verifying"}:
                        task.update({"status": "failed", "error": {
                            "type": type(exc).__name__, "message": str(exc),
                            "phase": "batch_execution", "config": task.get("path"),
                        }})
                _write_state(state_path, state)
            raise


def verify_batch_results(state_path: str | Path) -> dict[str, int]:
    """Verify manifest cardinality and every completed task's core artifacts."""
    path = Path(state_path)
    state = json.loads(path.read_text(encoding="utf-8"))
    config_paths = state.get("config_paths", [])
    tasks = state.get("tasks", {})
    if not isinstance(config_paths, list) or not isinstance(tasks, dict):
        raise ValueError("batch state is missing config_paths/tasks")
    if len(config_paths) != len(tasks):
        raise ValueError(
            "batch result count differs from input manifest: "
            f"inputs={len(config_paths)} tasks={len(tasks)}"
        )
    if len(set(config_paths)) != len(config_paths):
        raise ValueError("batch state contains duplicate config paths")
    task_paths = [str(task.get("path")) for task in tasks.values()]
    if set(task_paths) != set(config_paths):
        raise ValueError("batch task paths do not match the input manifest")
    completed = [task for task in tasks.values() if task.get("status") == "completed"]
    if len(completed) != len(tasks):
        raise ValueError(
            "batch state is incomplete: "
            f"completed={len(completed)} expected={len(tasks)}"
        )
    invalid = [task["path"] for task in completed if not _artifacts_complete(task)]
    if invalid:
        raise ValueError(f"completed batch tasks have incomplete artifacts: {invalid}")
    return {
        "input_count": len(config_paths),
        "completed_count": len(completed),
        "verified_count": len(completed) - len(invalid),
        "failed_count": sum(task.get("status") == "failed" for task in tasks.values()),
    }


__all__ = [
    "CanonicalBatchReport",
    "run_canonical_batch",
    "verify_batch_results",
]
