"""Reference-based ensemble runtime: YAML -> members -> OOF -> fuser -> persist.

Orchestration only: member lifecycles run through CanonicalBaseModelRunner,
output contracts (long schema, bundle layout, fingerprints) mirror the
single-model canonical runtime so downstream tooling stays uniform (v4 §8).
"""

from __future__ import annotations


from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any
from threadpoolctl import threadpool_limits

from model_ensemble.outputs import cache as oof_cache
from model_ensemble.configuration.loader import (
    load_ensemble_config,
    resolve_members,
    validate_member_sources,
)
from model_ensemble.contracts import BaseModelRunner, EnsembleRuntimeServices

from model_ensemble.inference.predictor import combine_members
from model_ensemble.configuration.specs import EnsembleConfigSpec, EnsembleSpecError
from model_ensemble.training.trainer import fit_ensemble, generate_oof_for_config
from model_ensemble.training.backtesting import run_outer_backtest
from model_ensemble.inference.forecast import build_ensemble_forecast
from model_ensemble.outputs.persistence import RunCompletion, build_ensemble_bundle
from model_ensemble.configuration.preflight import parse_member_configs
from model_ensemble.outputs.reporting import ensemble_output_paths, write_predictions_and_scores, write_run_metadata
from forecasting_core.temporal.origin import resolve_origin
from forecasting_core.probability.spec import probabilistic_spec_from_mapping
from data_loading import SourceRegistry

from model_predicting.loops.deployment import attach_bundle_prediction_intervals

from forecasting_core.probability.distribution import MarginalForecastDistribution


def run_ensemble_config_file(
    config_path: str | Path,
    output_root: str | Path | None = None,
    *,
    services: EnsembleRuntimeServices,
) -> Any:
    config = load_ensemble_config(config_path)
    return run_ensemble_config(
        config,
        output_root=output_root,
        base_dir=Path(config_path).resolve().parent,
        source_base_dir=Path.cwd(),
        services=services,
    )


def run_ensemble_config(
    config: EnsembleConfigSpec,
    output_root: str | Path | None = None,
    *,
    base_dir: str | Path = ".",
    source_base_dir: str | Path | None = None,
    use_oof_cache: bool = True,
    services: EnsembleRuntimeServices,
) -> Any:
    """Execute one reference-based ensemble configuration end to end."""
    completion = RunCompletion()
    try:
        result = _execute_ensemble_config(config, output_root, base_dir=base_dir,
                                          source_base_dir=source_base_dir, use_oof_cache=use_oof_cache,
                                          services=services, completion=completion)
        completion.complete(result)
        return result
    except Exception as exc:
        completion.fail(exc)
        raise


def _execute_ensemble_config(config, output_root, *, base_dir, source_base_dir, use_oof_cache, services, completion):
    lifecycle_started = perf_counter()
    raw_design_started = perf_counter()
    member_root = Path(base_dir).resolve()
    source_root = (
        member_root
        if source_base_dir is None
        else Path(source_base_dir).resolve()
    )
    cache_root = Path(
        output_root
        if output_root is not None
        else str(config.output.get("results_root", "results"))
    )
    resolved = resolve_members(config, base_dir=member_root)
    validate_member_sources(config, resolved)
    if len(resolved) < 2:
        raise EnsembleSpecError("ensemble requires at least two valid members")
    parent_budget = (
        services.resolve_budget(config)
        if services.resolve_budget is not None
        else None
    )
    member_budget = (
        replace(
            parent_budget,
            memory_limit_bytes=max(
                1,
                parent_budget.memory_limit_bytes // len(resolved),
            ),
            source=f"{parent_budget.source}+ensemble_member_share",
        )
        if parent_budget is not None
        else None
    )

    member_configs = parse_member_configs(config, resolved)
    runners: dict[str, BaseModelRunner] = {}
    member_fingerprints: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    registry_by_member: dict[str, SourceRegistry] = {}
    for member in config.members:
        member_config = member_configs[member.name]
        member_fingerprints[member.name] = member_config.fingerprint()
        registry = SourceRegistry(member_config.data, source_root)
        registry_by_member[member.name] = registry
        runner_kwargs: dict[str, Any] = {}
        if member_budget is not None:
            runner_kwargs["resource_budget"] = member_budget
        runners[member.name] = services.runner_factory(
            member_config,
            registry,
            resolve_origin(registry, config.validation.get("forecast_origin")),
            **runner_kwargs,
        )
        support_check = getattr(runners[member.name], "validate_ensemble_support", None)
        if callable(support_check):
            support_check()
        share_design = getattr(runners[member.name], "share_training_design", None)
        if callable(share_design):
            for previous_name, previous_runner in runners.items():
                if previous_name != member.name and share_design(previous_runner):
                    break
        if parent_budget is not None and sum(
            runner.workload.design_bytes for runner in runners.values()
        ) > parent_budget.memory_limit_bytes:
            raise ValueError(
                "ensemble member designs exceed the parent memory budget"
            )
        source_hashes.update(oof_cache.member_source_hashes(
            member.name, member_config.data, source_root, registry.generators,
        ))

    plan_kwargs = {"budget": parent_budget} if parent_budget is not None else {}
    resource_workload, resource_budget, execution_plan = services.plan_resources(
        config,
        runners,
        **plan_kwargs,
    )
    if execution_plan.selected_axis == "ensemble_member":
        for runner in runners.values():
            runner.apply_ensemble_member_plan(execution_plan)
    stage_wall_seconds = {
        "raw_design": perf_counter() - raw_design_started,
    }

    oof_semantics_payload = {
        "member_order": [member.name for member in config.members],
        "problem": config.problem.canonical_payload(),
        "probabilistic": config.probabilistic.canonical_payload(),
        "forecast_origins": {name: runner.origin.isoformat() for name, runner in runners.items()},
    }
    oof_payload = config.oof.payload()
    fingerprint = oof_cache.compute_oof_fingerprint(
        members=member_fingerprints,
        ensemble_payload=oof_semantics_payload,
        oof_payload=oof_payload,
        source_hashes=source_hashes,
    )

    origins = {runner.origin for runner in runners.values()}
    if len(origins) != 1:
        raise EnsembleSpecError("ensemble members must resolve to the same forecast origin")
    document_fingerprint = config.fingerprint()
    config = replace(
        config,
        resolved_member_fingerprints=tuple(member_fingerprints.items()),
        resolved_origin=next(iter(origins)).isoformat(),
    )
    run_dir, model_dir, test_dir, forecast_dir = ensemble_output_paths(
        config,
        output_root,
    )
    completion.start(model_dir, config.fingerprint())
    ensemble_fit_started = perf_counter()
    oof = None
    oof_cache_hit = False
    probability = probabilistic_spec_from_mapping(config.probabilistic.canonical_payload())
    calibration_tracker = None
    if probability.calibration is not None:
        if services.calibration_factory is None:
            raise EnsembleSpecError("fusion calibration requires an injected calibration_factory")
        calibration_tracker = services.calibration_factory(probability, freq_offset=next(iter(runners.values())).geometry.offset)
    def inner_oof(cutoff):
        key = oof_cache.compute_oof_fingerprint(
            members=member_fingerprints,
            ensemble_payload={**oof_semantics_payload, "outer_cutoff_origin": cutoff.isoformat()},
            oof_payload=oof_payload, source_hashes=source_hashes,
        )
        def generate():
            return generate_oof_for_config(config, runners, outer_cutoff_origin=cutoff, member_workers=execution_plan.member_workers)
        if not use_oof_cache:
            return generate()
        return oof_cache.get_or_create_oof_cache(cache_root, key, generate)[0]

    with threadpool_limits(limits=execution_plan.model_threads):
        backtest = run_outer_backtest(config, runners, get_oof=inner_oof, calibration_tracker=calibration_tracker)
        if use_oof_cache:
            oof, oof_cache_hit = oof_cache.get_or_create_oof_cache(
                cache_root,
                fingerprint,
                lambda: generate_oof_for_config(
                    config,
                    runners,
                    member_workers=execution_plan.member_workers,
                ),
            )
        artifact, oof, final_values, member_bundles, audit = fit_ensemble(
            config,
            runners,
            oof=oof,
            outer_cutoff_origin=None,
            member_workers=execution_plan.member_workers,
        )
    stage_wall_seconds["ensemble_fit"] = perf_counter() - ensemble_fit_started
    audit["run_evidence"]["member_oof"]["cache_hit"] = oof_cache_hit
    audit["run_evidence"]["source_hashes"] = dict(source_hashes)
    persist_started = perf_counter()
    artifact = replace(
        artifact,
        oof_fingerprint=fingerprint,
        oof_reference={
            "fingerprint": fingerprint,
            "folds": list(oof.folds),
        },
        config_fingerprint=config.fingerprint(),
    )
    combined = combine_members(artifact, final_values)
    first_runner = runners[config.members[0].name]
    forecast_times = first_runner.forecast_times(first_runner.origin)
    forecast = build_ensemble_forecast(
        combined, probability=probability, targets=config.problem.targets, method_name=config.method.name,
        series_ids=first_runner.series_ids,
        forecast_times=forecast_times,
    )
    combined = forecast.quantiles.values if isinstance(forecast, MarginalForecastDistribution) else forecast.values
    calibration_state = None
    if calibration_tracker is not None:
        correction, calibration_audit = calibration_tracker.final_correction(first_runner.origin)
        calibration_state = {
            "method": "cqr", **calibration_audit, "evaluation_role": "outer_holdout",
            "forecast_origin": first_runner.origin.isoformat(),
            "label_available_at_max": max((record.label_available_at.isoformat() for record in correction.selected_records), default=None),
        }
    bundle = build_ensemble_bundle(
        config, artifact, member_bundles, series_ids=first_runner.series_ids,
        dimensions=tuple(int(value) for value in combined.shape[:3]), calibration_state=calibration_state,
    )

    services.persist_bundle(bundle, model_dir)
    if isinstance(forecast, MarginalForecastDistribution):
        attach_bundle_prediction_intervals(bundle, forecast)
    write_predictions_and_scores(config, forecast=forecast, runner=first_runner, backtest=backtest,
                                 artifact=artifact, oof=oof, audit=audit, forecast_dir=forecast_dir, test_dir=test_dir)
    stage_wall_seconds["persist"] = perf_counter() - persist_started
    stage_wall_seconds["total"] = perf_counter() - lifecycle_started
    runtime_resources = {
        "workload": resource_workload.payload(),
        "budget": resource_budget.payload(),
        "execution_plan": execution_plan.payload(),
        "member_execution_plans": {
            name: runner.execution_plan.payload()
            for name, runner in runners.items()
        },
        "member_resources": {
            name: runner.runtime_resources_payload()
            for name, runner in runners.items()
        },
        "cache": {
            "oof_hit": oof_cache_hit,
            "oof_fingerprint": fingerprint,
        },
        "stage_wall_seconds": stage_wall_seconds,
    }
    write_run_metadata(config, document_fingerprint=document_fingerprint, artifact=artifact, oof=oof,
                       audit=audit, cache_hit=oof_cache_hit, resources=runtime_resources,
                       forecast_dir=forecast_dir, test_dir=test_dir)
    fused_scores = audit.get("fused_oof_scores") or {}

    return {
        "artifact": artifact,
        "config": config,
        "backtest": backtest,
        "document_fingerprint": document_fingerprint,
        "bundle": bundle,
        "run_dir": run_dir,
        "model_dir": model_dir,
        "test_dir": test_dir,
        "forecast_dir": forecast_dir,
        "oof": oof,
        "oof_fingerprint": fingerprint,
        "oof_cache_hit": oof_cache_hit,
        "resource_workload": resource_workload,
        "resource_budget": resource_budget,
        "execution_plan": execution_plan,
        "member_bundles": member_bundles,
        "member_final_values": final_values,
        "combined_values": combined,
        "fused_oof_scores": fused_scores,
        "forecast_times": forecast_times,
        "audit": audit,
    }


__all__ = ["run_ensemble_config", "run_ensemble_config_file"]
