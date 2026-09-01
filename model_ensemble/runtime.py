"""Reference-based ensemble runtime: YAML -> members -> OOF -> fuser -> persist.

Orchestration only: member lifecycles run through CanonicalBaseModelRunner,
output contracts (long schema, bundle layout, fingerprints) mirror the
single-model canonical runtime so downstream tooling stays uniform (v4 §8).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from model_ensemble import cache as oof_cache
from model_ensemble.loader import (
    load_ensemble_config,
    resolve_members,
    validate_member_sources,
)
from model_ensemble.contracts import BaseModelRunner, EnsembleRuntimeServices
from model_ensemble.artifacts import method_artifact_audit_payload
from model_ensemble.predictor import combine_members
from model_ensemble.specs import EnsembleConfigSpec, EnsembleSpecError
from model_ensemble.trainer import fit_ensemble
from model_testing.backtest import resolve_origin
from data_loading import SourceRegistry
from model_forecasting.results import backtest_tensors_to_long, write_forecast_results
from forecasting_core.specs.config import parse_model_config
from forecasting_core.tensors import (
    MarginalQuantileForecastTensor,
    PointForecastTensor,
)
from forecasting_core.artifacts import ForecastModelBundle, MarginalForecastDistribution


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


def _ensemble_output_paths(
    config: EnsembleConfigSpec,
    output_root: str | Path | None,
) -> tuple[Path, Path, Path, Path]:
    identity = config.output.get("identity", {})
    scenario = str(
        identity.get("scenario_subpath", "canonical")
        if isinstance(identity, Mapping)
        else config.output.get("scenario_subpath", "canonical")
    )
    if not scenario or scenario == "canonical":
        scenario = str(config.output.get("scenario_subpath", "canonical"))
    scenario = scenario.strip("/") or "canonical"
    if output_root is not None:
        run_dir = Path(output_root) / scenario / config.result_identity()
        return (
            run_dir,
            run_dir / "pretrained_models",
            run_dir / "results_test",
            run_dir / "results_forecast",
        )
    directories = config.output.get("directories", {})
    if isinstance(directories, Mapping) and {
        "checkpoints",
        "tests",
        "forecast",
    }.issubset(directories):
        model_dir = Path(str(directories["checkpoints"])) / scenario / config.result_identity()
        test_dir = Path(str(directories["tests"])) / scenario / config.result_identity()
        forecast_dir = Path(str(directories["forecast"])) / scenario / config.result_identity()
        return forecast_dir, model_dir, test_dir, forecast_dir
    root = Path(str(config.output.get("results_root", "results")))
    run_dir = root / scenario / config.result_identity()
    return (
        run_dir,
        run_dir / "pretrained_models",
        run_dir / "results_test",
        run_dir / "results_forecast",
    )


def _combined_forecast(
    config: EnsembleConfigSpec,
    combined: np.ndarray,
    *,
    series_ids: tuple[Any, ...],
    forecast_times: pd.DatetimeIndex,
) -> PointForecastTensor | MarginalForecastDistribution:
    targets = tuple(config.problem.targets)
    if combined.ndim == 3:
        return PointForecastTensor(
            values=combined,
            series_ids=series_ids,
            forecast_times=forecast_times,
            targets=targets,
        )
    if combined.ndim != 4:
        raise ValueError("ensemble forecast must have shape (N,H,K[,Q])")
    levels = tuple(float(value) for value in config.probabilistic.get("quantiles", ()))
    point_level = float(config.probabilistic.get("point_quantile", 0.5))
    quantiles = MarginalQuantileForecastTensor(
        values=combined,
        levels=levels,
        point_level=point_level,
        series_ids=series_ids,
        forecast_times=forecast_times,
        targets=targets,
    )
    return MarginalForecastDistribution(
        point=quantiles.point(),
        quantiles=quantiles,
        dependence_model=None,
        metadata={"ensemble_method": config.method.name},
    )


def _fused_oof_long(
    config: EnsembleConfigSpec,
    artifact: Any,
    oof: Any,
    runner: BaseModelRunner,
) -> pd.DataFrame:
    """Convert fused OOF predictions to the canonical long backtest schema."""
    combined = np.asarray(
        combine_members(artifact, oof.values_by_member),
        dtype=float,
    )
    origin_to_index = {
        origin.isoformat(): index
        for index, origin in enumerate(runner.supervised_origins)
    }
    frames = []
    for sample_index, fold in enumerate(oof.folds):
        origin_index = origin_to_index[str(fold["origin"])]
        origin = runner.supervised_origins[origin_index]
        forecast_times = runner.forecast_times(origin)
        actual = runner.actual(origin_index, forecast_times)
        prediction = _combined_forecast(
            config,
            combined[sample_index : sample_index + 1],
            series_ids=runner.series_ids,
            forecast_times=forecast_times,
        )
        frames.append(
            backtest_tensors_to_long(
                actual,
                prediction,
                window=int(fold.get("fold", sample_index + 1)),
            )
        )
    if not frames:
        raise ValueError("ensemble OOF requires at least one persisted backtest fold")
    return pd.concat(frames, ignore_index=True)


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
    member_root = Path(base_dir).resolve()
    source_root = (
        member_root
        if source_base_dir is None
        else Path(source_base_dir).resolve()
    )
    resolved = resolve_members(config, base_dir=member_root)
    validate_member_sources(config, resolved)
    if len(resolved) < 2:
        raise EnsembleSpecError("ensemble requires at least two valid members")

    member_configs: dict[str, Any] = {}
    runners: dict[str, BaseModelRunner] = {}
    member_fingerprints: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    registry_by_member: dict[str, SourceRegistry] = {}
    for member in config.members:
        member_raw = resolved[member.name]
        member_config = parse_model_config(member_raw, source=member.config_ref)
        member_configs[member.name] = member_config
        member_fingerprints[member.name] = member_config.fingerprint()
        registry = SourceRegistry(member_config.data, source_root)
        registry_by_member[member.name] = registry
        runners[member.name] = services.runner_factory(
            member_config,
            registry,
            resolve_origin(registry, config.validation.get("forecast_origin")),
        )
        for source in member_config.data.sources:
            if source.source_type != "file":
                raise EnsembleSpecError(
                    "ensemble OOF caching supports file sources only; generator "
                    "sources must RAISE (v4 §7.2)"
                )
            for path_role in ("history_path", "backtest_path", "future_path"):
                raw_path = getattr(source, path_role)
                if raw_path is None:
                    continue
                source_hashes[
                    f"{member.name}:{source.name}:{path_role}"
                ] = oof_cache.file_sha256(source_root / raw_path)

    ens_payload = {
        "members": [member.payload() for member in config.members],
        "method": config.method.payload(),
        "problem": config.problem.canonical_payload(),
        "probabilistic": config.probabilistic.canonical_payload(),
    }
    oof_payload = config.oof.payload()
    fingerprint = oof_cache.compute_oof_fingerprint(
        members=member_fingerprints,
        ensemble_payload=ens_payload,
        oof_payload=oof_payload,
        source_hashes=source_hashes,
    )

    run_dir, model_dir, test_dir, forecast_dir = _ensemble_output_paths(
        config,
        output_root,
    )
    cache_root = Path(
        output_root
        if output_root is not None
        else str(config.output.get("results_root", "results"))
    )
    oof = None
    if use_oof_cache:
        try:
            oof = oof_cache.load_oof_cache(cache_root, fingerprint)
        except FileNotFoundError:
            oof = None
        except ValueError:
            raise

    artifact, oof, final_values, member_bundles, audit = fit_ensemble(
        config,
        runners,
        oof=oof,
        outer_cutoff_origin=None,
    )
    artifact = replace(
        artifact,
        oof_fingerprint=fingerprint,
        oof_reference={
            "fingerprint": fingerprint,
            "folds": list(oof.folds),
        },
        config_fingerprint=config.fingerprint(),
    )
    if use_oof_cache:
        oof_for_cache = replace(oof, oof_fingerprint=fingerprint)
        oof_cache.save_oof_cache(cache_root, oof_for_cache)

    combined = combine_members(artifact, final_values)
    first_runner = runners[config.members[0].name]
    forecast_times = first_runner.forecast_times(first_runner.origin)
    forecast = _combined_forecast(
        config,
        combined,
        series_ids=first_runner.series_ids,
        forecast_times=forecast_times,
    )
    first_bundle = member_bundles[config.members[0].name]
    source_lineage = tuple(
        {"member": member_name, **item}
        for member_name, member_bundle in member_bundles.items()
        for item in member_bundle.source_lineage
    )
    ensemble_spec = {
        **config.payload(),
        "oof_fingerprint": fingerprint,
        "folds": list(oof.folds),
        "method_artifact": method_artifact_audit_payload(
            artifact.method_artifact
        ),
    }
    bundle = ForecastModelBundle(
        schema_version=2,
        model={
            "ensemble_artifact": artifact,
            "member_bundles": member_bundles,
        },
        feature_scaler=None,
        target_transform=None,
        selected_features=(),
        input_schema={
            "members": list(artifact.member_order),
            "panel": {
                "series_id_cols": list(config.problem.series_id_cols),
                "known_series_ids": [
                    list(value) if isinstance(value, tuple) else value
                    for value in first_runner.series_ids
                ],
                "unknown_series_policy": "raise",
            },
        },
        probabilistic_spec=first_bundle.probabilistic_spec,
        model_type="ensemble",
        pred_method=None,
        canonical_problem=config.problem.canonical_payload(),
        strategy_spec=None,
        ensemble_spec=ensemble_spec,
        estimator_spec=None,
        dimensions=(
            int(combined.shape[0]),
            int(combined.shape[1]),
            int(combined.shape[2]),
        ),
        series_ids=first_runner.series_ids,
        target_order=tuple(config.problem.targets),
        feature_lineage=(),
        source_lineage=source_lineage,
        training_scope=config.problem.training_scope,
        result_schema_version=2,
        config_fingerprint=config.fingerprint(),
    )

    services.persist_bundle(bundle, model_dir)
    write_forecast_results(forecast_dir, forecast)
    forecast_dir.mkdir(parents=True, exist_ok=True)
    (forecast_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                **config.canonical_payload(),
                "config_fingerprint": config.fingerprint(),
                "runtime": {
                    "oof_fingerprint": fingerprint,
                    "member_order": list(artifact.member_order),
                    "folds": list(oof.folds),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    test_dir.mkdir(parents=True, exist_ok=True)
    _fused_oof_long(config, artifact, oof, first_runner).to_csv(
        test_dir / "cv_plot_df.csv",
        index=False,
        encoding="utf_8_sig",
    )
    fused_scores = audit.get("fused_oof_scores") or {}
    point_scores = fused_scores.get("point")
    if point_scores is not None:
        point_scores.to_csv(
            test_dir / "test_scores_df.csv",
            index=False,
            encoding="utf_8_sig",
        )
    probabilistic_scores = fused_scores.get("probabilistic")
    if probabilistic_scores is not None:
        probabilistic_scores.to_csv(
            test_dir / "test_scores_probabilistic_df.csv",
            index=False,
            encoding="utf_8_sig",
        )
    (test_dir / "result_metadata.json").write_text(
        json.dumps(
            {
                "result_schema_version": 2,
                "method": config.method.name,
                "oof_fingerprint": fingerprint,
                "folds": list(oof.folds),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "artifact": artifact,
        "bundle": bundle,
        "run_dir": run_dir,
        "model_dir": model_dir,
        "test_dir": test_dir,
        "forecast_dir": forecast_dir,
        "oof": oof,
        "oof_fingerprint": fingerprint,
        "member_bundles": member_bundles,
        "member_final_values": final_values,
        "combined_values": combined,
        "fused_oof_scores": fused_scores,
        "forecast_times": forecast_times,
        "audit": audit,
    }


def _member_origin(member_config, registry):
    return resolve_origin(registry, member_config.validation.get("forecast_origin"))


__all__ = ["run_ensemble_config", "run_ensemble_config_file"]
