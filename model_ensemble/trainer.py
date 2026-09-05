"""Ensemble trainer: OOF generation, fuser fitting, member final refits (v4 §7).

The trainer consumes only `BaseModelRunner` facades — one per member — and the
resolved `EnsembleConfigSpec`; it never touches runtime private helpers.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from model_ensemble.artifacts import (
    EnsembleArtifact,
    MethodArtifact,
    OOFPredictionArtifact,
)
from model_ensemble.contracts import BaseModelRunner, member_execution_evidence
from model_ensemble.evaluation import evaluate_fused_oof
from model_ensemble.methods import averaging, linear_blending, stacking, weighted
from model_ensemble.oof import actual_for_folds, generate_oof
from model_ensemble.specs import EnsembleConfigSpec
from forecasting_core.artifacts import ForecastModelBundle, MarginalForecastDistribution

METHOD_IMPLEMENTATIONS: dict[str, Any] = {
    averaging.METHOD_NAME: averaging,
    weighted.METHOD_NAME: weighted,
    linear_blending.METHOD_NAME: linear_blending,
    stacking.METHOD_NAME: stacking,
}


def _quantile_levels_for_config(
    config: EnsembleConfigSpec,
) -> tuple[float, ...] | None:
    mode = str(config.probabilistic.get("mode", "point"))
    if mode != "quantile":
        return None
    if config.method.name == "stacking":
        raise ValueError(
            "stacking is point-only in the first version; quantile "
            "stacking directly RAISEs (v4 §3)"
        )
    return tuple(float(level) for level in config.probabilistic["quantiles"])


def generate_oof_for_config(
    config: EnsembleConfigSpec,
    runners: Mapping[str, BaseModelRunner],
    *,
    outer_cutoff_origin: pd.Timestamp | None = None,
    member_workers: int = 1,
) -> OOFPredictionArtifact:
    """Generate member OOF predictions using one ensemble config contract."""
    return generate_oof(
        runners,
        fold_count=config.oof.fold_count,
        stride_steps=config.oof.stride_steps,
        train_window_steps=config.oof.train_window_steps,
        gap_steps=config.oof.gap_steps,
        quantile_levels=_quantile_levels_for_config(config),
        outer_cutoff_origin=outer_cutoff_origin,
        member_workers=member_workers,
    )


@dataclass(frozen=True, slots=True)
class MemberAuditScores:
    """Per-member OOF quality: RMSE/MAE per target (v4 §8.1 audit outputs)."""

    scores_by_member: dict[str, dict[str, dict[str, float]]]


def _member_oof_scores(
    oof: OOFPredictionArtifact, actual: np.ndarray
) -> dict[str, dict[str, dict[str, float]]]:
    scores: dict[str, dict[str, dict[str, float]]] = {}
    quantile = oof.quantile_levels is not None
    for name in oof.member_order:
        values = oof.values_by_member[name]
        per_target: dict[str, dict[str, float]] = {}
        for index, target in enumerate(oof.targets):
            member_values = values[:, :, index]
            target_actual = actual[:, :, index]
            if quantile:
                # pinball-averaged absolute error across levels
                levels = np.asarray(oof.quantile_levels, dtype=float)
                err = member_values - target_actual[:, :, None]
                losses = np.maximum(levels * err, (levels - 1.0) * err)
                per_target[target] = {
                    "pinball": float(np.mean(losses)),
                    "mae": float(np.mean(np.abs(err))),
                }
            else:
                err = member_values - target_actual
                per_target[target] = {
                    "rmse": float(np.sqrt(np.mean(np.square(err)))),
                    "mae": float(np.mean(np.abs(err))),
                }
        scores[name] = per_target
    return scores


def fit_ensemble(
    config: EnsembleConfigSpec,
    runners: Mapping[str, BaseModelRunner],
    *,
    oof: OOFPredictionArtifact | None = None,
    outer_cutoff_origin: pd.Timestamp | None = None,
    member_workers: int = 1,
) -> tuple[
    EnsembleArtifact,
    OOFPredictionArtifact,
    dict[str, np.ndarray],
    dict[str, ForecastModelBundle],
    dict[str, Any],
]:
    """Generate (or reuse) OOF, fit the fuser, refit members on full history.

    Returns ``(EnsembleArtifact, oof, final_member_predictions_by_name,
    audit_payload)`` where final member predictions are produced at the
    ensemble forecast origin with full-history fits.
    """
    impls = METHOD_IMPLEMENTATIONS
    method_name = config.method.name
    impl = impls[method_name]
    names = tuple(runners)
    if set(names) != {member.name for member in config.members}:
        raise ValueError("runners do not match the ensemble member references")

    first = runners[names[0]]
    quantile_levels = _quantile_levels_for_config(config)

    if oof is None:
        oof = generate_oof_for_config(
            config,
            runners,
            outer_cutoff_origin=outer_cutoff_origin,
            member_workers=member_workers,
        )
    folds = oof.folds
    origin_to_index = {
        origin.isoformat(): index
        for index, origin in enumerate(first.supervised_origins)
    }
    actual = actual_for_folds(first, [
        {
            "origin_index": origin_to_index[fold["origin"]],
            "origin": pd.Timestamp(fold["origin"]),
        }
        for fold in folds
    ])
    oof_values = {
        name: oof.values_by_member[name] for name in oof.member_order
    }

    if method_name == "averaging":
        method_artifact: MethodArtifact = impl.fit_averaging(oof_values, actual)
    elif method_name == "weighted":
        method_artifact = impl.fit_weighted(
            oof_values,
            actual,
            metric=str(config.method.params.get("metric", "rmse")),
        )
    elif method_name == "linear_blending":
        method_artifact = impl.fit_linear_blending(
            oof_values,
            actual,
            quantile_levels=quantile_levels,
        )
    elif method_name == "stacking":
        method_artifact = impl.fit_stacking(oof_values, actual)
    else:  # defensive; specs already restrict names
        raise ValueError(f"unsupported ensemble method: {method_name!r}")

    # final refits: every member on the full visible history, predict at the
    # ensemble forecast origin, restored to original target space
    origin = first.origin
    times = first.forecast_times(origin)
    def final_refit(name: str):
        runner = runners[name]
        try:
            scaler, transform, X_all, Y_all = runner.final_bundle_inputs()
            trainer, artifact, capabilities = runner.fit_final(X_all, Y_all)
            designs, provider = runner.forecast_designs(origin, scaler, transform)
            prediction = runner.predict(artifact, designs, provider, times, transform)
            values = (
                prediction.quantiles.values
                if isinstance(prediction, MarginalForecastDistribution)
                else prediction.values
            )
            bundle = runner.build_final_bundle(
                scaler,
                transform,
                trainer,
                artifact,
                capabilities,
            )
            return values, bundle, member_execution_evidence(runner, artifact, transform)
        except Exception as exc:
            raise RuntimeError(f"final refit member={name!r} failed") from exc

    if member_workers > 1:
        with ThreadPoolExecutor(max_workers=member_workers) as executor:
            final_results = tuple(executor.map(final_refit, names))
    else:
        final_results = tuple(final_refit(name) for name in names)
    final_values = {
        name: values for name, (values, _bundle, _evidence) in zip(names, final_results)
    }
    member_bundles = {
        name: bundle for name, (_values, bundle, _evidence) in zip(names, final_results)
    }

    member_scores = _member_oof_scores(oof, actual)

    ens_artifact = EnsembleArtifact(
        method_artifact=method_artifact,
        member_order=tuple(oof.member_order),
        targets=tuple(oof.targets),
        horizon=oof.horizon,
        quantile_levels=oof.quantile_levels,
        oof_fingerprint=oof.oof_fingerprint,
        fold_summary=tuple(oof.folds),
    )
    # 融合 OOF 评分（2026-08-30 遗留收口）：ensemble 的无泄漏质量证据，
    # point/quantile 均产出；quantile 模式附带 pinball/central 区间指标。
    fused_scores = evaluate_fused_oof(
        ens_artifact,
        oof,
        actual,
        point_level=float(config.probabilistic.get("point_quantile", 0.5)),
    )
    evidence_complete = bool(oof.execution_evidence) and len(oof.execution_evidence) == len(folds) and all(
        set(row.get("members", {})) == set(names)
        and all(item.get("status") == "recorded" for item in row["members"].values())
        for row in oof.execution_evidence
    )
    audit = {
        "run_evidence": {
            "member_final": {name: evidence for name, (_values, _bundle, evidence) in zip(names, final_results)},
            "member_oof": {"status": "recorded" if evidence_complete else "unavailable",
                           "reason": None if evidence_complete else "missing_or_incomplete_historical_member_evidence",
                           "folds": list(oof.execution_evidence)},
        },
        "member_scores": member_scores,
        "fused_oof_scores": fused_scores,
        "method": method_name,
        "oof_fingerprint": oof.oof_fingerprint,
        "fold_count": len(folds),
    }

    return ens_artifact, oof, final_values, member_bundles, audit


__all__ = ["fit_ensemble", "generate_oof_for_config", "MemberAuditScores"]
