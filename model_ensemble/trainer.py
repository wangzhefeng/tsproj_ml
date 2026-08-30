"""Ensemble trainer: OOF generation, fuser fitting, member final refits (v4 §7).

The trainer consumes only `BaseModelRunner` facades — one per member — and the
resolved `EnsembleConfigSpec`; it never touches runtime private helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from model_ensemble.artifacts import (
    EnsembleArtifact,
    MethodArtifact,
    OOFPredictionArtifact,
)
from model_ensemble.contracts import BaseModelRunner
from model_ensemble.evaluation import evaluate_fused_oof
from model_ensemble.oof import actual_for_folds, generate_oof
from model_ensemble.specs import EnsembleConfigSpec
from probabilistic.types import MarginalForecastDistribution

METHOD_IMPLEMENTATIONS: dict[str, Any] = {}


def _implementations() -> dict[str, Any]:
    global METHOD_IMPLEMENTATIONS
    if not METHOD_IMPLEMENTATIONS:
        from model_ensemble.methods import averaging, linear_blending, stacking, weighted

        METHOD_IMPLEMENTATIONS = {
            averaging.METHOD_NAME: averaging,
            weighted.METHOD_NAME: weighted,
            linear_blending.METHOD_NAME: linear_blending,
            stacking.METHOD_NAME: stacking,
        }
    return METHOD_IMPLEMENTATIONS


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
) -> tuple[EnsembleArtifact, OOFPredictionArtifact, np.ndarray, dict[str, Any]]:
    """Generate (or reuse) OOF, fit the fuser, refit members on full history.

    Returns ``(EnsembleArtifact, oof, final_member_predictions_by_name,
    audit_payload)`` where final member predictions are produced at the
    ensemble forecast origin with full-history fits.
    """
    impls = _implementations()
    method_name = config.method.name
    impl = impls[method_name]
    names = tuple(runners)
    if set(names) != {member.name for member in config.members}:
        raise ValueError("runners do not match the ensemble member references")

    first = runners[names[0]]
    quantile_levels = None
    mode = str(config.probabilistic.get("mode", "point"))
    if mode == "quantile":
        quantile_levels = tuple(
            float(level) for level in config.probabilistic["quantiles"]
        )
        if method_name == "stacking":
            raise ValueError(
                "stacking is point-only in the first version; quantile "
                "stacking directly RAISEs (v4 §3)"
            )

    if oof is None:
        oof = generate_oof(
            runners,
            fold_count=config.oof.fold_count,
            stride=config.oof.stride,
            train_window_length=config.oof.train_window_length,
            gap_steps=config.oof.gap_steps,
            quantile_levels=quantile_levels,
            outer_cutoff_origin=outer_cutoff_origin,
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
        method_artifact = impl.fit_linear_blending(oof_values, actual)
    elif method_name == "stacking":
        method_artifact = impl.fit_stacking(oof_values, actual)
    else:  # defensive; specs already restrict names
        raise ValueError(f"unsupported ensemble method: {method_name!r}")

    # final refits: every member on the full visible history, predict at the
    # ensemble forecast origin, restored to original target space
    origin = first.origin
    times = first.forecast_times(origin)
    final_values: dict[str, np.ndarray] = {}
    for name in names:
        runner = runners[name]
        scaler, transform, _X, _Y, artifact = runner.fit(
            tuple(range(len(runner.supervised_origins)))
        )
        designs, provider = runner.forecast_designs(origin, scaler, transform)
        prediction = runner.predict(artifact, designs, provider, times, transform)
        if isinstance(prediction, MarginalForecastDistribution):
            final_values[name] = prediction.quantiles.values
        else:
            final_values[name] = prediction.values

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
    audit = {
        "member_scores": member_scores,
        "fused_oof_scores": fused_scores,
        "method": method_name,
        "oof_fingerprint": oof.oof_fingerprint,
        "fold_count": len(folds),
    }

    return ens_artifact, oof, final_values, audit


__all__ = ["fit_ensemble", "MemberAuditScores"]
