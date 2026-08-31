"""OOF generation: shared rolling-origin folds, per-member fit/predict/restore.

The fold geometry reuses `model_testing.validation.rolling_origin_folds` so the
single-model outer backtest and the ensemble OOF share one time contract.
Fusion methods never see this module — they only consume the produced
`OOFPredictionArtifact` (v4 §7.1).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from forecasting_core.artifacts import MarginalForecastDistribution

from model_ensemble.artifacts import OOFPredictionArtifact
from model_ensemble.contracts import BaseModelRunner


def oof_fold_origins(
    runner: BaseModelRunner,
    *,
    fold_count: int,
    stride_steps: int,
    train_window_steps: int,
    gap_steps: int = 0,
    outer_cutoff_origin: pd.Timestamp | None = None,
):
    """Compute OOF holdout folds from the member's supervised origin timeline.

    Folds are the last ``fold_count`` eligible origins spaced ``stride`` apart
    (chronological order), each training on the preceding origins whose labels
    end before the fold label start (same contract as the outer backtest).
    When ``outer_cutoff_origin`` is given, folds whose label range touches or
    crosses it are excluded — the outer backtest never sees labels at/after
    its holdout.
    """
    geometry = runner.geometry
    origins = runner.supervised_origins
    if gap_steps < 0:
        raise ValueError("gap_steps must be non-negative")

    # candidate fold origins: every origin with at least one safe training
    # sample behind it (label_end < label_start), enforcing the strict
    # non-overlap contract; newest-first, then reversed to chronological order
    candidates = []
    for index in range(len(origins) - 1, 0, -1):
        holdout_origin = origins[index]
        holdout_label_start = geometry.label_start(holdout_origin) + (
            gap_steps * geometry.offset if gap_steps else pd.Timedelta(0)
        )
        train_indices = tuple(
            candidate
            for candidate in range(0, index)
            if geometry.label_end(origins[candidate]) < holdout_label_start
        )[-train_window_steps:]
        if not train_indices:
            continue
        if outer_cutoff_origin is not None:
            cutoff_label_start = geometry.label_start(outer_cutoff_origin)
            if geometry.label_end(holdout_origin) >= cutoff_label_start:
                continue
        candidates.append((index, holdout_origin, train_indices))
        if len(candidates) == fold_count * stride_steps:
            break
    if not candidates:
        raise ValueError(
            "ensemble OOF requires at least one fold with non-overlapping "
            "training samples"
        )
    chosen = list(reversed(candidates))[::stride_steps][-fold_count:]
    folds = []
    for fold_number, (index, origin, train_indices) in enumerate(chosen, start=1):
        folds.append(
            {
                "fold": fold_number,
                "origin_index": index,
                "origin": origin,
                "train_indices": train_indices,
            }
        )
    return folds


def generate_oof(
    runners: Mapping[str, BaseModelRunner],
    *,
    fold_count: int,
    stride_steps: int,
    train_window_steps: int,
    gap_steps: int = 0,
    quantile_levels: tuple[float, ...] | None = None,
    outer_cutoff_origin: pd.Timestamp | None = None,
) -> OOFPredictionArtifact:
    """Run every member over the shared folds and collect restored predictions.

    Each fold fits every member independently (own transforms + model) and
    predicts at the fold origin; predictions are restored to the original
    target space before entering the artifact.
    """
    names = tuple(runners)
    if len(names) < 2:
        raise ValueError("OOF requires at least two members")
    first = runners[names[0]]
    folds = oof_fold_origins(
        first,
        fold_count=fold_count,
        stride_steps=stride_steps,
        train_window_steps=train_window_steps,
        gap_steps=gap_steps,
        outer_cutoff_origin=outer_cutoff_origin,
    )

    # every member must share the supervised origin timeline (guaranteed by
    # the loader's problem contract; double-check the geometry explicitly)
    for name, runner in runners.items():
        if runner.supervised_origins != first.supervised_origins:
            raise ValueError(
                f"member {name!r} supervised origin timeline differs from "
                f"{names[0]!r}; OOF folds must be shared"
            )

    per_fold = {name: [] for name in names}
    fold_summaries = []
    for fold in folds:
        times = first.forecast_times(fold["origin"])
        for name in names:
            runner = runners[name]
            scaler, transform, _X, _Y, artifact = runner.fit(
                fold["train_indices"]
            )
            designs, provider = runner.forecast_designs(
                fold["origin"], scaler, transform
            )
            prediction = runner.predict(
                artifact, designs, provider, times, transform
            )
            if isinstance(prediction, MarginalForecastDistribution):
                values = np.asarray(prediction.quantiles.values, dtype=float)
            else:
                values = np.asarray(prediction.values, dtype=float)
            if values.ndim not in (3, 4):
                raise ValueError(
                    f"member {name!r} prediction must be an (N,H,K[,Q]) tensor"
                )
            if values.shape[0] != 1:
                raise ValueError(
                    "OOF generation expects Local members (N=1); Global "
                    "panel members must be split per series before OOF"
                )
            per_fold[name].append(values[0])  # (H,K[,Q]) per OOF sample
        fold_summaries.append(
            {
                "fold": fold["fold"],
                "origin": fold["origin"].isoformat(),
                "label_start": first.geometry.label_start(fold["origin"]).isoformat(),
                "label_end": first.geometry.label_end(fold["origin"]).isoformat(),
                "training_sample_count": len(fold["train_indices"]),
            }
        )

    values_by_member = {
        name: _stack(per_fold[name]) for name in names
    }
    import hashlib
    import json

    fingerprint_payload = json.dumps(
        {
            "members": list(names),
            "folds": fold_summaries,
            "horizon": first.config.problem.horizon,
            "targets": list(first.config.problem.targets),
            "quantiles": list(quantile_levels) if quantile_levels else None,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    oof_fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()

    return OOFPredictionArtifact(
        values_by_member=values_by_member,
        member_order=names,
        targets=tuple(first.config.problem.targets),
        horizon=first.config.problem.horizon,
        quantile_levels=quantile_levels,
        oof_fingerprint=oof_fingerprint,
        folds=tuple(fold_summaries),
        series_ids=tuple(first.series_ids),
    )


def _stack(chunks):
    import numpy as np

    return np.stack([np.asarray(chunk, dtype=float) for chunk in chunks], axis=0)


def actual_for_folds(
    runner: BaseModelRunner,
    folds,
) -> Any:
    """Ground-truth tensor stack aligned with the OOF folds (samples,H,K[,Q])."""
    import numpy as np

    chunks = []
    for fold in folds:
        times = runner.forecast_times(fold["origin"])
        actual = runner.actual(fold["origin_index"], times)
        chunks.append(actual.values[0])  # Local members: drop the N axis
    return np.stack(chunks, axis=0)


__all__ = [
    "actual_for_folds",
    "generate_oof",
    "oof_fold_origins",
]
