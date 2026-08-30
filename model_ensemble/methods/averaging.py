"""averaging: unweighted member mean (v4 §3). Hard baseline of the redesign."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from model_ensemble.artifacts import EqualWeightsArtifact

METHOD_NAME = "averaging"


def fit_averaging(
    oof_values_by_member: Mapping[str, np.ndarray],
    actual: np.ndarray,
) -> EqualWeightsArtifact:
    return EqualWeightsArtifact(method_name=METHOD_NAME)


def combine_averaging(
    method_artifact: EqualWeightsArtifact,
    member_values: Mapping[str, np.ndarray],
) -> np.ndarray:
    if not member_values:
        raise ValueError("averaging requires at least two members")
    stacked = np.stack(
        [np.asarray(value, dtype=float) for value in member_values.values()],
        axis=0,
    )
    return stacked.mean(axis=0)


__all__ = ["METHOD_NAME", "combine_averaging", "fit_averaging"]
