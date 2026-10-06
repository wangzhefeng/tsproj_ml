"""逐 horizon 权重的纯数组适配；不持有成员模型或时间切分。"""
from typing import Callable, Mapping

import numpy as np

from model_ensemble.artifacts import HorizonWeightsArtifact, PerTargetWeightsArtifact
from model_ensemble.methods.weighted import combine_weighted


def fit_horizon_weights(
    fit: Callable[..., PerTargetWeightsArtifact],
    member_values: Mapping[str, np.ndarray],
    actual: np.ndarray,
    **kwargs,
) -> HorizonWeightsArtifact:
    artifacts = tuple(
        fit({name: values[:, step:step + 1] for name, values in member_values.items()},
            actual[:, step:step + 1], **kwargs)
        for step in range(actual.shape[1])
    )
    return HorizonWeightsArtifact(artifacts[0].method_name, artifacts)


def combine_horizon_weights(artifact: HorizonWeightsArtifact, member_values: Mapping[str, np.ndarray]) -> np.ndarray:
    if any(values.shape[1] != len(artifact.artifacts_by_horizon) for values in member_values.values()):
        raise ValueError("horizon weights do not match prediction horizon")
    return np.concatenate([
        combine_weighted(method, {name: values[:, step:step + 1] for name, values in member_values.items()})
        for step, method in enumerate(artifact.artifacts_by_horizon)
    ], axis=1)
