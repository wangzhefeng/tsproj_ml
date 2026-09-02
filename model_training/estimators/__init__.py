"""Estimator capability primitives."""

from model_training.estimators.capabilities import (
    CapabilityRegistry,
    EstimatorCapabilities,
    MODEL_FACTORY_CAPABILITY_REGISTRY,
    ProbeResult,
    SharedMultiQuantilePool,
    make_model_factory,
    probe_native_multioutput,
    resolve_model_capabilities,
    supports_native_multi_quantile,
)
from model_training.estimators.multi_target import (
    IndependentMultiTargetAdapter,
    NativeMultiTargetAdapter,
    RegressorChainMultiTargetAdapter,
)

__all__ = [
    "EstimatorCapabilities",
    "IndependentMultiTargetAdapter",
    "NativeMultiTargetAdapter",
    "CapabilityRegistry",
    "MODEL_FACTORY_CAPABILITY_REGISTRY",
    "ProbeResult",
    "RegressorChainMultiTargetAdapter",
    "SharedMultiQuantilePool",
    "make_model_factory",
    "probe_native_multioutput",
    "resolve_model_capabilities",
    "supports_native_multi_quantile",
]
