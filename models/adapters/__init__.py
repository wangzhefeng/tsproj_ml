"""canonical 合同 estimator 适配层与原生序列模型注册表。"""

from models.adapters.canonical import (
    ModelFactoryEstimator,
    ProbeResult,
    make_model_factory,
    probe_native_multioutput,
    supports_native_multi_quantile,
)
from models.adapters.native_registry import (
    NATIVE_HISTORY_MODELS,
    native_history_cls,
)

__all__ = [
    "ModelFactoryEstimator",
    "ProbeResult",
    "NATIVE_HISTORY_MODELS",
    "make_model_factory",
    "native_history_cls",
    "probe_native_multioutput",
    "supports_native_multi_quantile",
]
