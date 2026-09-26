"""Estimator capability declarations and behavioral probes.

2026-09-26 适配层下沉拆分：
- ndarray 合同适配器（``ModelFactoryEstimator``）、工厂、行为探测与
  原生多分位判定下沉至 ``models/adapters/canonical.py``（零合同层依赖）；
- 本模块保留合同层职责：``EstimatorCapabilities`` 类型实例的能力注册表
  （``CapabilityRegistry`` / ``MODEL_FACTORY_CAPABILITY_REGISTRY``）、
  ``resolve_model_capabilities``、依赖 checkpoint 的
  ``SharedMultiQuantilePool``，并 re-export 下沉件以兼容既有导入。
"""

import itertools
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Callable, Sequence

import numpy as np

from forecasting_core.checkpoints import FitCheckpoint
from forecasting_core.specs.estimator import EstimatorCapabilities
from models.adapters.canonical import (
    ModelFactoryEstimator,
    ProbeResult,
    _normalize_model_type,
    make_model_factory,
    probe_native_multioutput,
    supports_native_multi_quantile,
)
from models.catalog import MODEL_CATALOG

__all__ = ["EstimatorCapabilities"]  # 合同类型自 specs 再导出（向后兼容）

# 下沉件的向后兼容别名：旧测试与内部引用使用私有名
_ModelFactoryEstimator = ModelFactoryEstimator


@dataclass(frozen=True, slots=True, init=False)
class CapabilityRegistry:
    capabilities: Mapping[str, EstimatorCapabilities]

    def __init__(
        self,
        registrations: Mapping[str, EstimatorCapabilities]
        | Iterable[tuple[str, EstimatorCapabilities]] = (),
    ) -> None:
        items = registrations.items() if isinstance(registrations, Mapping) else registrations
        normalized: dict[str, EstimatorCapabilities] = {}
        for model_type, capabilities in items:
            normalized_model_type = _normalize_model_type(model_type)
            if not isinstance(capabilities, EstimatorCapabilities):
                raise TypeError("registry values must be EstimatorCapabilities")
            if normalized_model_type in normalized:
                raise ValueError(
                    f"duplicate capability registration: {normalized_model_type!r}"
                )
            normalized[normalized_model_type] = capabilities

        object.__setattr__(
            self,
            "capabilities",
            MappingProxyType(dict(sorted(normalized.items()))),
        )

    def lookup(self, model_type: str) -> EstimatorCapabilities:
        normalized_model_type = _normalize_model_type(model_type)
        try:
            return self.capabilities[normalized_model_type]
        except KeyError as exc:
            raise KeyError(
                f"unknown estimator capabilities: {normalized_model_type!r}"
            ) from exc

    def canonical_payload(self) -> dict[str, dict[str, bool]]:
        return {
            model_type: capabilities.canonical_payload()
            for model_type, capabilities in self.capabilities.items()
        }


_MODEL_FACTORY_TYPES = tuple(MODEL_CATALOG)


MODEL_FACTORY_CAPABILITY_REGISTRY = CapabilityRegistry(
    {
        model_type: EstimatorCapabilities(
            scalar_target=True,
            scalar_quantile=MODEL_CATALOG[model_type].quantile_style is not None,
            native_multi_target_point=False,
            native_multi_target_quantile=False,
            sample_weight=MODEL_CATALOG[model_type].sample_weight,
            categorical=MODEL_CATALOG[model_type].categorical,
            nan_support=MODEL_CATALOG[model_type].nan_support,
        )
        for model_type in _MODEL_FACTORY_TYPES
    }
)


def resolve_model_capabilities(
    model_type: str,
    params: Mapping[str, object] | None = None,
    *,
    feature_names: Sequence[str] | None = None,
    probe_native: bool = False,
) -> EstimatorCapabilities:
    normalized = _normalize_model_type(model_type)
    capabilities = MODEL_FACTORY_CAPABILITY_REGISTRY.lookup(normalized)
    if not probe_native:
        return capabilities
    probe = probe_native_multioutput(
        make_model_factory(
            normalized,
            params,
            # Native multi-output support is independent of the caller's
            # runtime feature width. The behavioral probe owns its synthetic
            # two-column design, so binding the runtime schema here would
            # create a false negative whenever that schema is not width two.
            feature_names=None,
        )
    )
    return replace(
        capabilities,
        native_multi_target_point=probe.supported,
        native_multi_target_quantile=False,
    )


class _QuantileSliceEstimator:
    """共享 booster 的逐 level 视图：fit 委托池（仅首次生效），predict 切列。"""

    def __init__(
        self,
        pool: "SharedMultiQuantilePool",
        position: int,
        level_index: int,
    ) -> None:
        self._pool = pool
        self._position = position
        self._level_index = level_index

    def fit(self, X: object, y: object, sample_weight=None):
        self._pool.fit_position(self._position, X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: object) -> np.ndarray:
        values = self._pool.predict_position(self._position, X)
        return values[:, self._level_index]


class SharedMultiQuantilePool:
    """xgb 原生多分位共享池：每个子模型位置只训练一个 booster。

    对齐不变量：canonical 训练对每个 level 使用同一 ``StrategyTargetPlan``
    （同一 config → 同一调用顺序），因此 ``factory_for_level`` 为每个 level
    返回独立的逻辑位置计数器，按 ``(position)`` 对齐共享 booster——首个
    到达该位置的 level 完成真实训练，后续 level 的 fit 为幂等空操作。

    与 level 并行互斥：共享路径要求逐 level 串行（调用方必须
    ``max_workers=1``），否则位置对齐在多线程下不成立。
    """

    def __init__(
        self,
        model_type: str,
        params: Mapping[str, object] | None,
        levels: Sequence[float],
        feature_names: Sequence[str] | None,
    ) -> None:
        normalized = _normalize_model_type(model_type)
        if not supports_native_multi_quantile(normalized):
            raise ValueError(
                f"model_type {normalized!r} does not support native multi-quantile"
            )
        self.model_type = normalized
        self.levels = tuple(float(level) for level in levels)
        if not self.levels:
            raise ValueError("levels must not be empty")
        self.params = {
            **dict(params or {}),
            "objective": "reg:quantileerror",
            "quantile_alpha": list(self.levels),
        }
        self.feature_names = tuple(feature_names or ())
        self.checkpoint: FitCheckpoint | None = None
        self._fitted: dict[int, ModelFactoryEstimator] = {}
        self._lock = threading.Lock()

    def __getstate__(self) -> dict:
        # 线程锁不可序列化；bundle 部署期只读 pool（不再 fit），重建即可
        state = dict(self.__dict__)
        state["_lock"] = None
        state["checkpoint"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def factory_for_level(self, level_index: int) -> Callable[[], object]:
        if not 0 <= level_index < len(self.levels):
            raise ValueError(
                f"level_index {level_index} out of range for {len(self.levels)} levels"
            )
        position_counter = itertools.count()

        def factory() -> _QuantileSliceEstimator:
            return _QuantileSliceEstimator(
                self,
                next(position_counter),
                level_index,
            )

        return factory

    def fit_position(
        self,
        position: int,
        X: object,
        y: object,
        *,
        sample_weight=None,
    ) -> None:
        with self._lock:
            if position in self._fitted:
                return  # 幂等：该位置已由首个 level 训练
            estimator = ModelFactoryEstimator(
                self.model_type,
                self.params,
                self.feature_names,
            )
            def fit_booster():
                estimator.fit(X, y, sample_weight=sample_weight)
                values = np.asarray(estimator.predict(X), dtype=float)
                if values.ndim != 2 or values.shape[1] != len(self.levels):
                    raise ValueError(
                        "native multi-quantile predict must return "
                        f"(n_samples, {len(self.levels)}); got {values.shape}"
                    )
                return estimator
            if self.checkpoint is None:
                estimator = fit_booster()
            else:
                estimator = self.checkpoint.run(
                    identity={"model": f"shared_quantile/{position}",
                              "levels": self.levels, "feature_schema": self.feature_names,
                              "params": self.params},
                    arrays=(np.asarray(X), np.asarray(y), sample_weight), fit=fit_booster,
                )
            self._fitted[position] = estimator

    def predict_position(self, position: int, X: object) -> np.ndarray:
        try:
            estimator = self._fitted[position]
        except KeyError as exc:
            raise ValueError(
                f"shared quantile position {position} predicted before fit"
            ) from exc
        values = np.asarray(estimator.predict(X), dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.levels):
            raise ValueError(
                "native multi-quantile predict must return "
                f"(n_samples, {len(self.levels)}); got {values.shape}"
            )
        return values
