"""Estimator capability declarations and behavioral probes."""

import copy
import importlib
import itertools
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from functools import partial
from types import MappingProxyType
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone

from forecasting_core.specs.estimator import EstimatorCapabilities

__all__ = ["EstimatorCapabilities"]  # 合同类型自 specs 再导出（向后兼容）


def _normalize_model_type(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("model_type must be a string")
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError("model_type must not be blank")
    return normalized


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


@dataclass(frozen=True, slots=True)
class ProbeResult:
    supported: bool
    reason: str | None


def probe_native_multioutput(estimator_factory: Callable[[], object]) -> ProbeResult:
    if not callable(estimator_factory):
        raise TypeError("estimator_factory must be callable")

    try:
        factory_estimator = estimator_factory()
    except Exception as exc:
        return ProbeResult(False, f"factory failed: {type(exc).__name__}: {exc}")

    try:
        estimator = clone(factory_estimator)
    except Exception:
        try:
            estimator = copy.deepcopy(factory_estimator)
        except Exception as exc:
            return ProbeResult(
                False,
                f"estimator copy failed: {type(exc).__name__}: {exc}",
            )

    if not callable(getattr(estimator, "fit", None)):
        return ProbeResult(False, "factory result has no callable fit method")
    if not callable(getattr(estimator, "predict", None)):
        return ProbeResult(False, "factory result has no callable predict method")

    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=float,
    )
    Y = np.column_stack((X[:, 0] + X[:, 1], X[:, 0] - X[:, 1]))

    try:
        estimator.fit(X, Y)
    except Exception as exc:
        return ProbeResult(False, f"fit failed: {type(exc).__name__}: {exc}")

    try:
        prediction = np.asarray(estimator.predict(X))
    except Exception as exc:
        return ProbeResult(False, f"predict failed: {type(exc).__name__}: {exc}")

    expected_shape = (X.shape[0], Y.shape[1])
    if prediction.shape != expected_shape:
        return ProbeResult(
            False,
            f"prediction shape {prediction.shape} does not match {expected_shape}",
        )
    try:
        finite = bool(np.isfinite(prediction).all())
    except TypeError as exc:
        return ProbeResult(
            False,
            f"prediction finiteness check failed: {type(exc).__name__}: {exc}",
        )
    if not finite:
        return ProbeResult(False, "prediction contains non-finite values")
    return ProbeResult(True, None)


class _ModelFactoryEstimator:
    """Adapt the project ModelFactory wrappers to the canonical ndarray contract."""

    def __init__(
        self,
        model_type: str,
        params: Mapping[str, object],
        feature_names: Sequence[str] | None,
    ) -> None:
        model_factory_type = importlib.import_module(
            "models.ModelFactory"
        ).ModelFactory
        self.model_type = _normalize_model_type(model_type)
        self.feature_names = tuple(feature_names or ())
        self.model = model_factory_type(log_prefix="CanonicalModelFactory").create_model(
            self.model_type,
            dict(params),
            log_params=False,
        )

    def _input(self, X: object) -> np.ndarray | pd.DataFrame:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError("canonical estimator input must be two-dimensional")
        columns = self.feature_names or tuple(
            f"x{index}" for index in range(values.shape[1])
        )
        if len(columns) != values.shape[1]:
            raise ValueError("feature_names width does not match estimator input")
        if self.model_type in {
            "seasonaltemplate",
            "st",
            "lightgbm",
            "lgb",
        }:
            frame = pd.DataFrame(values)
            frame.columns = list(columns)
            return frame
        return values

    def _frame(self, X: object) -> pd.DataFrame:
        values = self._input(X)
        if isinstance(values, pd.DataFrame):
            return values
        columns = self.feature_names or tuple(
            f"x{index}" for index in range(values.shape[1])
        )
        return pd.DataFrame(values, columns=columns)

    def fit(self, X: object, y: object, sample_weight=None):
        targets = np.asarray(y)
        if self.model_type == "ridge" and targets.ndim != 1:
            raise ValueError("ridge scalar adapter requires one-dimensional targets")
        self.model.fit(self._input(X), targets, sample_weight=sample_weight)
        return self

    def predict(self, X: object) -> np.ndarray:
        return np.asarray(self.model.predict(self._input(X)), dtype=float)


    def fit_multi_output(self, X: object, Y: object, sample_weight=None):
        if self.model_type != "ridge":
            raise ValueError("multi-output fast path is only available for ridge")
        design = self._input(X)
        targets = np.asarray(Y, dtype=float)
        if targets.ndim != 2:
            raise ValueError("ridge multi-output targets must be two-dimensional")
        self.model.model.fit(design, targets, sample_weight=sample_weight)
        self.model.is_fitted = True
        return self


class _MultiOutputSliceEstimator:
    """Scalar prediction view over one shared multi-output estimator."""

    def __init__(self, shared: _ModelFactoryEstimator, column_index: int) -> None:
        self._shared = shared
        self._column_index = column_index

    def predict(self, X: object) -> np.ndarray:
        values = np.asarray(self._shared.predict(X), dtype=float)
        if values.ndim != 2 or self._column_index >= values.shape[1]:
            raise ValueError("shared multi-output prediction has an invalid shape")
        return values[:, self._column_index]


def _fit_ridge_independent_outputs(
    params: Mapping[str, object],
    feature_names: Sequence[str] | None,
    X: object,
    Y: object,
    sample_weight=None,
) -> tuple[_MultiOutputSliceEstimator, ...]:
    shared = _ModelFactoryEstimator("ridge", params, feature_names)
    shared.fit_multi_output(X, Y, sample_weight=sample_weight)
    width = np.asarray(Y).shape[1]
    return tuple(
        _MultiOutputSliceEstimator(shared, column_index)
        for column_index in range(width)
    )


_SCALAR_QUANTILE_MODEL_TYPES = frozenset(
    {
        "lightgbm",
        "lgb",
        "xgboost",
        "xgb",
        "catboost",
        "cat",
        "histgb",
        "histgradientboosting",
        "quantileregressor",
        "qr",
    }
)
_CATEGORICAL_MODEL_TYPES = frozenset(
    {"lightgbm", "lgb", "catboost", "cat", "histgb", "histgradientboosting"}
)
_NAN_MODEL_TYPES = frozenset(
    {
        "lightgbm",
        "lgb",
        "xgboost",
        "xgb",
        "catboost",
        "cat",
        "histgb",
        "histgradientboosting",
    }
)
_NO_SAMPLE_WEIGHT_MODEL_TYPES = frozenset({"seasonaltemplate", "st"})
_MODEL_FACTORY_TYPES = (
    "lightgbm",
    "lgb",
    "xgboost",
    "xgb",
    "catboost",
    "cat",
    "randomforest",
    "rf",
    "histgb",
    "histgradientboosting",
    "ridge",
    "elasticnet",
    "enet",
    "lasso",
    "quantileregressor",
    "qr",
    "seasonaltemplate",
    "st",
)


MODEL_FACTORY_CAPABILITY_REGISTRY = CapabilityRegistry(
    {
        model_type: EstimatorCapabilities(
            scalar_target=True,
            scalar_quantile=model_type in _SCALAR_QUANTILE_MODEL_TYPES,
            native_multi_target_point=False,
            native_multi_target_quantile=False,
            sample_weight=model_type not in _NO_SAMPLE_WEIGHT_MODEL_TYPES,
            categorical=model_type in _CATEGORICAL_MODEL_TYPES,
            nan_support=model_type in _NAN_MODEL_TYPES,
        )
        for model_type in _MODEL_FACTORY_TYPES
    }
)


def _quantile_params(
    model_type: str,
    params: Mapping[str, object],
    quantile: float,
) -> dict[str, object]:
    resolved = dict(params)
    if model_type in {"lightgbm", "lgb"}:
        resolved.update(objective="quantile", alpha=quantile)
    elif model_type in {"xgboost", "xgb"}:
        resolved.update(objective="reg:quantileerror", quantile_alpha=quantile)
    elif model_type in {"catboost", "cat"}:
        resolved["loss_function"] = f"Quantile:alpha={quantile}"
    elif model_type in {"histgb", "histgradientboosting"}:
        resolved.update(loss="quantile", quantile=quantile)
    elif model_type in {"quantileregressor", "qr"}:
        resolved["quantile"] = quantile
    else:
        raise ValueError(
            f"model_type {model_type!r} does not declare scalar quantile support"
        )
    return resolved


def make_model_factory(
    model_type: str,
    params: Mapping[str, object] | None = None,
    *,
    feature_names: Sequence[str] | None = None,
    quantile: float | None = None,
) -> Callable[[], object]:
    normalized = _normalize_model_type(model_type)
    capabilities = MODEL_FACTORY_CAPABILITY_REGISTRY.lookup(normalized)
    resolved_params = dict(params or {})
    if quantile is not None:
        if not capabilities.scalar_quantile:
            raise ValueError(
                f"model_type {normalized!r} does not support scalar quantiles"
            )
        resolved_params = _quantile_params(normalized, resolved_params, float(quantile))
    factory = partial(
        _ModelFactoryEstimator,
        normalized,
        resolved_params,
        feature_names,
    )
    if normalized == "ridge" and quantile is None:
        setattr(
            factory,
            "fit_independent_outputs",
            partial(
                _fit_ridge_independent_outputs,
                resolved_params,
                feature_names,
            ),
        )
    return factory


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


_NATIVE_MULTI_QUANTILE_MODEL_TYPES = frozenset({"xgboost", "xgb"})


def supports_native_multi_quantile(model_type: str) -> bool:
    """该模型类型是否支持单次训练输出整个 quantile grid（原生多分位）。

    目前仅 xgboost>=2.0（``quantile_alpha`` 接受列表，单 booster 每叶
    输出全部 level）。pyproject 钉 ``xgboost>=3.2.0``，运行时仍做版本
    防御性探测，不支持时回落 False（调用方走逐 level 独立训练）。
    """
    normalized = _normalize_model_type(model_type)
    if normalized not in _NATIVE_MULTI_QUANTILE_MODEL_TYPES:
        return False
    try:
        xgboost = importlib.import_module("xgboost")
        major = int(str(xgboost.__version__).split(".")[0])
    except Exception:
        return False
    return major >= 2


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
        self._fitted: dict[int, _ModelFactoryEstimator] = {}
        self._lock = threading.Lock()

    def __getstate__(self) -> dict:
        # 线程锁不可序列化；bundle 部署期只读 pool（不再 fit），重建即可
        state = dict(self.__dict__)
        state["_lock"] = None
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
            estimator = _ModelFactoryEstimator(
                self.model_type,
                self.params,
                self.feature_names,
            )
            estimator.fit(X, y, sample_weight=sample_weight)
            values = np.asarray(estimator.predict(X), dtype=float)
            if values.ndim != 2 or values.shape[1] != len(self.levels):
                raise ValueError(
                    "native multi-quantile predict must return "
                    f"(n_samples, {len(self.levels)}); got {values.shape}"
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
