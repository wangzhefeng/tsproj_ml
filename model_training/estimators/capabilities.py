"""Estimator capability declarations and behavioral probes."""

import copy
import importlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from functools import partial
from types import MappingProxyType
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone

from model_forecasting.specs.estimator import EstimatorCapabilities

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
        self.feature_names = tuple(feature_names or ())
        self.model = model_factory_type(log_prefix="CanonicalModelFactory").create_model(
            model_type,
            dict(params),
            log_params=False,
        )

    def _frame(self, X: object) -> pd.DataFrame:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError("canonical estimator input must be two-dimensional")
        columns = self.feature_names or tuple(
            f"x{index}" for index in range(values.shape[1])
        )
        if len(columns) != values.shape[1]:
            raise ValueError("feature_names width does not match estimator input")
        return pd.DataFrame(values, columns=columns)

    def fit(self, X: object, y: object, sample_weight=None):
        self.model.fit(self._frame(X), np.asarray(y), sample_weight=sample_weight)
        return self

    def predict(self, X: object) -> np.ndarray:
        return np.asarray(self.model.predict(self._frame(X)), dtype=float)


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
    return partial(
        _ModelFactoryEstimator,
        normalized,
        resolved_params,
        feature_names,
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
