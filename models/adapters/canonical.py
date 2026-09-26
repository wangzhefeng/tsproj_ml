"""canonical ndarray 合同下的 estimator 适配层。

把项目 ModelFactory 的 DataFrame 接口适配为 canonical ``(N, K)`` ndarray
fit/predict 合同，并提供 quantile 目标注入与能力行为探测。

分层边界（tests/test_package_layering.py：models 只允许依赖 utils）：
- 本模块零 forecasting_core 依赖；``quantile`` 支持判断直接读 catalog
  descriptor（``quantile_style is not None``），不经合同层类型。
- ``CapabilityRegistry`` / ``MODEL_FACTORY_CAPABILITY_REGISTRY`` /
  ``resolve_model_capabilities`` 构造并返回合同层
  ``EstimatorCapabilities`` 实例，留在 ``model_training/estimators``。
- ``SharedMultiQuantilePool`` 依赖 checkpoint，同留训练层。
"""

import copy
import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone

from models.catalog import MODEL_CATALOG, quantile_parameters
from models.factory import ModelFactory


def _normalize_model_type(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("model_type must be a string")
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError("model_type must not be blank")
    return normalized


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


class ModelFactoryEstimator:
    """Adapt the project ModelFactory wrappers to the canonical ndarray contract.

    原 ``model_training/estimators/capabilities.py::_ModelFactoryEstimator``
    （2026-09-26 适配层下沉 models/adapters，私有名同步公开化；
    引用旧私有类路径的存量 pickle 按仓库惯例作废重训）。
    """

    def __init__(
        self,
        model_type: str,
        params: Mapping[str, object],
        feature_names: Sequence[str] | None,
    ) -> None:
        model_factory_type = ModelFactory
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
        if MODEL_CATALOG[self.model_type].dataframe_input:
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
        assert self.model is not None
        self.model.model.fit(design, targets, sample_weight=sample_weight)
        self.model.is_fitted = True
        return self


class _MultiOutputSliceEstimator:
    """Scalar prediction view over one shared multi-output estimator."""

    def __init__(self, shared: ModelFactoryEstimator, column_index: int) -> None:
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
    shared = ModelFactoryEstimator("ridge", params, feature_names)
    shared.fit_multi_output(X, Y, sample_weight=sample_weight)
    width = np.asarray(Y).shape[1]
    return tuple(
        _MultiOutputSliceEstimator(shared, column_index)
        for column_index in range(width)
    )


def _quantile_params(
    model_type: str,
    params: Mapping[str, object],
    quantile: float,
) -> dict[str, object]:
    return quantile_parameters(model_type, dict(params), quantile)


def make_model_factory(
    model_type: str,
    params: Mapping[str, object] | None = None,
    *,
    feature_names: Sequence[str] | None = None,
    quantile: float | None = None,
) -> Callable[[], object]:
    """构造 canonical ndarray 合同的 estimator 工厂。

    quantile 支持判断直接读 catalog descriptor（``quantile_style``），
    不经合同层能力注册表——注册表构造合同类型实例，属训练层。
    """
    normalized = _normalize_model_type(model_type)
    resolved_params = dict(params or {})
    if quantile is not None:
        if MODEL_CATALOG[normalized].quantile_style is None:
            raise ValueError(
                f"model_type {normalized!r} does not support scalar quantiles"
            )
        resolved_params = _quantile_params(normalized, resolved_params, float(quantile))
    factory = partial(
        ModelFactoryEstimator,
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


_NATIVE_MULTI_QUANTILE_MODEL_TYPES = frozenset(
    name for name, descriptor in MODEL_CATALOG.items() if descriptor.native_multi_quantile
)


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
