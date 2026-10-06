# -*- coding: utf-8 -*-
"""训练期监督特征选择（SelectKBest），canonical 接线版。

定位决策（2026-08-30 专项）：特征选择是**有监督**步骤（需要 Y），因此挂在
训练 fit 边界而非编译边界——每个回测窗口与最终训练各自重拟合选择器，
只消费当前训练窗的 (X, Y)，天然满足 as-of/无泄漏契约；选中的特征集写入
strategy artifact 的 feature_schema，预测端按同名子集对齐，bundles 自足。

legacy `features/FeatureSelection.py::FeatureSelector` 的 canonical 复活：
ndarray 化、严格配置校验、默认关闭（未配置 = 行为零变化、不进 fingerprint）。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.feature_selection import f_regression, mutual_info_regression, r_regression

_SELECTION_FIELDS = frozenset(
    {"enabled", "method", "max_features", "min_features", "force_keep"}
)
_METHODS = {
    "f_regression": f_regression,
    "mutual_info": mutual_info_regression,
}


class FeatureSelectionSpec:
    """features.selection 配置（严格解析，未知字段 RAISE）。"""

    __slots__ = ("enabled", "method", "max_features", "min_features", "force_keep")

    def __init__(
        self,
        *,
        enabled: bool = False,
        method: str = "f_regression",
        max_features: int = 80,
        min_features: int = 10,
        force_keep: tuple[str, ...] = (),
    ) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("features.selection.enabled must be a bool")
        if method not in _METHODS:
            raise ValueError(
                f"features.selection.method must be one of {sorted(_METHODS)}"
            )
        for name, value in (("max_features", max_features), ("min_features", min_features)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"features.selection.{name} must be a positive integer")
        if min_features > max_features:
            raise ValueError("features.selection.min_features must be <= max_features")
        if not isinstance(force_keep, tuple) or not all(
            isinstance(name, str) for name in force_keep
        ):
            raise TypeError("features.selection.force_keep must be a list of strings")
        self.enabled = enabled
        self.method = method
        self.max_features = max_features
        self.min_features = min_features
        self.force_keep = force_keep

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": self.method,
            "max_features": self.max_features,
            "min_features": self.min_features,
            "force_keep": list(self.force_keep),
        }


def normalize_feature_selection(value: Any) -> FeatureSelectionSpec | None:
    """解析 features.selection；缺省/None = 不启用（不进 fingerprint）。"""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("features.selection must be a mapping")
    unknown = set(value) - _SELECTION_FIELDS
    if unknown:
        raise ValueError(f"features.selection unknown fields: {sorted(unknown)}")
    force_keep = value.get("force_keep", ())
    if isinstance(force_keep, (str, bytes)) or not isinstance(force_keep, (list, tuple)):
        raise TypeError("features.selection.force_keep must be a sequence of strings")
    return FeatureSelectionSpec(
        enabled=value.get("enabled", False),
        method=value.get("method", "f_regression"),
        max_features=value.get("max_features", 80),
        min_features=value.get("min_features", 10),
        force_keep=tuple(value.get("force_keep", ())),
    )


class CanonicalFeatureSelector:
    """fit on 训练窗 / transform on 推理的列子集选择器（ndarray 契约）。

    所有 design 共享同一固定 feature schema（canonical 合同），因此列子集
    对每个 design 一致应用。
    """

    def __init__(
        self,
        spec: FeatureSelectionSpec,
        feature_schema: tuple[str, ...],
    ) -> None:
        if len(set(feature_schema)) != len(feature_schema):
            raise ValueError("feature schema must be unique for selection")
        self._spec = spec
        self._schema = tuple(feature_schema)
        self.selected_names_: tuple[str, ...] | None = None

    def fit(self, X: np.ndarray, y_signal: np.ndarray) -> "CanonicalFeatureSelector":
        return self.fit_calls((X,), np.asarray(y_signal, dtype=float).reshape(-1, 1, 1))

    def fit_calls(self, calls, targets: np.ndarray, *, call_horizons=None) -> "CanonicalFeatureSelector":
        """各 call 与各目标通道独立评分，单位最大值归一后等权聚合。

        不平均物理目标值，避免反向目标相消及不同量纲主导选择。
        保留一个统一列子集，所有训练/预测 call 使用相同 schema。
        """
        arrays = tuple(np.asarray(call, dtype=float) for call in calls)
        y = np.asarray(targets, dtype=float)
        if not arrays or y.ndim != 3 or not y.shape[0] or not np.isfinite(y).all():
            raise ValueError("selection requires nonempty calls and finite (samples,H,K) targets")
        for array in arrays:
            if array.ndim != 2 or array.shape != (len(y), len(self._schema)) or not np.isfinite(array).all():
                raise ValueError("selection X must have finite rows matching target samples and feature_schema width")
        for name in self._spec.force_keep:
            if name not in self._schema:
                raise ValueError(f"force_keep feature {name!r} not in feature schema")
        spec = self._spec
        if not spec.enabled or len(self._schema) <= spec.min_features:
            self.selected_names_ = self._schema
            return self
        scores = np.zeros(len(self._schema), dtype=float)
        if call_horizons is None:
            call_horizons = tuple((index,) for index in range(len(arrays))) if len(arrays) == y.shape[1] else (tuple(range(y.shape[1])),) * len(arrays)
        if len(call_horizons) != len(arrays) or any(not steps or any(type(step) is not int or not 0 <= step < y.shape[1] for step in steps) for steps in call_horizons):
            raise ValueError("selection call_horizons must match calls and target horizon")
        for array, steps in zip(arrays, call_horizons):
            for signal in y[:, steps, :].reshape(len(y), -1).T:
                # 常数目标不提供选列信息；不产生随机互信息噪声。
                if np.ptp(signal) == 0:
                    continue
                if spec.method == "f_regression":
                    # F 分数由有界相关系数推导；sklearn 在 r 略大于1时
                    # 可产生负的巨大 F 值。仅数值上截断理论有界的 r。
                    correlation = np.clip(r_regression(array, signal, force_finite=True), -1., 1.)
                    squared = correlation ** 2
                    current = squared / np.maximum(1. - squared, np.finfo(float).eps)
                else:
                    current = mutual_info_regression(array, signal, random_state=0)
                maximum = float(np.max(current))
                if maximum > 0:
                    scores += current / maximum
        k = min(spec.max_features, len(self._schema))
        selected = {self._schema[index] for index in np.argsort(-scores, kind="stable")[:k]}
        selected.update(spec.force_keep)
        self.selected_names_ = tuple(name for name in self._schema if name in selected)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_names_ is None:
            raise RuntimeError("selector must be fitted before transform")
        indices = self.indices(self._schema)
        return np.asarray(X, dtype=float)[:, indices]

    def indices(self, full_schema: tuple[str, ...]) -> tuple[int, ...]:
        if self.selected_names_ is None:
            raise RuntimeError("selector must be fitted before indices")
        return tuple(full_schema.index(name) for name in self.selected_names_)


def selected_indices_for_artifact(
    full_schema: tuple[str, ...], artifact_schema: tuple[str, ...]
) -> tuple[int, ...] | None:
    """预测端列子集推导：artifact 记录的选中 schema 名 → 全 schema 列索引。

    两者一致（未启用选择）时返回 None，调用端零开销直通。
    """
    if tuple(artifact_schema) == tuple(full_schema):
        return None
    missing = [name for name in artifact_schema if name not in full_schema]
    if missing:
        raise ValueError(
            f"artifact feature schema is not a subset of the compiled schema: {missing}"
        )
    return tuple(full_schema.index(name) for name in artifact_schema)


__all__ = [
    "CanonicalFeatureSelector",
    "FeatureSelectionSpec",
    "normalize_feature_selection",
    "selected_indices_for_artifact",
]
