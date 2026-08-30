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
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

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
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self._schema):
            raise ValueError("X must be two-dimensional with feature_schema width")
        y_signal = np.asarray(y_signal, dtype=float).reshape(-1)
        if len(y_signal) != X.shape[0]:
            raise ValueError("y_signal must match X row count")

        n_features = X.shape[1]
        spec = self._spec
        if not spec.enabled or n_features <= spec.min_features:
            self.selected_names_ = self._schema
            return self

        k = min(max(spec.max_features, spec.min_features), n_features)
        selector = SelectKBest(score_func=_METHODS[spec.method], k=k)
        selector.fit(X, y_signal)
        support = np.asarray(selector.get_support(), dtype=bool)
        selected = [
            name for name, keep in zip(self._schema, support) if keep
        ]
        for name in spec.force_keep:
            if name not in self._schema:
                raise ValueError(f"force_keep feature {name!r} not in feature schema")
            if name not in selected:
                selected.append(name)
        # 保持 schema 顺序，保证列索引推导确定
        selected_set = set(selected)
        self.selected_names_ = tuple(
            name for name in self._schema if name in selected_set
        )
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
