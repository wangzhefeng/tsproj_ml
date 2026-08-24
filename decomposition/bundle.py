# -*- coding: utf-8 -*-
"""分解模块的持久化 bundle。

统一封装 model + scaler + decomposition pipeline 的保存与加载，
替代旧的散装 pkl（model.pkl / target_scaler.pkl / target_decomposer.pkl
各自独立保存、无 schema 关联）。
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class DecompositionBundle:
    """一次分解预测所需的全部可恢复状态（内嵌式单文件部署产物）。

    schema_version=2：内嵌 model / target_scaler / pipeline；
    schema_version=1（历史）：仅 pipeline + spec 摘要。
    """

    pipeline: Any  # DecompositionPipeline（已 fit）
    spec_summary: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 2
    model: Any = None
    target_scaler: Any = None

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path: Path) -> "DecompositionBundle":
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        if not isinstance(bundle, DecompositionBundle):
            raise TypeError(
                f"Expected DecompositionBundle, got {type(bundle).__name__} from {path}"
            )
        if bundle.schema_version not in (1, 2):
            raise ValueError(
                f"Unsupported bundle schema_version={bundle.schema_version}: {path}"
            )
        return bundle

    @staticmethod
    def from_components(
        pipeline: Any,
        model: Any = None,
        target_scaler: Any = None,
    ) -> "DecompositionBundle":
        """从已 fit 的组件构造 v2 bundle（model/scaler 内嵌，可缺省）。"""
        bundle = DecompositionBundle.from_pipeline(pipeline)
        bundle.model = model
        bundle.target_scaler = target_scaler
        bundle.schema_version = 2
        return bundle

    @staticmethod
    def from_pipeline(pipeline: Any) -> "DecompositionBundle":
        """从已 fit 的 pipeline 构造 bundle（spec 摘要自动提取，保留 v1 兼容入口）。"""
        spec = getattr(pipeline, "spec", None)
        summary: dict[str, Any] = {"method": getattr(pipeline, "method", "none")}
        if spec is not None and getattr(spec, "preset", None) is not None:
            p = spec.preset
            summary.update(
                {
                    "composition": p.composition,
                    "periods": list(p.periods),
                    "robust": p.robust,
                    "trend_degree": p.trend_degree,
                    "trend_forecast": p.trend_forecast,
                    "damping": p.damping,
                    "trend_lookback": p.trend_lookback,
                    "seasonal_cycles": p.seasonal_cycles,
                }
            )
        return DecompositionBundle(pipeline=pipeline, spec_summary=summary)
