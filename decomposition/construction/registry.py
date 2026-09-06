# -*- coding: utf-8 -*-
"""Registry：从 DecompositionSpec 构造 DecompositionPipeline。

只做构造，不做预测逻辑。未实现方法（custom + RESERVED_METHODS）在此层 fail-fast。
"""
from __future__ import annotations

from typing import Any

from decomposition.orchestration.pipeline import DecompositionPipeline
from decomposition.configuration.spec import DecompositionSpec, resolve_decomposition_spec


def build_pipeline(spec: DecompositionSpec) -> DecompositionPipeline:
    """由 spec 构造 pipeline 实例。custom/预留方法在 spec.resolve 已拦截，
    此处兜底防御。"""
    if spec.method == "custom":
        raise ValueError(
            "decomposition method 'custom' is not implemented; "
            "use presets none/linear/stl/mstl."
        )
    return DecompositionPipeline(spec)


def build_pipeline_from_args(args: Any) -> DecompositionPipeline:
    """主链入口：cfg → resolve → build pipeline。运行时唯一配置构造入口。"""
    spec = resolve_decomposition_spec(args)
    return build_pipeline(spec)
