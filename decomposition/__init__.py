# -*- coding: utf-8 -*-
"""时间序列分解模块（重构后主入口）。

构造：Configuration / Presets → Registry / Component Factory。
运行：Pipeline 编排 Extraction、Extrapolation、Composition。
基础契约：Contracts；诊断旁路：Diagnostics。
配置契约见 decomposition/configuration/spec.py；设计文档 docs/redesign/decomposition.md。
"""
from decomposition.orchestration.pipeline import DecompositionPipeline
from decomposition.construction.registry import build_pipeline, build_pipeline_from_args
from decomposition.configuration.spec import (
    RESERVED_METHODS,
    TREND_FORECAST_MODES,
    ComponentSpec,
    DecompositionSpec,
    PresetParams,
    resolve_decomposition_spec,
)
from decomposition.contracts.types import ComponentForecast, ComponentFrame

__all__ = [
    "RESERVED_METHODS",
    "TREND_FORECAST_MODES",
    "ComponentSpec",
    "ComponentForecast",
    "ComponentFrame",
    "DecompositionPipeline",
    "DecompositionSpec",
    "PresetParams",
    "build_pipeline",
    "build_pipeline_from_args",
    "resolve_decomposition_spec",
]
