# -*- coding: utf-8 -*-
"""时间序列分解模块（重构后主入口）。

架构：Extractor → Forecaster → Composer → Pipeline（编排）→ Registry（构造）。
配置契约见 decomposition/spec.py；设计文档 docs/time_series_decomposition_redesign.md。
"""
from decomposition.pipeline import DecompositionPipeline
from decomposition.registry import build_pipeline, build_pipeline_from_args
from decomposition.diagnostics import write_diagnostics_report
from decomposition.residual_diagnostics import (
    diagnose_window_residual,
    summarize_window_residuals,
    write_residual_diagnostics,
)
from decomposition.periods import acf_periods, detect_periodicity, fft_dominant_period
from decomposition.spec import (
    RESERVED_METHODS,
    ComponentSpec,
    DecompositionSpec,
    PresetParams,
    resolve_decomposition_spec,
)
from decomposition.types import ComponentForecast, ComponentFrame

__all__ = [
    "RESERVED_METHODS",
    "ComponentSpec",
    "ComponentForecast",
    "ComponentFrame",
    "DecompositionPipeline",
    "DecompositionSpec",
    "PresetParams",
    "build_pipeline",
    "build_pipeline_from_args",
    "resolve_decomposition_spec",
    "write_diagnostics_report",
    "acf_periods",
    "detect_periodicity",
    "fft_dominant_period",
    "diagnose_window_residual",
    "summarize_window_residuals",
    "write_residual_diagnostics",
]
