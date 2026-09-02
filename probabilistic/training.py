# -*- coding: utf-8 -*-
"""按 quantile grid 编排训练并构造强类型模型 bundle。"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Sequence

import numpy as np

from model_training.estimators import EstimatorCapabilities
from model_training.trainer import CanonicalTrainer
from forecasting_core.specs import ForecastConfigSpec
from forecasting_core.probabilistic_spec import validate_quantile_grid


@dataclass(frozen=True, slots=True)
class CanonicalMarginalQuantileArtifact:
    levels: tuple[float, ...]
    point_level: float
    artifacts_by_level: MappingProxyType
    dependence_model: None = None
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("canonical quantile artifact schema_version must be 2")
        if self.dependence_model is not None:
            raise ValueError("canonical marginal artifact dependence_model must be None")
        if tuple(self.artifacts_by_level) != self.levels:
            raise ValueError("artifacts_by_level keys must exactly match levels")

    def __reduce__(self):
        return (
            type(self),
            (
                self.levels,
                self.point_level,
                dict(self.artifacts_by_level),
                self.dependence_model,
                self.schema_version,
            ),
        )


class CanonicalMarginalQuantileTrainer:
    """Fit one canonical strategy artifact per marginal quantile level."""

    def __init__(
        self,
        config: ForecastConfigSpec,
        *,
        estimator_factory_for_level: Callable[[float], Callable[[], object]],
        capabilities: EstimatorCapabilities,
        feature_schema: Sequence[str],
    ) -> None:
        if not isinstance(config, ForecastConfigSpec):
            raise TypeError("config must be a ForecastConfigSpec")
        if str(config.probabilistic.get("mode", "point")) != "quantile":
            raise ValueError(
                "CanonicalMarginalQuantileTrainer requires probabilistic.mode=quantile"
            )
        if not callable(estimator_factory_for_level):
            raise TypeError("estimator_factory_for_level must be callable")
        raw_levels = config.probabilistic.get("quantiles")
        if not isinstance(raw_levels, (list, tuple)):
            raise TypeError("probabilistic.quantiles must be a sequence")
        # 网格校验统一走合同层唯一实现（2026-09-01 去重：不再本地重复校验）
        levels = validate_quantile_grid(raw_levels, point_quantile=float(
            config.probabilistic.get("point_quantile", 0.5)
        ))
        self.config = config
        self.estimator_factory_for_level = estimator_factory_for_level
        self.capabilities = capabilities
        self.feature_schema = tuple(feature_schema)
        self.levels = levels
        self.point_level = float(config.probabilistic.get("point_quantile", 0.5))

    def train(
        self,
        X_by_call: Sequence[np.ndarray],
        Y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        n_series: int = 1,
        max_workers: int | None = None,
        output_workers: int = 1,
    ) -> CanonicalMarginalQuantileArtifact:
        """逐 level 训练完整 canonical 策略 artifact。

        level 之间相互独立（同一 config、同一 (X, Y)，仅 objective 不同），
        默认以线程池并行（GBM 拟合释放 GIL）；数值与串行完全一致。
        共享 booster 路径（xgb 原生多分位）的调用方必须传
        ``max_workers=1``——位置对齐不变量只在串行下成立。
        """
        levels = self.levels
        workers = (
            min(len(levels), 4) if max_workers is None else int(max_workers)
        )
        if workers < 1:
            raise ValueError("max_workers must be >= 1")
        if (
            isinstance(output_workers, bool)
            or not isinstance(output_workers, int)
            or output_workers < 1
        ):
            raise ValueError("output_workers must be a positive integer")
        effective_output_workers = 1 if workers > 1 else output_workers
        if workers == 1 or len(levels) <= 1:
            artifacts = {
                level: self._train_level(
                    level,
                    X_by_call,
                    Y,
                    sample_weight,
                    n_series,
                    effective_output_workers,
                )
                for level in levels
            }
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        self._train_level,
                        level,
                        X_by_call,
                        Y,
                        sample_weight,
                        n_series,
                        effective_output_workers,
                    ): level
                    for level in levels
                }
                results = {}
                for future in futures:
                    level = futures[future]
                    results[level] = future.result()
                # 按 levels 顺序重建，结果 dict 与串行路径完全同构
                artifacts = {level: results[level] for level in levels}
        return CanonicalMarginalQuantileArtifact(
            levels=self.levels,
            point_level=self.point_level,
            artifacts_by_level=MappingProxyType(artifacts),
        )

    def _train_level(
        self,
        level: float,
        X_by_call: Sequence[np.ndarray],
        Y: np.ndarray,
        sample_weight: np.ndarray | None,
        n_series: int,
        output_workers: int,
    ):
        estimator_factory = self.estimator_factory_for_level(level)
        if not callable(estimator_factory):
            raise TypeError(
                "estimator_factory_for_level must return an estimator factory"
            )
        return CanonicalTrainer(
            self.config,
            estimator_factory=estimator_factory,
            capabilities=self.capabilities,
            feature_schema=self.feature_schema,
        ).train(
            X_by_call,
            Y,
            sample_weight=sample_weight,
            n_series=n_series,
            max_workers=output_workers,
        )
