# -*- coding: utf-8 -*-
"""按 quantile grid 编排训练并构造强类型模型 bundle。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional, Sequence

import pandas as pd

from models.multistep.spec import RolloutFamily, get_strategy_spec
from probabilistic.spec import ProbabilisticSpec
from probabilistic.types import BlendQuantileModel, ProbabilisticModelBundle


SingleTrainer = Callable[[float, pd.DataFrame, pd.DataFrame, Sequence[str]], Any]
BlendTrainer = Callable[
    [float, pd.DataFrame, pd.DataFrame, pd.DataFrame, Sequence[str]],
    BlendQuantileModel,
]


class QuantileTrainer:
    """概率训练编排器；具体 estimator fit 由现有 Trainer backend 提供。"""

    def __init__(
        self,
        spec: ProbabilisticSpec,
        model_type: str,
        pred_method: str,
        train_single: SingleTrainer,
        train_blend: Optional[BlendTrainer] = None,
        max_workers: int = 1,
    ) -> None:
        if spec.mode != "quantile":
            raise ValueError("QuantileTrainer requires spec.mode=quantile")
        self.spec = spec
        self.model_type = str(model_type)
        self.strategy_spec = get_strategy_spec(pred_method)
        self.pred_method = self.strategy_spec.method
        self.train_single = train_single
        self.train_blend = train_blend
        self.max_workers = max(1, int(max_workers or 1))

    @property
    def is_blend(self) -> bool:
        return self.strategy_spec.rollout == RolloutFamily.BLEND

    @staticmethod
    def _split_blend_targets(Y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        recursive_columns = [
            column for column in Y.columns if str(column).endswith("_shift_0")
        ]
        direct_columns = [column for column in Y.columns if column not in recursive_columns]
        if not direct_columns or len(recursive_columns) != 1:
            raise ValueError(
                "Blend quantile requires at least one direct target and exactly one "
                f"recursive shift_0 target; got direct={len(direct_columns)}, "
                f"recursive={len(recursive_columns)}"
            )
        return Y[direct_columns], Y[recursive_columns]

    def _train_one(
        self,
        quantile: float,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        categorical_features: Sequence[str],
        Y_direct: Optional[pd.DataFrame],
        Y_recursive: Optional[pd.DataFrame],
    ) -> tuple[float, Any]:
        if self.is_blend:
            if self.train_blend is None or Y_direct is None or Y_recursive is None:
                raise RuntimeError("blend quantile training backend is not configured")
            model = self.train_blend(
                quantile,
                X,
                Y_direct,
                Y_recursive,
                categorical_features,
            )
            if not isinstance(model, BlendQuantileModel):
                raise TypeError("train_blend must return BlendQuantileModel")
        else:
            model = self.train_single(quantile, X, Y, categorical_features)
        return quantile, model

    def fit(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        categorical_features: Sequence[str],
    ) -> ProbabilisticModelBundle:
        if not isinstance(X, pd.DataFrame) or not isinstance(Y, pd.DataFrame):
            raise TypeError("QuantileTrainer.fit requires pandas DataFrame X and Y")
        if len(X) != len(Y):
            raise ValueError(f"QuantileTrainer X/Y length mismatch: X={len(X)}, Y={len(Y)}")

        Y_direct = None
        Y_recursive = None
        if self.is_blend:
            Y_direct, Y_recursive = self._split_blend_targets(Y)

        models: dict[float, Any] = {}
        if self.max_workers > 1 and len(self.spec.quantiles) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(
                        self._train_one,
                        quantile,
                        X,
                        Y,
                        categorical_features,
                        Y_direct,
                        Y_recursive,
                    )
                    for quantile in self.spec.quantiles
                ]
                for future in as_completed(futures):
                    quantile, model = future.result()
                    models[quantile] = model
        else:
            for quantile in self.spec.quantiles:
                level, model = self._train_one(
                    quantile,
                    X,
                    Y,
                    categorical_features,
                    Y_direct,
                    Y_recursive,
                )
                models[level] = model

        return ProbabilisticModelBundle(
            schema_version=1,
            spec=self.spec,
            model_type=self.model_type,
            pred_method=self.pred_method,
            models_by_quantile=models,
            recursive_propagation=self.spec.recursive_propagation,
        )
