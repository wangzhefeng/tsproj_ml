"""CatBoost 封装：对称树梯度提升（有序提升 + 目标统计类别编码）。"""

import copy
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import catboost as cab
from utils.log_util import logger
from model_building.preflight import filter_fit_params as _filter_fit_params, filter_valid_params as _filter_valid_params
from model_building.preflight.catboost import process_synonym_params
from model_building.wrappers.base import BaseModel, DEFAULT_EARLY_STOPPING_ROUNDS


class CatBoostModel(BaseModel):
    """
    CatBoost 对称树梯度提升封装

    特点:
    - 有序提升（ordered Boosting）缓解梯度偏差引起的预测偏移
    - 目标统计类别编码内置（类别高基数下防目标泄漏）
    - 同层对称树（oblivious tree）推理快、正则化强
    - 同义参数（如 verbose/logging_level）在构造前归一化，避免
      默认值压过显式别名
    """


    DEFAULT_PARAMS = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 6,
        "verbose": False,
        "random_seed": 42,
        "thread_count": 1,
        "allow_writing_files": False,
    }

    def __init__(self, params: Dict[str, Any], log_prefix: str="CatBoostModel", log_params: bool = True):
        super().__init__(params, log_prefix=log_prefix, log_params=log_params)

    def _resolve_params(self, supplied: Dict[str, Any]) -> Dict[str, Any]:
        # 原生同义参数规则先作用于用户输入与默认值，避免默认值压过显式别名
        supplied = _filter_valid_params(copy.deepcopy(supplied), cab.CatBoostRegressor)
        defaults = copy.deepcopy(self.DEFAULT_PARAMS)
        process_synonym_params(supplied)
        process_synonym_params(defaults)
        if "logging_level" in supplied:
            defaults.pop("verbose", None)
        merged_params = {**defaults, **supplied}
        return _filter_valid_params(merged_params, cab.CatBoostRegressor)

    def _build_estimator(self):
        return cab.CatBoostRegressor(**self.params)

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            categorical_feature: Optional[list] = None,
            eval_set: Optional[tuple] = None,
            eval_metric: Optional[str] = None,
            early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
            native_train_data = None,
            native_eval_data = None,
            sample_weight: Optional[Any] = None,
            **kwargs):
        """
        训练 CatBoost 模型

        Args:
            X: 训练特征
            y: 训练目标
            categorical_feature: 类别特征列表
            eval_set: 验证集 (X_val, y_val)
            eval_metric: 仅为跨模型统一接口而保留；CatBoost 的评估指标由构造参数
                eval_metric 决定，fit 不使用此参数
            early_stopping_rounds: 早停轮数
            native_train_data: 原生训练容器(Pool);权重已内嵌其中,无需再传 sample_weight
            native_eval_data: 原生验证容器(Pool)
            sample_weight: 非 native 路径的样本权重(例如时间衰减权重);native 路径忽略此项
            **kwargs: 为跨模型统一接口而保留，静默忽略
        """
        # 设置训练参数
        fit_params = {}
        if eval_set is not None:
            fit_params["eval_set"] = native_eval_data if native_eval_data is not None else eval_set
            fit_params["early_stopping_rounds"] = early_stopping_rounds
        if categorical_feature is not None and native_train_data is None:
            fit_params["cat_features"] = categorical_feature
        # 非 native 路径下补充样本权重;native 路径权重已在 Pool 中
        if sample_weight is not None and native_train_data is None:
            fit_params["sample_weight"] = sample_weight
        fit_params = _filter_fit_params(self.model, fit_params)
        # 模型训练
        assert self.model is not None  # _build_estimator 已构造
        fit_input = native_train_data if native_train_data is not None else X
        fit_target = None if native_train_data is not None else y
        self.model.fit(fit_input, fit_target, **fit_params)
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测
        """
        self._require_fitted()
        assert self.model is not None

        return self.model.predict(X)
