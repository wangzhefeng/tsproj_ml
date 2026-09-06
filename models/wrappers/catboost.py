"""catboost: estimator wrappers extracted from the model factory."""

import copy
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import catboost as cab
from utils.log_util import logger
from models.wrappers.base import BaseModel, DEFAULT_EARLY_STOPPING_ROUNDS, _filter_valid_params, _filter_fit_params


class CatBoostModel(BaseModel):
    """
    CatBoost 模型封装

    特点:
    - 自动处理类别特征
    - 对默认参数不敏感
    - 过拟合风险低
    - 性能优秀
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
        supplied = _filter_valid_params(copy.deepcopy(params or {}), cab.CatBoostRegressor)
        defaults = copy.deepcopy(self.DEFAULT_PARAMS)
        # 原生同义参数规则先作用于用户输入，避免默认值压过显式别名。
        cab.core._process_synonyms(supplied)
        cab.core._process_synonyms(defaults)
        if "logging_level" in supplied:
            defaults.pop("verbose", None)
        merged_params = {**defaults, **supplied}
        self.params = _filter_valid_params(merged_params, cab.CatBoostRegressor)
        if self.log_params:
            logger.info(f"{log_prefix} model parameters: \n{self.params}")
        # 模型构建
        self.model = cab.CatBoostRegressor(**self.params)

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
        fit_input = native_train_data if native_train_data is not None else X
        fit_target = None if native_train_data is not None else y
        self.model.fit(fit_input, fit_target, **fit_params)
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测
        """
        if not self.is_fitted:
            raise ValueError(f"{self.log_prefix} 模型尚未训练(Model not fitted yet).")

        return self.model.predict(X)
