"""sklearn_tree: estimator wrappers extracted from the model factory."""

import copy
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from utils.log_util import logger
from models.wrappers.base import BaseModel, _filter_valid_params


class RandomForestModel(BaseModel):
    """
    Random Forest 模型封装

    特点:
    - 鲁棒性强
    - 不易过拟合
    - 可解释性好
    - 并行化训练
    """

    DEFAULT_PARAMS = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "n_jobs": -1,
        "random_state": 42,
    }

    def __init__(self, params: Dict[str, Any], log_prefix: str="RandomForestModel", log_params: bool = True):
        super().__init__(params, log_prefix=log_prefix, log_params=log_params)
        # 参数合并（用户参数优先，避免被默认值覆盖）
        merged_params = {**copy.deepcopy(self.DEFAULT_PARAMS), **(params or {})}
        # 模型参数
        self.params = _filter_valid_params(merged_params, RandomForestRegressor)
        if self.log_params:
            logger.info(f"{log_prefix} model parameters: \n{self.params}")
        # 模型构建
        self.model = RandomForestRegressor(**self.params)

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            sample_weight: Optional[Any] = None,
            **kwargs):
        """
        训练 Random Forest 模型

        Args:
            X: 训练特征
            y: 训练目标
            sample_weight: 训练样本权重(例如时间衰减权重)
            **kwargs: 为跨模型统一接口而保留（eval_set / early_stopping_rounds 等），
                Random Forest 不支持验证集与早停，静默忽略
        """
        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测
        """
        if not self.is_fitted:
            raise ValueError(f"{self.log_prefix} 模型尚未训练(Model not fitted yet).")

        return self.model.predict(X)

class HistGBModel(BaseModel):
    """
    HistGradientBoosting 模型封装（sklearn 原生直方图 GBDT）

    特点:
    - 零外部依赖（sklearn 内置）
    - 原生支持 MAE（absolute_error）与分位数（quantile）损失
    - 支持类别特征（构造参数 categorical_features，列索引）
    - 训练/预测速度快，适合作为 LightGBM 的无依赖对照组

    注意:
    - 无线程数构造参数（底层走 OpenMP，由 OMP_NUM_THREADS 环境变量控制）
    - 早停为训练集内部自动切分（构造参数 early_stopping / validation_fraction /
      n_iter_no_change），不支持外部 eval_set
    """

    DEFAULT_PARAMS = {
        "loss": "absolute_error",
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "max_depth": None,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "max_bins": 255,
        "early_stopping": False,
        "random_state": 42,
        "verbose": 0,
    }

    def __init__(self, params: Dict[str, Any], log_prefix: str="HistGBModel", log_params: bool = True):
        super().__init__(params, log_prefix=log_prefix, log_params=log_params)
        merged_params = {**copy.deepcopy(self.DEFAULT_PARAMS), **(params or {})}
        # 模型参数
        self.params = _filter_valid_params(merged_params, HistGradientBoostingRegressor)
        if self.log_params:
            logger.info(f"{log_prefix} model parameters: \n{self.params}")
        # 模型构建
        self.model = HistGradientBoostingRegressor(**self.params)

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            categorical_feature: Optional[list] = None,
            sample_weight: Optional[Any] = None,
            **kwargs):
        """
        训练 HistGradientBoosting 模型

        Args:
            X: 训练特征
            y: 训练目标
            categorical_feature: 类别特征列名列表；HistGB 在构造参数
                categorical_features 中以列索引声明，此处做 名称→索引 映射
            sample_weight: 训练样本权重(例如时间衰减权重)
            **kwargs: 为跨模型统一接口而保留（eval_set / eval_metric /
                early_stopping_rounds），HistGB 早停为内部自动切分，静默忽略
        """
        # 类别特征 名称→索引 映射，注入构造参数
        if categorical_feature:
            if hasattr(X, "columns"):
                indices = [X.columns.get_loc(col) for col in categorical_feature]
                self.model.set_params(categorical_features=indices)
                self.params["categorical_features"] = indices
            else:
                logger.warning(
                    f"{self.log_prefix} categorical_feature 需要 DataFrame 输入做名称→索引映射，"
                    f"当前输入为 {type(X).__name__}，已忽略。"
                )
        # 模型训练
        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        预测
        """
        if not self.is_fitted:
            raise ValueError(f"{self.log_prefix} 模型尚未训练(Model not fitted yet).")

        return self.model.predict(X)
