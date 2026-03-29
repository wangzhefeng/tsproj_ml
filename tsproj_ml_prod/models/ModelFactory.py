# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ModelFactory.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-03-29
# * Version     : 1.0.032909
# * Description : 生产环境模型工厂
# * Link        : link
# * Requirement : lightgbm, catboost, pandas, numpy
# ***************************************************

# python libraries
import copy
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cab

from tsproj_ml_prod.utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

"""
模型抽象层 (Model Abstraction Layer)
====================================

提供统一的模型接口，支持多种机器学习模型的无缝切换

支持的模型:
- LightGBM
- CatBoost

使用示例:
    # 创建模型
    model = ModelFactory.create_model('lightgbm', params)
    
    # 训练
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
"""


# ##############################
# 定义模型基类
# ##############################
class BaseModel(ABC):
    """
    模型基类 (Base Model Class)
    
    所有具体模型必须继承此类并实现抽象方法
    """
    
    def __init__(self, params: Dict[str, Any], log_prefix: str="BaseModel"):
        """
        初始化模型
        
        Args:
            params (Dict[str, Any]): 模型参数字典
        """
        self.log_prefix = log_prefix
        self.params = params
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        训练模型
        
        Args:
            X: 特征数据
            y: 目标数据
            **kwargs: 其他参数（如验证集、类别特征等）
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征数据
            **kwargs: 其他参数
        
        Returns:
            预测结果
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        return self.params
    
    def get_feature_importance(self, X) -> Optional[np.ndarray]:
        """
        获取特征重要性
        
        Returns:
            特征重要性数组，如果模型不支持则返回None
        """
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            if importance is not None:
                print(f"前 5 个重要特征:")
                top_features = np.argsort(importance)[-5:][::-1]
                for i, idx in enumerate(top_features, 1):
                    print(f"{i}. {X.columns[idx]}: {importance[idx]:.4f}")
                return importance
        return None


# TODO 这个函数的作用是什么？
def _filter_valid_params(params: Dict[str, Any], estimator_cls) -> Dict[str, Any]:
    signature = inspect.signature(estimator_cls.__init__)
    # 部分 sklearn 风格封装通过 **kwargs 接收额外原生参数，
    # 这类模型不能用显式签名做白名单过滤，否则会错误丢弃合法配置。
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(params)
    valid = set(signature.parameters.keys())
    valid.discard("self")
    return {k: v for k, v in params.items() if k in valid}

# ##############################
# 具体模型实现
# ##############################
class LightGBMModel(BaseModel):
    """
    LightGBM 模型封装
    
    特点:
    - 训练速度快
    - 内存占用小
    - 支持类别特征
    - 适合大数据集
    """

    DEFAULT_PARAMS = {
        "boosting_type": "gbdt",
        "objective": "regression_l1",
        "metric": "mae",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_bin": 63,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "n_jobs": 1,
        "random_state": 42,
        "force_col_wise": True,
    }

    def __init__(self, params: Dict[str, Any], log_prefix: str="LightGBMModel"):
        super().__init__(params)
        merged_params = {**copy.deepcopy(self.DEFAULT_PARAMS), **(params or {})}
        # 模型参数
        self.params = merged_params
        logger.info(f"{log_prefix} model parameters: \n{self.params}")
        # 模型构建
        self.model = lgb.LGBMRegressor(**self.params)
    
    def fit(self, 
            X: pd.DataFrame,
            y: pd.Series,
            categorical_feature: Optional[list] = None,
            eval_set: Optional[tuple] = None,
            eval_metric: str = "mae",
            early_stopping_rounds: int = 100,
            verbose: bool = False):
        """
        训练 LightGBM 模型
        
        Args:
            X: 训练特征
            y: 训练目标
            categorical_feature: 类别特征列表
            eval_set: 验证集 [(X_val, y_val)]
            eval_metric: 评估指标
            early_stopping_rounds: 早停轮数
            verbose: 是否显示训练过程
        """
        # 设置训练参数
        fit_params = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["eval_metric"] = eval_metric
            fit_params["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=verbose)]
        if verbose is not None:
            fit_params["verbose"] = verbose
        if categorical_feature is not None:
            fit_params["categorical_feature"] = categorical_feature
        # 兼容不同 lightgbm 版本的 sklearn API 参数差异（例如 fit 不再接受 verbose）
        supported_fit_params = set(inspect.signature(self.model.fit).parameters.keys())
        fit_params = {k: v for k, v in fit_params.items() if k in supported_fit_params}
        # 模型训练
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测
        """
        if not self.is_fitted:
            raise ValueError(f"{self.log_prefix} 模型尚未训练(Model not fitted yet).")
        
        return self.model.predict(X)


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

    def __init__(self, params: Dict[str, Any], log_prefix: str="CatBoostModel"):
        super().__init__(params)
        merged_params = {**copy.deepcopy(self.DEFAULT_PARAMS), **(params or {})}
        # CatBoost 的 iterations / n_estimators / num_boost_round / num_trees 是同义参数，只能保留一个
        iteration_aliases = ["n_estimators", "num_boost_round", "num_trees"]
        for alias in iteration_aliases:
            if alias in merged_params:
                merged_params["iterations"] = merged_params.pop(alias)
        # CatBoost 的 random_seed / random_state 是同义参数，只能保留一个
        if "random_seed" in merged_params and "random_state" in merged_params:
            merged_params.pop("random_state", None)
        # 清理从其他树模型配置复用过来的不兼容参数
        incompatible_params = ["num_leaves", "feature_fraction", "bagging_fraction", "bagging_freq", "force_col_wise"]
        for param in incompatible_params:
            merged_params.pop(param, None)
        # 模型参数
        self.params = _filter_valid_params(merged_params, cab.CatBoostRegressor)
        logger.info(f"{log_prefix} model parameters: \n{self.params}")
        # 模型构建
        self.model = cab.CatBoostRegressor(**self.params)
        self.log_prefix = log_prefix
    
    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            categorical_feature: Optional[list] = None,
            eval_set: Optional[tuple] = None,
            eval_metric: str = "mae",
            early_stopping_rounds: int = 50):
        """
        训练 CatBoost 模型
        
        Args:
            X: 训练特征
            y: 训练目标
            eval_set: 验证集 (X_val, y_val)
            eval_metric: 评估指标
            categorical_feature: 类别特征列表
            early_stopping_rounds: 早停轮数
        """
        # 设置训练参数
        fit_params = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["eval_metric"] = eval_metric
            fit_params["early_stopping_rounds"] = early_stopping_rounds
        if categorical_feature is not None:
            fit_params["cat_features"] = categorical_feature
        # 模型训练
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True

        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测
        """
        if not self.is_fitted:
            raise ValueError(f"{self.log_prefix} 模型尚未训练")
        
        return self.model.predict(X)

# ##############################
# 模型工厂
# ##############################
class ModelFactory:
    """
    模型工厂 (Model Factory)
    
    用于创建不同类型的模型实例
    """
    # 支持的模型映射
    _models = {
        "lightgbm": LightGBMModel,
        "lgb": LightGBMModel,
        "catboost": CatBoostModel,
        "cat": CatBoostModel,
    }

    def __init__(self, log_prefix: str = "ModelFactory"):
        self.log_prefix = log_prefix

    @staticmethod
    def _normalize_model_type(model_type: str) -> str:
        model_type = str(model_type).lower()
        if model_type not in ModelFactory._models:
            supported = ", ".join(ModelFactory._models.keys())
            raise ValueError(
                f"不支持的模型类型: {model_type}\n"
                f"支持的模型: {supported}"
            )
        return model_type

    @classmethod
    def get_default_model_params(cls, model_type: str) -> Dict[str, Any]:
        normalized_type = cls._normalize_model_type(model_type)
        model_class = cls._models[normalized_type]

        return copy.deepcopy(getattr(model_class, "DEFAULT_PARAMS", {}))

    @classmethod
    def resolve_model_params(cls, model_type: str, model_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        resolved = cls.get_default_model_params(model_type)
        resolved.update(copy.deepcopy(model_params or {}))

        return resolved
    
    def create_model(self, model_type: str, model_params: Dict[str, Any]) -> BaseModel:
        """
        创建模型实例
        
        Args:
            model_type: 模型类型 ('lightgbm', 'catboost')
            params: 模型参数字典
        
        Returns:
            模型实例
        
        Raises:
            ValueError: 如果模型类型不支持
        
        Examples:
            >>> params = {'n_estimators': 1000, 'learning_rate': 0.05}
            >>> model = ModelFactory.create_model('lightgbm', params)
            >>> model.fit(X_train, y_train)
            >>> y_pred = model.predict(X_test)
        """
        # 模型类型
        model_type = self._normalize_model_type(model_type)
        # 创建模型实例
        model_class = ModelFactory._models[model_type]
        resolved_params = self.resolve_model_params(model_type, model_params)
        
        return model_class(resolved_params, log_prefix=f"{self.log_prefix} {model_type.capitalize()}")
    
    @staticmethod
    def list_models() -> list:
        """
        列出所有支持的模型类型
        """
        return list(ModelFactory._models.keys())




# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()
