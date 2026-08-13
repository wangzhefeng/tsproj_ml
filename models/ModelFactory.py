# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ModelFactory.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-11
# * Version     : 1.0.021110
# * Description : 生产环境模型工厂
# * Link        : link
# * Requirement : lightgbm, xgboost, catboost, scikit-learn, pandas, numpy
# ***************************************************

# python libraries
import copy
import inspect
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cab
from scipy.optimize import nnls
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso, QuantileRegressor

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

"""
模型抽象层 (Model Abstraction Layer)
====================================

提供统一的模型接口，支持多种机器学习模型的无缝切换

支持的模型:
- LightGBM
- XGBoost
- CatBoost
- Random Forest
- HistGradientBoosting（sklearn 原生直方图 GBDT）
- 线性基线: Ridge / ElasticNet / LASSO / QuantileRegressor

使用示例:
    # 创建模型
    model = ModelFactory().create_model('lightgbm', params)

    # 训练
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
"""

# 各模型 fit 接口统一的早停轮数默认值（调用方通常显式传 patience，此值仅为兜底）
DEFAULT_EARLY_STOPPING_ROUNDS = 50


# ##############################
# 定义模型基类
# ##############################
class BaseModel(ABC):
    """
    模型基类 (Base Model Class)

    所有具体模型必须继承此类并实现抽象方法
    """

    def __init__(self, params: Dict[str, Any], log_prefix: str="BaseModel", log_params: bool = True):
        """
        初始化模型

        Args:
            params (Dict[str, Any]): 模型参数字典
        """
        self.log_prefix = log_prefix
        self.params = params
        self.model = None
        self.is_fitted = False
        self.log_params = log_params

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "BaseModel":
        """
        训练模型

        Args:
            X: 特征数据
            y: 目标数据
            **kwargs: 其他参数（如验证集、类别特征等）

        Returns:
            模型实例自身（支持链式调用）
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

    def get_feature_importance(self, X: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        """
        获取特征重要性

        Args:
            X: 特征数据（可选，仅用于取列名；非 DataFrame 时回退为 feature_{idx}）

        Returns:
            特征重要性数组，如果模型不支持则返回 None
        """
        importance = getattr(self.model, "feature_importances_", None)
        if importance is None:
            return None
        importance = np.asarray(importance)
        columns = getattr(X, "columns", None)
        top_features = np.argsort(importance)[-5:][::-1]
        lines = []
        for rank, idx in enumerate(top_features, 1):
            if columns is not None and idx < len(columns):
                name = str(columns[idx])
            else:
                name = f"feature_{idx}"
            lines.append(f"{rank}. {name}: {importance[idx]:.4f}")
        logger.info(f"{self.log_prefix} top-5 important features:\n" + "\n".join(lines))

        return importance


def _filter_valid_params(params: Dict[str, Any], estimator_cls) -> Dict[str, Any]:
    """
    按估计器 ``__init__`` 签名过滤模型参数，仅保留该估计器显式声明的合法参数。

    对于通过 ``**kwargs`` 透传原生参数的封装（签名含 VAR_KEYWORD），
    无法用显式签名做白名单，直接原样返回，避免误删合法配置。
    """
    signature = inspect.signature(estimator_cls.__init__)
    # 部分 sklearn 风格封装通过 **kwargs 接收额外原生参数，
    # 这类模型不能用显式签名做白名单过滤，否则会错误丢弃合法配置。
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(params)
    valid = set(signature.parameters.keys())
    valid.discard("self")
    return {k: v for k, v in params.items() if k in valid}


def _filter_fit_params(model, fit_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    按底层估计器 ``fit`` 签名过滤训练参数，兼容不同版本 sklearn API 的参数差异
    （例如 lightgbm >= 4.x 的 fit 不再接受 verbose）。
    """
    supported = set(inspect.signature(model.fit).parameters.keys())
    return {k: v for k, v in fit_params.items() if k in supported}


def _warn_unrecognized_lgbm_params(params: Dict[str, Any], log_prefix: str) -> None:
    """
    对 LightGBM 参数做弱校验（仅告警、不拦截）。

    LGBMRegressor 通过 ``**kwargs`` 透传原生 booster 参数，签名白名单对其无效，
    拼写错误的参数会被静默忽略。合法名全集 = sklearn 封装显式参数 +
    原生参数及其别名（``_ConfigAliases``）。内省失败（如 lightgbm 版本差异）时安全跳过。
    """
    try:
        explicit = {p for p in inspect.signature(lgb.LGBMRegressor.__init__).parameters if p != "self"}
        alias_map = lgb.basic._ConfigAliases._get_all_param_aliases()
        valid = explicit | set(alias_map) | {a for aliases in alias_map.values() for a in aliases}
    except Exception:
        return
    unknown = sorted(k for k in params if k not in valid)
    if unknown:
        logger.warning(
            f"{log_prefix} 以下 LightGBM 参数无法识别（可能存在拼写错误），"
            f"将原样透传给原生 booster: {unknown}"
        )


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
        "n_jobs": -1,
        "random_state": 42,
        "force_col_wise": True,
    }

    def __init__(self, params: Dict[str, Any], log_prefix: str="LightGBMModel", log_params: bool = True):
        super().__init__(params, log_prefix=log_prefix, log_params=log_params)
        merged_params = {**copy.deepcopy(self.DEFAULT_PARAMS), **(params or {})}
        # 模型参数
        self.params = merged_params
        # 弱校验：**kwargs 透传导致拼写错误静默生效，仅告警不拦截
        _warn_unrecognized_lgbm_params(self.params, log_prefix)
        if self.log_params:
            logger.info(f"{log_prefix} model parameters: \n{self.params}")
        # 模型构建
        self.model = lgb.LGBMRegressor(**self.params)

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            categorical_feature: Optional[list] = None,
            eval_set: Optional[tuple] = None,
            eval_metric: str = "mae",
            early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
            verbose: bool = False,
            sample_weight: Optional[Any] = None,
            **kwargs):
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
            sample_weight: 训练样本权重(例如时间衰减权重)
            **kwargs: 为跨模型统一接口而保留，静默忽略
        """
        # 设置训练参数
        fit_params: Dict[str, Any] = {"verbose": verbose}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["eval_metric"] = eval_metric
            fit_params["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=verbose)]
        if categorical_feature is not None:
            fit_params["categorical_feature"] = categorical_feature
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        # 兼容不同 lightgbm 版本的 sklearn API 参数差异（例如 fit 不再接受 verbose）
        fit_params = _filter_fit_params(self.model, fit_params)
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


class XGBoostModel(BaseModel):
    """
    XGBoost模型封装

    特点:
    - 性能优秀
    - 正则化能力强
    - GPU加速支持
    - 广泛应用
    """

    DEFAULT_PARAMS = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "random_state": 42,
    }

    def __init__(self, params: Dict[str, Any], log_prefix: str="XGBoostModel", log_params: bool = True):
        super().__init__(params, log_prefix=log_prefix, log_params=log_params)
        merged_params = {**copy.deepcopy(self.DEFAULT_PARAMS), **(params or {})}
        # 模型参数
        self.params = _filter_valid_params(merged_params, xgb.XGBRegressor)
        if self.log_params:
            logger.info(f"{log_prefix} model parameters: \n{self.params}")
        # 模型构建
        self.model = xgb.XGBRegressor(**self.params)

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            eval_set: Optional[tuple] = None,
            eval_metric: Optional[str] = None,
            early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
            verbose: bool = False,
            sample_weight: Optional[Any] = None,
            **kwargs):
        """
        训练XGBoost模型

        Args:
            X: 训练特征
            y: 训练目标
            eval_set: 验证集 [(X_val, y_val)]
            eval_metric: 评估指标。xgboost >= 2.0 起为构造参数，此处显式传入时
                通过 set_params 注入构造参数（缺省 None 表示沿用构造参数）
            early_stopping_rounds: 早停轮数。xgboost >= 2.0 起为构造参数，
                仅在提供 eval_set 时注入（无验证集时设置会直接报错）
            verbose: 是否显示训练过程
            sample_weight: 训练样本权重(例如时间衰减权重)
            **kwargs: 为跨模型统一接口而保留，静默忽略
        """
        # 设置训练参数
        fit_params: Dict[str, Any] = {"verbose": verbose}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            # xgboost >= 2.0 的 fit 不再接受 eval_metric / early_stopping_rounds，
            # 二者是构造参数，需通过 set_params 注入
            if eval_metric is not None:
                self.model.set_params(eval_metric=eval_metric)
                self.params["eval_metric"] = eval_metric
            if early_stopping_rounds:
                self.model.set_params(early_stopping_rounds=early_stopping_rounds)
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        # 兼容不同 xgboost 版本的 sklearn API 参数差异
        fit_params = _filter_fit_params(self.model, fit_params)
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

    def __init__(self, params: Dict[str, Any], log_prefix: str="CatBoostModel", log_params: bool = True):
        super().__init__(params, log_prefix=log_prefix, log_params=log_params)
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


class _LinearModelBase(BaseModel):
    """
    线性模型封装基类（Ridge / ElasticNet / LASSO / QuantileRegressor）

    四者接口完全一致（fit(X, y, sample_weight)），差异仅在估计器类与默认参数。

    注意:
    - 线性模型对特征量纲敏感，建议开启 scale_features / scale_target
    - 不识别类别特征语义，类别列需先编码（encode_categorical_features）
    - 不支持验证集与早停，相关 fit kwargs 静默忽略
    """

    ESTIMATOR_CLS = None
    DEFAULT_PARAMS: Dict[str, Any] = {}

    def __init__(self, params: Dict[str, Any], log_prefix: str="LinearModel", log_params: bool = True):
        super().__init__(params, log_prefix=log_prefix, log_params=log_params)
        # 参数合并（用户参数优先，避免被默认值覆盖）
        merged_params = {**copy.deepcopy(self.DEFAULT_PARAMS), **(params or {})}
        # 模型参数
        self.params = _filter_valid_params(merged_params, self.ESTIMATOR_CLS)
        if self.log_params:
            logger.info(f"{log_prefix} model parameters: \n{self.params}")
        # 模型构建
        if self.ESTIMATOR_CLS is None:
            raise NotImplementedError(f"{self.log_prefix} 子类必须声明 ESTIMATOR_CLS")
        self.model = self.ESTIMATOR_CLS(**self.params)

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            sample_weight: Optional[Any] = None,
            **kwargs):
        """
        训练线性模型

        Args:
            X: 训练特征
            y: 训练目标
            sample_weight: 训练样本权重(例如时间衰减权重)
            **kwargs: 为跨模型统一接口而保留（categorical_feature / eval_set /
                early_stopping_rounds 等），线性模型不支持，静默忽略

        注意: 线性估计器不容忍 NaN——fit 丢弃含 NaN 的行（训练窗起始行的长滞后
        特征为 NaN，GBDT 原生容忍、线性模型必须处理），并记录列中位数供
        predict 端填补。
        """
        if hasattr(X, "notna"):
            yv = np.asarray(y, dtype=float).ravel()
            mask = X.notna().all(axis=1).to_numpy() & np.isfinite(yv)
            self._col_medians = X.median().to_numpy(dtype=float)
            self._columns = list(X.columns)
            X_fit = X[mask]
            y_fit = yv[mask]
            sw_fit = np.asarray(sample_weight, dtype=float)[mask] if sample_weight is not None else None
        else:
            X_fit, y_fit, sw_fit = X, y, sample_weight
        self.model.fit(X_fit, y_fit, sample_weight=sw_fit)
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        预测（NaN 用训练期列中位数填补；正常预测路径无 NaN，属防御性处理）
        """
        if not self.is_fitted:
            raise ValueError(f"{self.log_prefix} 模型尚未训练(Model not fitted yet).")

        if hasattr(X, "isna") and getattr(self, "_col_medians", None) is not None and X.isna().any().any():
            X = X.fillna(pd.Series(self._col_medians, index=self._columns))

        return self.model.predict(X)


class RidgeModel(_LinearModelBase):
    """Ridge 岭回归（L2 正则线性基线）"""

    ESTIMATOR_CLS = Ridge
    DEFAULT_PARAMS = {
        "alpha": 1.0,
        "fit_intercept": True,
        "random_state": 42,
    }


class ElasticNetModel(_LinearModelBase):
    """ElasticNet 回归（L1 + L2 混合正则线性基线）"""

    ESTIMATOR_CLS = ElasticNet
    DEFAULT_PARAMS = {
        "alpha": 1.0,
        "l1_ratio": 0.5,
        "max_iter": 2000,
        "random_state": 42,
    }


class LassoModel(_LinearModelBase):
    """LASSO 回归（L1 正则线性基线，带特征选择效果）"""

    ESTIMATOR_CLS = Lasso
    DEFAULT_PARAMS = {
        "alpha": 1.0,
        "max_iter": 2000,
        "random_state": 42,
    }


class QuantileRegressorModel(_LinearModelBase):
    """
    QuantileRegressor 线性分位数回归基线

    默认 quantile=0.5 + alpha=0.0，即不加正则的中位数回归（MAE 口径的线性对照）。
    分位数预测路径由 Trainer._inject_quantile_params 注入目标 quantile。
    """

    ESTIMATOR_CLS = QuantileRegressor
    DEFAULT_PARAMS = {
        "quantile": 0.5,
        "alpha": 0.0,
        "solver": "highs",
    }


class SeasonalTemplateModel(BaseModel):
    """
    季节模板基线（seasonal template / climatology / 相似日法）

    不学习一般的 特征→目标 映射，而是把「历史周期同一时刻的实测值」
    （特征工程生成的 {target}_lag_{lag} 滞后列）按非负权重重放：

        ŷ = Σ_j w_j · y_lag_j,  w_j >= 0

    - 权重由 NNLS 在训练集上学习（equal_weight=True 时退化为等权 climatology）；
    - day_type_split=True 且存在 dt_day_of_week 列时，工作日/周末各学一组权重；
    - 特例 w=[1,0,...,0] 即 seasonal naive（昨日同时刻，对应 ModelTesting 的
      _build_seasonal_naive 评估对照）。

    误差来源（形态持续性假设：平常日强、节假日/突变日失效）与 GBDT/线性成员
    结构性不同，主要用作模型融合的多样性成员；也可作为单模型基线。
    在 USMR 递归且 horizon <= 最短滞后阶数时，滞后列全程引用实测值，
    模板不会被模型自身预测污染。

    注意:
    - 依赖特征矩阵中的滞后列（自动识别 ^.+_lag_\\d+$，按滞后阶数排序），
      无滞后列时 raise；
    - fit 自动丢弃含 NaN 的行（训练窗起始行的长滞后特征为 NaN）；
    - predict 端 NaN 用训练期滞后列中位数填补（防御性，正常不出现）；
    - 无线程参数；不支持分位数目标（Trainer 注入时走 warning 分支）。
    """

    DEFAULT_PARAMS = {
        "day_type_split": True,   # 工作日/周末分组建模板
        "equal_weight": False,    # True=等权 climatology；False=NNLS 学权重
        "min_group_samples": 10,  # 分组样本不足该数时回退全局权重
    }

    LAG_COL_PATTERN = r"^.+_lag_\d+$"
    DOW_COL = "dt_day_of_week"

    def __init__(self, params: Dict[str, Any], log_prefix: str="SeasonalTemplateModel", log_params: bool = True):
        super().__init__(params, log_prefix=log_prefix, log_params=log_params)
        # 参数合并（用户参数优先，避免被默认值覆盖）
        merged_params = {**copy.deepcopy(self.DEFAULT_PARAMS), **(params or {})}
        self.params = merged_params
        if self.log_params:
            logger.info(f"{log_prefix} model parameters: \n{self.params}")
        # 无底层估计器：非负模板权重即模型
        self.model = None
        self.lag_columns_ = None
        self.weights_ = None
        self.group_weights_ = None
        self.lag_medians_ = None

    def _detect_lag_columns(self, X: pd.DataFrame) -> list:
        """自动识别滞后列并按滞后阶数排序，保证 fit/predict 列序一致"""
        cols = [c for c in X.columns if re.match(self.LAG_COL_PATTERN, str(c))]
        cols.sort(key=lambda c: int(str(c).rsplit("_lag_", 1)[1]))
        return cols

    def _fit_weights(self, Z: np.ndarray, y: np.ndarray) -> np.ndarray:
        """NNLS 学非负模板权重；equal_weight=True 时等权；数值异常时退化等权"""
        n_lags = Z.shape[1]
        if self.params.get("equal_weight"):
            return np.full(n_lags, 1.0 / n_lags)
        w, _ = nnls(Z, y)
        if not np.isfinite(w).all() or w.sum() <= 1e-12:
            logger.warning(f"{self.log_prefix} NNLS 权重异常，退化为等权模板。")
            w = np.full(n_lags, 1.0 / n_lags)
        return w

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            **kwargs):
        """
        训练季节模板基线

        Args:
            X: 训练特征（DataFrame，需含 {target}_lag_{lag} 滞后列）
            y: 训练目标
            **kwargs: 为跨模型统一接口而保留（sample_weight / eval_set 等），静默忽略
        """
        if not hasattr(X, "columns"):
            raise ValueError(f"{self.log_prefix} 需要 DataFrame 输入以识别滞后列，当前为 {type(X).__name__}")
        lag_cols = self._detect_lag_columns(X)
        if not lag_cols:
            raise ValueError(
                f"{self.log_prefix} 特征中未找到滞后列（模式 {self.LAG_COL_PATTERN}），"
                f"无法构建季节模板。"
            )
        self.lag_columns_ = lag_cols
        Z_df = X[lag_cols]
        yv = np.asarray(y, dtype=float).ravel()
        # fit 端丢弃含 NaN 的行（训练窗起始行的长滞后特征为 NaN）
        mask = Z_df.notna().all(axis=1).to_numpy() & np.isfinite(yv)
        Z_c = Z_df.to_numpy(dtype=float)[mask]
        y_c = yv[mask]
        if len(y_c) == 0:
            raise ValueError(f"{self.log_prefix} 滞后列全部含 NaN，无有效训练行。")
        # predict 端 NaN 填补用的训练期中位数
        self.lag_medians_ = Z_df.median().to_numpy(dtype=float)
        # 全局权重
        self.weights_ = self._fit_weights(Z_c, y_c)
        logger.info(f"{self.log_prefix} template weights (global): {np.round(self.weights_, 4)}")
        # 工作日/周末分组权重（两组样本都足够才启用，否则回退全局）
        self.group_weights_ = None
        if self.params.get("day_type_split") and self.DOW_COL in X.columns:
            dow = X[self.DOW_COL].to_numpy()[mask]
            min_n = int(self.params.get("min_group_samples", 10))
            group_weights = {}
            for group_name, group_mask in [("weekday", dow <= 4), ("weekend", dow >= 5)]:
                if int(group_mask.sum()) >= min_n:
                    group_weights[group_name] = self._fit_weights(Z_c[group_mask], y_c[group_mask])
            if len(group_weights) == 2:
                self.group_weights_ = group_weights
                logger.info(
                    f"{self.log_prefix} day-type split enabled: "
                    f"weekday={np.round(group_weights['weekday'], 4)}, "
                    f"weekend={np.round(group_weights['weekend'], 4)}"
                )
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        预测
        """
        if not self.is_fitted:
            raise ValueError(f"{self.log_prefix} 模型尚未训练(Model not fitted yet).")
        assert self.lag_columns_ is not None and self.weights_ is not None
        assert self.lag_medians_ is not None
        Z = X[self.lag_columns_].to_numpy(dtype=float)
        # 防御性 NaN 填补（horizon <= 最短滞后时预测端 lag 全为实测，不会出现）
        if np.isnan(Z).any():
            nan_rows, nan_cols = np.where(np.isnan(Z))
            Z[nan_rows, nan_cols] = self.lag_medians_[nan_cols]
        if self.group_weights_ is not None and self.DOW_COL in X.columns:
            dow = X[self.DOW_COL].to_numpy()
            is_weekday = dow <= 4
            out = np.empty(len(Z), dtype=float)
            out[is_weekday] = Z[is_weekday] @ self.group_weights_["weekday"]
            out[~is_weekday] = Z[~is_weekday] @ self.group_weights_["weekend"]
            return out

        return Z @ self.weights_

    def get_feature_importance(self, X: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        """模板权重即各滞后列的重要性"""
        if self.weights_ is None or self.lag_columns_ is None:
            return None
        lines = [f"{name}: {w:.4f}" for name, w in zip(self.lag_columns_, self.weights_)]
        logger.info(f"{self.log_prefix} template weights:\n" + "\n".join(lines))

        return self.weights_


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
        "xgboost": XGBoostModel,
        "xgb": XGBoostModel,
        "catboost": CatBoostModel,
        "cat": CatBoostModel,
        "randomforest": RandomForestModel,
        "rf": RandomForestModel,
        "histgb": HistGBModel,
        "histgradientboosting": HistGBModel,
        "ridge": RidgeModel,
        "elasticnet": ElasticNetModel,
        "enet": ElasticNetModel,
        "lasso": LassoModel,
        "quantileregressor": QuantileRegressorModel,
        "qr": QuantileRegressorModel,
        "seasonaltemplate": SeasonalTemplateModel,
        "st": SeasonalTemplateModel,
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

    def create_model(self, model_type: str, model_params: Dict[str, Any], log_params: bool = True) -> BaseModel:
        """
        创建模型实例

        Args:
            model_type: 模型类型 ('lightgbm', 'xgboost', 'catboost', 'randomforest')
            model_params: 模型参数字典（缺省参数由各模型封装的 DEFAULT_PARAMS 补齐）
            log_params: 是否记录模型参数日志

        Returns:
            模型实例

        Raises:
            ValueError: 如果模型类型不支持

        Examples:
            >>> params = {'n_estimators': 1000, 'learning_rate': 0.05}
            >>> model = ModelFactory().create_model('lightgbm', params)
            >>> model.fit(X_train, y_train)
            >>> y_pred = model.predict(X_test)
        """
        # 模型类型
        model_type = self._normalize_model_type(model_type)
        # 创建模型实例（默认参数在模型封装内部合并，此处不做重复 merge）
        model_class = ModelFactory._models[model_type]

        return model_class(
            model_params,
            log_prefix=f"{self.log_prefix} {model_type.capitalize()}",
            log_params=log_params,
        )

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
