"""xgboost: estimator wrappers extracted from the model factory."""

import copy
import json
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.data import pandas_feature_info
from utils.log_util import logger
from models.xgb_validation import validate_xgb_parameters
from models.wrappers.base import BaseModel, DEFAULT_EARLY_STOPPING_ROUNDS, _filter_valid_params, _filter_fit_params


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
        feature_names = None
        if isinstance(X, pd.DataFrame):
            names, _ = pandas_feature_info(
                X, meta=None, feature_names=None, feature_types=self.model.feature_types,
                enable_categorical=self.model.enable_categorical,
            )
            feature_names = tuple(names) if names is not None else None
        targets = np.asarray(y)
        # get_xgb_params 会从 RNG 对象抽取 seed；预检不得额外推进真实模型的 RNG。
        validation_model = copy.copy(self.model)
        validation_model.random_state = copy.deepcopy(self.model.random_state)
        self.parameter_validation = validate_xgb_parameters(
            validation_model.get_xgb_params(), num_features=X.shape[1],
            num_targets=targets.shape[1] if targets.ndim > 1 else 1,
            feature_names=feature_names,
        )
        # 模型训练
        self.model.fit(X, y, **fit_params)
        self.parameter_validation["fitted_config"] = json.loads(self.model.get_booster().save_config())
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测
        """
        if not self.is_fitted:
            raise ValueError(f"{self.log_prefix} 模型尚未训练(Model not fitted yet).")

        return self.model.predict(X)
