"""xgboost: estimator wrappers extracted from the model factory."""

import json
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from utils.log_util import logger
from models.preflight import filter_fit_params as _filter_fit_params, filter_valid_params as _filter_valid_params
from models.preflight.xgboost_estimator import validate_estimator as validate_xgb_estimator
from models.wrappers.base import BaseModel, DEFAULT_EARLY_STOPPING_ROUNDS


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

    def _resolve_params(self, supplied: Dict[str, Any]) -> Dict[str, Any]:
        # 合并默认参数后按 XGBRegressor 显式签名白名单校验
        return _filter_valid_params(super()._resolve_params(supplied), xgb.XGBRegressor)

    def _build_estimator(self):
        return xgb.XGBRegressor(**self.params)

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
        assert self.model is not None  # _build_estimator 已构造
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
        # 子进程预检：参数表面 + 维度 + 特征名（私有 API 与 RNG 防护收口在 preflight 层）
        self.parameter_validation = validate_xgb_estimator(self.model, X, y)
        # 模型训练
        self.model.fit(X, y, **fit_params)
        self.parameter_validation["fitted_config"] = json.loads(self.model.get_booster().save_config())
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测
        """
        self._require_fitted()
        assert self.model is not None

        return self.model.predict(X)
