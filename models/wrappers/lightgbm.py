"""lightgbm: estimator wrappers extracted from the model factory."""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from utils.log_util import logger
from models.preflight.lightgbm import validate_lgbm_params
from models.preflight import filter_fit_params as _filter_fit_params
from models.wrappers.base import BaseModel, DEFAULT_EARLY_STOPPING_ROUNDS


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

    def _resolve_params(self, supplied: Dict[str, Any]) -> Dict[str, Any]:
        # 合并默认参数后按 LightGBM 原生参数及别名全集严格校验
        merged_params = super()._resolve_params(supplied)
        validate_lgbm_params(merged_params)
        return merged_params

    def _build_estimator(self):
        return lgb.LGBMRegressor(**self.params)

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
        assert self.model is not None  # _build_estimator 已构造
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测
        """
        self._require_fitted()
        assert self.model is not None

        return self.model.predict(X)
