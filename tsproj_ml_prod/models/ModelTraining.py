# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ModelTraining.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-03-29
# * Version     : 1.0.032909
# * Description : 生产环境模型训练模块
# * Link        : link
# * Requirement : lightgbm, catboost, scikit-learn, pandas, numpy
# ***************************************************

# python libraries
import copy
import os
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.multioutput import MultiOutputRegressor, RegressorChain

from tsproj_ml_prod.models.ModelFactory import ModelFactory
from tsproj_ml_prod.utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Trainer:

    def __init__(self, args: Dict, log_prefix: str):
        self.args = args
        self.log_prefix = log_prefix
        self.model_factory = ModelFactory(log_prefix=log_prefix)
        self.model_type = getattr(self.args, "model_type", "lightgbm")
        self.model_param_overrides = copy.deepcopy(getattr(self.args, "model_params", {}) or {})
        self.model_params = self.model_factory.resolve_model_params(
            self.model_type,
            self.model_param_overrides,
        )
        self._apply_model_thread_limits()

    def _resolve_worker_count(self, attr_name: str, default: int = 1) -> int:
        value = int(getattr(self.args, attr_name, default) or default)
        if value <= 0:
            cpu_count = os.cpu_count() or 1
            return max(1, cpu_count)

        return value

    def _apply_model_thread_limits(self):
        if not hasattr(self.args, "model_thread_count"):
            return
        raw_value = getattr(self.args, "model_thread_count", None)
        if raw_value is None:
            return
        model_threads = self._resolve_worker_count("model_thread_count", default=1)
        mt = str(self.model_type).lower()
        if mt in ["lightgbm", "lgb"]:
            self.model_params["n_jobs"] = model_threads
        elif mt in ["catboost", "cat"]:
            self.model_params["thread_count"] = model_threads

    def _build_multi_output_model(self, base_estimator, n_outputs: int):
        """
        根据配置构建多输出策略模型
        """
        if n_outputs <= 1:
            return base_estimator
        strategy = str(getattr(self.args, "multi_output_strategy", "multioutput")).lower()
        if strategy == "regressor_chain":
            logger.info(f"{self.log_prefix} Multi-output strategy: RegressorChain")
            return RegressorChain(estimator=base_estimator)
        multi_output_n_jobs = self._resolve_worker_count("multi_output_n_jobs", default=1)
        logger.info(f"{self.log_prefix} Multi-output strategy: MultiOutputRegressor")

        return MultiOutputRegressor(estimator=base_estimator, n_jobs=multi_output_n_jobs)
    
    def train(self, X_train, Y_train, feature_scaler, target_scaler, categorical_features):
        """
        模型训练
        """
        # 训练集
        X_train_df = X_train.copy()
        Y_train_df = Y_train.copy()
        # ------------------------------
        # 数据增强 + 特征选择
        # ------------------------------
        # 生产环境关闭数据增强与特征选择，保留全部训练特征
        selected_features = X_train_df.columns.tolist()
        # ------------------------------
        # 学习率配置（固定 or 自动）
        # ------------------------------
        # 生产环境直接使用传入模型参数，不额外做学习率搜索
        # ------------------------------
        # 归一化/标准化
        # ------------------------------
        # 生产环境不做特征转换和目标转换，直接使用原始特征
        X_train_df_processed = X_train_df.copy()
        Y_train_df_processed = Y_train_df.copy()
        if Y_train_df_processed.ndim == 1:
            Y_train_df_processed = Y_train_df_processed.to_frame()
        
        # 生产环境保留类别特征列表，直接按模型原生方式传递
        lgbm_categorical = list(categorical_features or [])
        # ------------------------------
        # 模型训练
        # ------------------------------
        model_type = self.model_type
        estimator_wrapper = self.model_factory.create_model(
            model_type=model_type,
            model_params=self.model_params,
        )
        # ------------------------------
        # 单模型 - 点预测
        # ------------------------------
        if Y_train_df_processed.shape[1] == 1:
            logger.info(f"{self.log_prefix} Training single-output regressor...")
            logger.info(f"{self.log_prefix} {'-' * 71}")
            model = estimator_wrapper
            fit_kwargs = {}
            if str(model_type).lower() in ["lightgbm", "lgb"] and lgbm_categorical is not None:
                fit_kwargs["categorical_feature"] = lgbm_categorical
            if str(model_type).lower() in ["catboost", "cat"] and lgbm_categorical is not None:
                fit_kwargs["categorical_feature"] = lgbm_categorical
            model.fit(X_train_df_processed, np.ravel(Y_train_df_processed.values), **fit_kwargs)
        else:
            logger.info(f"{self.log_prefix} Training multi-output regressor with {Y_train_df_processed.shape[1]} outputs...")
            logger.info(f"{self.log_prefix} {'-' * 71}")
            model = self._build_multi_output_model(estimator_wrapper.model, n_outputs=Y_train_df_processed.shape[1])
            try:
                model.fit(X_train_df_processed, Y_train_df_processed)
            except PermissionError as e:
                if hasattr(model, "n_jobs") and getattr(model, "n_jobs", 1) != 1:
                    logger.warning(f"{self.log_prefix} Multi-output parallel training failed, fallback to n_jobs = 1. error: {e}")
                    model.set_params(n_jobs=1)
                    model.fit(X_train_df_processed, Y_train_df_processed)
                else:
                    raise
        logger.info(f"{self.log_prefix} Model training completed!")

        return model, feature_scaler, target_scaler, selected_features

# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()
