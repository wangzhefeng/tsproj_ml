# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ModelEnsemble.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-11
# * Version     : 1.0.021117
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from typing import List, Any
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
# model evaluation
from sklearn.metrics import (
    r2_score,                        # R2
    mean_squared_error,              # MSE
    root_mean_squared_error,         # RMSE
    mean_absolute_error,             # MAE
    mean_absolute_percentage_error,  # MAPE
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger

"""
模型融合模块 (Model Ensemble)
============================

提供多种模型融合策略，提升预测精度

支持的融合方法:
1. Voting - 投票法
    - Hard Voting（硬投票）
    - Soft Voting（软投票）
2. Averaging - 平均法
    - Simple Average（简单平均）
    - Weighted Average（加权平均）
4. Stacking - 堆叠法
    - 两层堆叠
    - 多层堆叠
5. Blending（混合）
"""


class ModelEnsemble:
    """
    模型融合类
    
    支持三种融合策略:
    1. average: 简单平均
    2. weighted: 加权平均（权重可优化）
    3. stacking: 堆叠集成（两层模型）
    """
    
    def __init__(self, models: List[tuple[str, Any]], method: str = 'average'):
        """
        Args:
            models: 模型列表[(name, model), ...]
            method: 融合方法 ['voting', 'averaging', 'stacking', 'blending']
        """
        self.models = models
        self.method = method
        self.meta_model = None  # 用于stacking
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        训练所有基模型和元模型
        """
        # 训练基模型
        for name, model in self.models:
            logger.info(f"Training model {name}")
            model.fit(X_train, y_train)
        
        # 如果是stacking，训练元模型
        if self.method == 'stacking' and X_val is not None:
            # 获取基模型在验证集上的预测
            meta_features = self._get_meta_features(X_val)
            # 训练元模型
            self.meta_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05)
            self.meta_model.fit(meta_features, y_val)
        
        # 如果是加权平均，优化权重
        if self.method == 'weighted' and X_val is not None:
            self.optimize_weights(X_val, y_val)
    
    def predict(self, X):
        """
        融合预测
        """
        if self.method == 'voting':
            return self._voting_predict(X)
        elif self.method == 'averaging':
            return self._averaging_predict(X)
        elif self.method == 'weighted_averaging':
            return self._weighted_averaging_predict(X, self.weights)
        elif self.method == 'stacking':
            return self._stacking_predict(X)
        elif self.method == 'blending':
            return self._blending_predict(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _voting_predict(self, X):
        """投票预测（用于分类，回归用平均）"""
        return self._averaging_predict(X)
    
    def _averaging_predict(self, X):
        """平均预测"""
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)
    
    def _weighted_averaging_predict(self, X, weights):
        """加权平均预测"""
        predictions = np.array([model.predict(X) for model in self.models])
        return np.average(predictions, axis=0, weights=weights)
    
    def _stacking_predict(self, X):
        """堆叠预测"""
        # 获取基模型预测作为元特征
        meta_features = self._get_meta_features(X)
        # 使用元模型预测
        return self.meta_model.predict(meta_features)
    
    def _blending_predict(self, X):
        """混合预测（类似stacking但使用固定权重）"""
        # 简化版：使用平均
        return self._averaging_predict(X)
    
    def _get_meta_features(self, X):
        """获取元特征（基模型的预测）"""
        meta_features = [model.predict(X) for model in self.models]
        return np.column_stack(meta_features)

    def optimize_weights(self, X_val, y_val):
        """
        优化加权平均的权重
        """
        from scipy.optimize import minimize
        
        def objective(weights):
            preds = [model.predict(X_val) for _, model in self.models]
            ensemble_pred = np.average(preds, axis=0, weights=weights)
            return mean_squared_error(y_val, ensemble_pred)
        
        n_models = len(self.models)
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1)] * n_models
        initial = np.ones(n_models) / n_models
        
        result = minimize(
            objective, 
            initial, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        self.weights = result.x


# ##############################
# 使用示例
# ##############################
from models.ModelFactory import LightGBMModel, XGBoostModel, CatBoostModel
def train_with_ensemble(X_train, y_train, X_val, y_val):
    # 创建多个模型
    models = {
        "lightgmb": LightGBMModel({'n_estimators': 1000, 'learning_rate': 0.05}),
        "xgboost": XGBoostModel({'n_estimators': 1000, 'learning_rate': 0.05}),
        "catboost": CatBoostModel({'iterations': 1000, 'learning_rate': 0.05}),
    }
    
    # 创建融合器
    ensemble = ModelEnsemble(models, method='stacking')
    
    # 训练
    ensemble.fit(X_train, y_train, X_val, y_val)
    
    # 预测
    y_pred = ensemble.predict(X_val)
    
    return ensemble, y_pred




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
