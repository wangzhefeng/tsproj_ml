# -*- coding: utf-8 -*-
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

import numpy as np
import pandas as pd
from typing import List
from model_abstraction import BaseModel
import lightgbm as lgb


class ModelEnsemble:
    """模型融合器"""
    
    def __init__(self, models: List[BaseModel], method='averaging', weights=None):
        self.models = models
        self.method = method
        self.weights = weights
        self.meta_model = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """训练所有基模型"""
        for i, model in enumerate(self.models):
            print(f"训练模型 {i+1}/{len(self.models)}")
            model.fit(X_train, y_train)
        
        if self.method == 'stacking' and X_val is not None:
            meta_features = self._get_meta_features(X_val)
            self.meta_model = lgb.LGBMRegressor(n_estimators=100)
            self.meta_model.fit(meta_features, y_val)
    
    def predict(self, X):
        """融合预测"""
        if self.method == 'averaging':
            predictions = [model.predict(X) for model in self.models]
            return np.mean(predictions, axis=0)
        elif self.method == 'weighted_averaging':
            predictions = [model.predict(X) for model in self.models]
            return np.average(predictions, axis=0, weights=self.weights)
        elif self.method == 'stacking':
            meta_features = self._get_meta_features(X)
            return self.meta_model.predict(meta_features)
        else:
            return self._averaging_predict(X)
    
    def _get_meta_features(self, X):
        """获取元特征"""
        return np.column_stack([model.predict(X) for model in self.models])



class ModelEnsemble:
    """模型融合器"""
    
    def __init__(self, models: List[BaseModel], method: str = 'voting'):
        """
        初始化
        
        Args:
            models: 模型列表
            method: 融合方法 ['voting', 'averaging', 'stacking', 'blending']
        """
        self.models = models
        self.method = method
        self.meta_model = None  # 用于stacking
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """训练所有模型"""
        # 训练基模型
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}")
            model.fit(X_train, y_train)
        
        # 如果是stacking，训练元模型
        if self.method == 'stacking' and X_val is not None:
            # 获取基模型在验证集上的预测
            meta_features = self._get_meta_features(X_val)
            
            # 训练元模型
            self.meta_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05)
            self.meta_model.fit(meta_features, y_val)
    
    def predict(self, X):
        """融合预测"""
        if self.method == 'voting':
            return self._voting_predict(X)
        elif self.method == 'averaging':
            return self._averaging_predict(X)
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
        return predictions.mean(axis=0)
    
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
        meta_features = []
        for model in self.models:
            pred = model.predict(X)
            meta_features.append(pred)
        
        return np.column_stack(meta_features)

# 使用示例
def train_with_ensemble(X_train, y_train, X_val, y_val):
    # 创建多个模型
    models = [
        LightGBMModel({'n_estimators': 1000, 'learning_rate': 0.05}),
        XGBoostModel({'n_estimators': 1000, 'learning_rate': 0.05}),
        CatBoostModel({'iterations': 1000, 'learning_rate': 0.05}),
    ]
    
    # 创建融合器
    ensemble = ModelEnsemble(models, method='stacking')
    
    # 训练
    ensemble.fit(X_train, y_train, X_val, y_val)
    
    # 预测
    y_pred = ensemble.predict(X_val)
    
    return ensemble, y_pred