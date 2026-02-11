# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_factory.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-11
# * Version     : 1.0.021110
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cab
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain


"""
模型抽象层 (Model Abstraction Layer)
====================================

提供统一的模型接口，支持多种机器学习模型的无缝切换

支持的模型:
- LightGBM
- XGBoost  
- CatBoost
- Random Forest
- Extra Trees
- Gradient Boosting

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
    
    def __init__(self, params: Dict[str, Any]):
        """
        初始化模型
        
        Args:
            params (Dict[str, Any]): 模型参数字典
        """
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

# ##############################
# 具体模型实现
# ##############################
class LightGBMModel(BaseModel):
    """
    LightGBM模型封装
    
    特点:
    - 训练速度快
    - 内存占用小
    - 支持类别特征
    - 适合大数据集
    """

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        # 设置默认参数
        default_params_v1 = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': -1,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbose': -1,
            'n_jobs': -1,
            'random_state': 42,
        }
        default_params = {
            "boosting_type": "gbdt",
            "objective": "regression_l1",  # "regression_l1": L1 loss or MAE, "regression": L2 loss or MSE
            "metric": "mae",  # if objective=="regression_l1": mae, if objective=="regression": rmse
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_bin": 31,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 0.5,
            "lambda_l2": 0.5,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }
        default_params.update(params)
        self.params = default_params
        self.model = lgb.LGBMRegressor(**self.params)
    
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            categorical_features: Optional[list] = None,
            eval_set: Optional[tuple] = None,
            eval_metric: str = "mae",
            early_stopping_rounds: int = 100,
            verbose: bool = False):
        """
        训练LightGBM模型
        
        Args:
            X: 训练特征
            y: 训练目标
            eval_set: 验证集 [(X_val, y_val)]
            eval_metric: 评估指标
            categorical_features: 类别特征列表
            early_stopping_rounds: 早停轮数
            verbose: 是否显示训练过程
        """
        fit_params = {}
        
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            fit_params['eval_metric'] = eval_metric
            fit_params['callbacks'] = [lgb.early_stopping(early_stopping_rounds, verbose=verbose)]
        
        if categorical_features is not None:
            fit_params['categorical_feature'] = categorical_features
        
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练 (Model not fitted yet)")
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

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        # 设置默认参数
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': -1,
            'random_state': 42,
        }
        default_params.update(params)
        self.params = default_params
        self.model = xgb.XGBRegressor(**self.params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            eval_set: Optional[tuple] = None,
            early_stopping_rounds: int = 50,
            verbose: bool = False):
        """
        训练XGBoost模型
        
        Args:
            X: 训练特征
            y: 训练目标
            eval_set: 验证集 [(X_val, y_val)]
            early_stopping_rounds: 早停轮数
            verbose: 是否显示训练过程
        """
        fit_params = {}
        
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            fit_params['early_stopping_rounds'] = early_stopping_rounds
            fit_params['verbose'] = verbose
        
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        return self.model.predict(X)

class CatBoostModel(BaseModel):
    """
    CatBoost模型封装
    
    特点:
    - 自动处理类别特征
    - 对默认参数不敏感
    - 过拟合风险低
    - 性能优秀
    """
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        # 设置默认参数
        default_params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'verbose': False,
            'random_state': 42,
            'thread_count': -1,
        }
        default_params.update(params)
        self.params = default_params
        self.model = cab.CatBoostRegressor(**self.params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            eval_set: Optional[tuple] = None,
            categorical_features: Optional[list] = None,
            early_stopping_rounds: int = 50):
        """
        训练CatBoost模型
        
        Args:
            X: 训练特征
            y: 训练目标
            eval_set: 验证集 (X_val, y_val)
            categorical_features: 类别特征列表
            early_stopping_rounds: 早停轮数
        """
        fit_params = {}
        
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            fit_params['early_stopping_rounds'] = early_stopping_rounds
        
        if categorical_features is not None:
            fit_params['cat_features'] = categorical_features
        
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        return self.model.predict(X)

class RandomForestModel(BaseModel):
    """
    Random Forest模型封装
    
    特点:
    - 鲁棒性强
    - 不易过拟合
    - 可解释性好
    - 并行化训练
    """
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'n_jobs': -1,
            'random_state': 42,
        }
        default_params.update(params)
        self.params = default_params
        self.model = RandomForestRegressor(**self.params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """训练Random Forest模型"""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
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
        'lightgbm': LightGBMModel,
        'lgb': LightGBMModel,
        'xgboost': XGBoostModel,
        'xgb': XGBoostModel,
        'catboost': CatBoostModel,
        'cat': CatBoostModel,
        'randomforest': RandomForestModel,
        'rf': RandomForestModel,
    }
    
    @staticmethod
    def create_model(model_type: str, params: Dict[str, Any]) -> BaseModel:
        """
        创建模型实例
        
        Args:
            model_type: 模型类型 ('lightgbm', 'xgboost', 'catboost', 'randomforest')
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
        model_type = model_type.lower()
        
        if model_type not in ModelFactory._models:
            supported = ', '.join(ModelFactory._models.keys())
            raise ValueError(
                f"不支持的模型类型: {model_type}\n"
                f"支持的模型: {supported}"
            )
        
        model_class = ModelFactory._models[model_type]
        return model_class(params)
    
    @staticmethod
    def list_models() -> list:
        """列出所有支持的模型类型"""
        return list(ModelFactory._models.keys())

# ##############################
# 示例：在Model类中使用
# ##############################
class Model:

    def __init__(self, args):
        self.args = args
        # 使用工厂创建模型
        self.model_factory = ModelFactory()
    
    def train(self, X_train, Y_train, categorical_features):
        # 创建模型
        base_model = self.model_factory.create_model(
            self.args.model_type,  # 'lightgbm' / 'xgboost' / 'catboost'
            self.args.model_params
        )
        
        # 模型训练
        if Y_train.shape[1] == 1:
            model = base_model
        else:
            model = MultiOutputRegressor(base_model)
        
        model.fit(X_train, Y_train)
        return model




# 测试代码 main 函数
def main():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # 创建示例数据
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1000, 10), columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(np.random.randn(1000), name='target')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 测试不同模型
    for model_type in ['lightgbm', 'xgboost', 'catboost']:
        print(f"\n测试 {model_type} 模型:")
        print("=" * 50)
        
        # 创建模型
        params = {'n_estimators': 100, 'learning_rate': 0.1}
        model = ModelFactory.create_model(model_type, params)
        
        # 训练
        model.fit(X_train, y_train, eval_set=(X_test, y_test))
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 评估
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        # 特征重要性
        importance = model.get_feature_importance()
        if importance is not None:
            print(f"前5个重要特征:")
            top_features = np.argsort(importance)[-5:][::-1]
            for i, idx in enumerate(top_features, 1):
                print(f"  {i}. {X.columns[idx]}: {importance[idx]:.4f}")

if __name__ == "__main__":
    main()
