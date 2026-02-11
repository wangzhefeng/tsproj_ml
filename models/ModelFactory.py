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
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# 1. 定义模型基类
class BaseModel(ABC):
    """模型基类"""
    
    @abstractmethod
    def fit(self, X, y, **kwargs):
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X, **kwargs):
        """预测"""
        pass
    
    @abstractmethod
    def get_params(self):
        """获取参数"""
        pass

# 2. 具体模型实现
class LightGBMModel(BaseModel):
    """LightGBM模型"""
    def __init__(self, params):
        self.params = params
        self.model = lgb.LGBMRegressor(**params)
    
    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self
    
    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

class XGBoostModel(BaseModel):
    """XGBoost模型"""
    def __init__(self, params):
        self.params = params
        self.model = xgb.XGBRegressor(**params)
    
    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self
    
    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

class CatBoostModel(BaseModel):
    """CatBoost模型"""
    def __init__(self, params):
        self.params = params
        self.model = cab.CatBoostRegressor(**params)
    
    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self
    
    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

# 3. 模型工厂
class ModelFactory:
    """模型工厂"""
    
    @staticmethod
    def create_model(model_type: str, params: dict) -> BaseModel:
        models = {
            'lightgbm': LightGBMModel,
            'xgboost': XGBoostModel,
            'catboost': CatBoostModel,
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return models[model_type](params)

# 4. 在Model类中使用
class Model:
    def __init__(self, args):
        # ...
        # 使用工厂创建模型
        self.model_factory = ModelFactory()
    
    def train(self, X_train, Y_train, categorical_features):
        # 创建模型
        base_model = self.model_factory.create_model(
            self.args.model_type,  # 'lightgbm' / 'xgboost' / 'catboost'
            self.model_params
        )
        
        # 训练
        if self.args.pred_method in [...]:
            model = base_model
        else:
            model = MultiOutputRegressor(base_model)
        
        model.fit(X_train, Y_train)
        return model




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
