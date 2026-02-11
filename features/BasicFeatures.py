# -*- coding: utf-8 -*-

# ***************************************************
# * File        : basic_features.py
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


class FeatureScaler:
    """统一的特征缩放器"""
    
    def __init__(self, scaler, encode_categorical: bool = False):
        self.scaler = scaler
        self.encode_categorical = encode_categorical
        self.category_encoders = {}
    
    def fit_transform(self, X: pd.DataFrame, categorical_features: List[str]):
        """训练并转换"""
        X_scaled = X.copy()
        
        # 分离数值和类别特征
        numeric_features = [col for col in X.columns if col not in categorical_features]
        
        # 缩放数值特征
        if numeric_features and self.scaler is not None:
            X_scaled[numeric_features] = self.scaler.fit_transform(X[numeric_features])
        
        # 编码类别特征
        if self.encode_categorical:
            for col in categorical_features:
                if col in X.columns:
                    X_scaled[col] = X[col].astype('category')
                    self.category_encoders[col] = {
                        'categories': X_scaled[col].cat.categories.tolist()
                    }
                    X_scaled[col] = X_scaled[col].cat.codes
        
        return X_scaled
    
    def transform(self, X: pd.DataFrame, categorical_features: List[str]):
        """仅转换"""
        X_scaled = X.copy()
        
        # 分离数值和类别特征
        numeric_features = [col for col in X.columns if col not in categorical_features]
        
        # 缩放数值特征
        if numeric_features and self.scaler is not None:
            X_scaled[numeric_features] = self.scaler.transform(X[numeric_features])
        
        # 编码类别特征
        if self.encode_categorical:
            for col in categorical_features:
                if col in X.columns and col in self.category_encoders:
                    encoder_info = self.category_encoders[col]
                    X_scaled[col] = pd.Categorical(
                        X[col],
                        categories=encoder_info['categories']
                    )
                    X_scaled[col] = X_scaled[col].cat.codes
        
        return X_scaled


# 测试代码 main 函数
def main():
    # 训练时
    scaler = FeatureScaler(StandardScaler(), encode_categorical=True)
    X_train_scaled = scaler.fit_transform(X_train, categorical_features)

    # 预测时
    X_test_scaled = scaler.transform(X_test, categorical_features)

if __name__ == "__main__":
    main()
