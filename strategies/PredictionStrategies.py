# -*- coding: utf-8 -*-

# ***************************************************
# * File        : prediction_strategies.py
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


class PredictionHelper:
    """预测辅助类 - 所有预测方法的公共逻辑"""
    
    @staticmethod
    def prepare_features(X, preprocessor, categorical_features):
        """
        统一的特征预处理
        
        所有7种预测方法共用此方法，避免重复代码
        """
        return preprocessor.transform(X, categorical_features)
    
    @staticmethod
    def build_lag_features(df, features, lags):
        """
        统一的滞后特征构建
        
        适用于单变量和多变量方法
        """
        for feat in features:
            for lag in lags:
                df[f'{feat}_lag_{lag}'] = df[feat].shift(lag)
        return df
    
    @staticmethod
    def recursive_predict_step(model, current_features, history, target, step):
        """
        递归预测的单步逻辑
        
        USMR和MSMR共用此方法
        """
        # 预测
        prediction = model.predict(current_features)[0]
        
        # 更新历史
        history.loc[len(history)] = prediction
        
        return prediction





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
