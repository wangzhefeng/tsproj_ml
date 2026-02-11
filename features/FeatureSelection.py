# -*- coding: utf-8 -*-

# ***************************************************
# * File        : feature_selection.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-11
# * Version     : 1.0.021111
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
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

# ------------------------------
# 特征选择
# ------------------------------
def feature_selection(X_train, y_train):
    # 选择最重要的K个特征
    selector = SelectKBest(score_func=f_regression, k=50)
    X_selected = selector.fit_transform(X_train, y_train)

    return X_selected

# ------------------------------
# 特征重要性分析
# ------------------------------
def feature_importance_analysis(model, X_train, y_train):
    # 训练后查看特征重要性
    feature_importance = model.feature_importances_

    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    # 只保留重要特征
    top_features = importance_df.head(50)['feature'].tolist()
    X_train_selected = X_train[top_features]
    
    return X_train_selected




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
