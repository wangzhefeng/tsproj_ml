# -*- coding: utf-8 -*-

# ***************************************************
# * File        : learning_rate.py
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

# 使用学习率调度器
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    return mean_absolute_error(y_val, y_pred)


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
