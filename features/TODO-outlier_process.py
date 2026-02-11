# -*- coding: utf-8 -*-

# ***************************************************
# * File        : outlier_process.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-11
# * Version     : 1.0.021111
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


from scipy import stats

def remove_outliers(df, column, threshold=3):
    """移除异常值（Z-score方法）"""
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
