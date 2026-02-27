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
import numpy as np
import pandas as pd
from scipy import stats


def remove_outliers(df: pd.DataFrame, column: str, threshold: int=3):
    """
    移除异常值（Z-score方法）
    """
    z_scores = np.abs(stats.zscore(df[column]))

    return df[z_scores < threshold]




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
