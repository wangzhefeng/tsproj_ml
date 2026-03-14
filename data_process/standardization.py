# -*- coding: utf-8 -*-

# ***************************************************
# * File        : standard.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-20
# * Version     : 0.1.102023
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
from math import sqrt

from sklearn.preprocessing import StandardScaler
import torch

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def SeriesDataStandardScaler(series):
    """
    数据序列标准化函数, 不受异常值影响
    (https://scikit-learn.org/stable/modules/generated/
    sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)

    Parameters: 
        series: np.array (n, m)
    
    Returns:
        scaler: 标准化对象
        normalized: 标准化序列
    """
    logger.info(f"series: \n{series}")
    # 定义标准化模型
    scaler = StandardScaler()
    scaler.fit(series)
    # logger.info(f"Mean: {scaler.mean_}, StandardDeviation: {sqrt(scaler.var_)}")
    # 标准化数据
    normalized = scaler.transform(series)
    # 逆标准化数据
    inversed = scaler.inverse_transform(normalized)

    return scaler, normalized, inversed


class StandardScaler_TOOD:
    """
    标准化
    """
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class StandardScalerTorch:
    """
    标准化
    """
    
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean




# 测试代码 main 函数
def main():    
    # ------------------------------
    # 判断数据是否适用标准化
    # ------------------------------ 
    # 数据读取
    import pandas as pd
    series = pd.read_csv("./dataset/ETT-small/ETTh1.csv", index_col = 0)
    logger.info(f"series: \n{series.head()}")
    
    # 根据数据分布图判断数据是否服从正太分布
    # import matplotlib.pyplot as plt
    # series["OT"].hist()
    # plt.show()
    
    # ------------------------------
    # 时间序列标准化
    # ------------------------------
    scaled, normalized, inversed = SeriesDataStandardScaler(series[["OT"]])
    logger.info(f"normalized: \n{normalized}")
    logger.info(f"inversed: \n{inversed}")
    
if __name__ == "__main__":
    main()
