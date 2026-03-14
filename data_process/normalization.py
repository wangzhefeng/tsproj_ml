# -*- coding: utf-8 -*-

# ***************************************************
# * File        : normalization.py
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

from sklearn.preprocessing import MinMaxScaler

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def SeriesDataNormalize(series):
    """
    数据序列归一化函数, 受异常值影响

    Parameters: 
        series: np.array (n, m)
    
    Returns:
        scaler: 归一化对象
        normalized: 归一化序列
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    """
    logger.info(f"series: \n{series}")
    # 定义标准化模型
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaler.fit(series)
    # 标准化数据
    normalized = scaler.transform(series)
    # 逆标准化数据
    inversed = scaler.inverse_transform(normalized)

    return scaler, normalized, inversed




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
    # 时间序列归一化
    # ------------------------------
    scaled, normalized, inversed = SeriesDataNormalize(series[["OT"]])
    logger.info(f"normalized: \n{normalized}")
    logger.info(f"inversed: \n{inversed}")

if __name__ == "__main__":
    main()
