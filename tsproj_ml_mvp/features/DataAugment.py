# -*- coding: utf-8 -*-

# ***************************************************
# * File        : DataAugment.py
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


def augment_time_series(df, target_feature, noise_level=0.01):
    """
    时间序列数据增强
    - 添加噪声增强
    """
    df_augmented = df.copy()
    noise = np.random.normal(0, noise_level, len(df))
    df_augmented[target_feature] = df[target_feature] + df[target_feature] * noise

    return df_augmented




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
