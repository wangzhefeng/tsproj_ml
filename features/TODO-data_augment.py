# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_augment.py
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

import numpy as np

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def augment_time_series(df, noise_level=0.01):
    """
    时间序列数据增强
    添加噪声增强
    """
    df_augmented = df.copy()
    noise = np.random.normal(0, noise_level, len(df))
    df_augmented['target'] = df['target'] + df['target'] * noise

    return df_augmented




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
