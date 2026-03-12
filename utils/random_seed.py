# -*- coding: utf-8 -*-

# ***************************************************
# * File        : random_seed.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-28
# * Version     : 0.1.022823
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
from pathlib import Path
import random

import numpy as np

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def set_seed_ml(seed: int = 2025):
    """
    设置可重复随机数
    """
    random.seed(seed)
    np.random.seed(seed)



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
