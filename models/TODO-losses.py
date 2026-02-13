# -*- coding: utf-8 -*-

# ***************************************************
# * File        : losses.py
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
from sklearn.metrics import make_scorer


def huber_loss(y_true, y_pred, delta=1.0):
    """
    Huber Loss (对异常值更鲁棒)

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
        delta (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss).mean()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
