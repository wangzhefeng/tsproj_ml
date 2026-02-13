# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-07
# * Version     : 1.0.020720
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

import pandas as pd

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger







# 测试代码 main 函数
def main():
    import pandas as pd
    df = pd.DataFrame({
        "time": pd.date_range('2022-01-01 00:00:00', periods=10, freq='H'), 
        "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    print(df)
    df["value_lag_7"] = df["value"].shift(7)
    df["value_lag_6"] = df["value"].shift(6)
    df["value_lag_5"] = df["value"].shift(5)
    df["value_lag_4"] = df["value"].shift(4)
    df["value_lag_3"] = df["value"].shift(3)
    df["value_lag_2"] = df["value"].shift(2)
    df["value_lag_1"] = df["value"].shift(1)
    df["value_shift_0"] = df["value"].shift(0)
    df["value_shift_1"] = df["value"].shift(-1)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

if __name__ == "__main__":
    main()
