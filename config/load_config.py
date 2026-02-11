# -*- coding: utf-8 -*-

# ***************************************************
# * File        : load_config.py
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


import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_type = config['model']['type']
model_params = config['model']['params']




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
