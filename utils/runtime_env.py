# -*- coding: utf-8 -*-

# ***************************************************
# * File        : runtime_env.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-06-13
# * Version     : 1.0.061317
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import tempfile
from pathlib import Path


def ensure_runtime_environment():
    mpl_dir = Path(tempfile.gettempdir()).joinpath("tsproj_ml_matplotlib")
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()

