# -*- coding: utf-8 -*-

# ***************************************************
# * File        : runtime_env.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-11
# * Version     : 1.0.061317
# * Description : 运行期环境变量辅助
# ***************************************************


# python libraries
import os
import tempfile
from pathlib import Path

def ensure_runtime_environment():
    mpl_dir = Path(tempfile.gettempdir()).joinpath("tsproj_ml_matplotlib")
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

