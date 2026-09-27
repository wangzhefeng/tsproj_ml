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

# 运行时依赖包名清单：实现指纹（model_performance/checkpoints.py）与运行证据
# （model_predicting/artifacts/evidence_collect.py）共用的唯一来源；两侧的缺失哨兵串各自维护。
RUNTIME_DEPENDENCY_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "catboost",
    "statsmodels",
    "chinese-calendar",
)

def ensure_runtime_environment():
    mpl_dir = Path(tempfile.gettempdir()).joinpath("tsproj_ml_matplotlib")
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

