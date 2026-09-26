"""XGBoost 预检父进程侧辅助：特征名提取（私有 API 收口）。

私有 API：``xgboost.data.pandas_feature_info``——升级 xgboost 时复核
（见 docs/packages/models.md 私有 API 依赖清单）。
本模块仅父进程导入；不进入 worker 子进程（worker 模块文件名为
xgboost.py，子进程直跑时不得重依赖 pandas/xgboost.data，避免
sys.path[0] 下的同名脚本遮蔽第三方包）。
"""

from typing import Any

import numpy as np
import pandas as pd

from xgboost.data import pandas_feature_info

from models.preflight.xgb_preflight import validate_xgb_parameters


def validate_estimator(
    estimator,
    X: pd.DataFrame,
    y,
    *,
    num_targets: int = 1,
) -> dict[str, Any]:
    """XGBoostModel wrapper fit 边界的高层预检入口。

    从估计器提取 native 参数与特征名，不推进估计器 RNG、不修改估计器状态。

    Args:
        estimator: 已构造的 ``xgb.XGBRegressor`` 实例
        X: 训练特征（DataFrame 时提取特征名做维度预检）
        y: 训练目标（仅取维度）
        num_targets: 目标维度（缺省 1；y 二维时自动取列数）

    Returns:
        ``validate_xgb_parameters`` 的完整响应字典
    """
    assert estimator is not None
    feature_names = None
    if isinstance(X, pd.DataFrame):
        names, _ = pandas_feature_info(
            X, meta=None, feature_names=None, feature_types=estimator.feature_types,
            enable_categorical=estimator.enable_categorical,
        )
        feature_names = tuple(names) if names is not None else None
    targets = np.asarray(y)
    # get_xgb_params 会从 RNG 对象抽取 seed；预检不得额外推进真实模型的 RNG。
    import copy

    validation_model = copy.copy(estimator)
    validation_model.random_state = copy.deepcopy(estimator.random_state)
    return validate_xgb_parameters(
        validation_model.get_xgb_params(), num_features=X.shape[1],
        num_targets=targets.shape[1] if targets.ndim > 1 else num_targets,
        feature_names=feature_names,
    )
