"""面向部署的协议合同（消费方注入的 callable）。

部署/融合路径消费方注入的协议类型唯一来源；无实现、无状态。
"""

from collections.abc import Callable

import numpy as np

# 特征注入协议：调用方按 (call_index, coordinates, dependencies, predicted)
# 返回特征矩阵；训练期由 pipeline/fold_fit 注入，部署期由部署调用方/ensemble 注入。
FeatureProvider = Callable[..., np.ndarray]

__all__ = ["FeatureProvider"]
