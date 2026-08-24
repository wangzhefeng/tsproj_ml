# -*- coding: utf-8 -*-
"""分解模块的共享数据类型。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ComponentFrame.components 中承载原始目标 y 的保留键（residual = __y__ - sum(其余分量)）
TARGET_KEY = "__y__"


@dataclass
class ComponentFrame:
    """训练窗口内的历史分解结果。

    ``components`` 键为分量名（如 ``trend``、``seasonal``），
    值为与 ``times`` 对齐的分量序列；原始 y 存于 ``TARGET_KEY``，
    residual 不显式存储，由 ``TARGET_KEY - sum(components)`` 隐式定义。
    """

    times: pd.DatetimeIndex
    components: dict[str, np.ndarray]


@dataclass
class ComponentForecast:
    """未来时点的结构化分量预测。"""

    times: pd.DatetimeIndex
    components: dict[str, np.ndarray]
