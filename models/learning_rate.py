# -*- coding: utf-8 -*-

"""
学习率策略工具：
- 固定学习率（配置给定）
- 可选自动学习率（基于样本规模的启发式缩放）
"""

from typing import Optional


def resolve_learning_rate(
    base_learning_rate: Optional[float],
    n_samples: int,
    auto_enabled: bool = False,
    min_lr: float = 0.005,
    max_lr: float = 0.2,
) -> Optional[float]:
    """
    根据样本规模返回最终学习率。

    经验规则：
    - 样本越大，学习率适当降低；
    - 样本越小，学习率适当提高；
    - 最终裁剪到 [min_lr, max_lr] 区间。
    """
    if base_learning_rate is None and not auto_enabled:
        return None

    base = float(base_learning_rate) if base_learning_rate is not None else 0.05
    if not auto_enabled:
        return max(min(base, max_lr), min_lr)

    n = max(1, int(n_samples))
    # 以 1e4 样本作为基准点，sqrt 缩放
    scaled = base * (10000.0 / n) ** 0.5
    scaled = max(min(float(scaled), max_lr), min_lr)
    return scaled
