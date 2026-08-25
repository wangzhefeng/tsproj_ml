#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多步预测的信息可得性与形状契约。"""

from __future__ import annotations

from typing import Any

from models.multistep.resolve import resolve_strategy


def validate_direct_feature_alignment(args: Any, horizon: int) -> None:
    """校验 ``align_direct_features_to_target`` 的可用范围。

    USMDP 多步模式通过 safe-lag 直接构造每个未来目标时点的特征；所有 lag
    必须不短于 horizon，保证未来第 1..H 行只引用预测原点之前的真实历史。
    其他 Direct/DirRec 方法沿用现有单步目标对齐语义。
    """
    # 兼容旧调用入口；所有规则由唯一解析器维护。
    resolve_strategy(args, horizon)
