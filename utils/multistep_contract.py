#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多步预测的信息可得性与形状契约。"""

from __future__ import annotations

from typing import Any


USMDP_METHOD = "univariate-single-multistep-direct-pointwise"


def validate_direct_feature_alignment(args: Any, horizon: int) -> None:
    """校验 ``align_direct_features_to_target`` 的可用范围。

    USMDP 多步模式通过 safe-lag 直接构造每个未来目标时点的特征；所有 lag
    必须不短于 horizon，保证未来第 1..H 行只引用预测原点之前的真实历史。
    其他 Direct/DirRec 方法沿用现有单步目标对齐语义。
    """
    if not bool(getattr(args, "align_direct_features_to_target", False)):
        return

    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError(f"forecast horizon must be positive; got {horizon}.")

    pred_method = str(getattr(args, "pred_method", "")).lower()
    if pred_method != USMDP_METHOD:
        if horizon != 1:
            raise ValueError(
                "align_direct_features_to_target currently requires predict_steps=1 "
                f"for pred_method={pred_method}."
            )
        return

    if not bool(getattr(args, "enable_lags_features", False)):
        raise ValueError(
            "USMDP safe-lag requires enable_lags_features=true when "
            "align_direct_features_to_target=true."
        )

    raw_lags = list(getattr(args, "lags", []) or [])
    if not raw_lags:
        raise ValueError(
            "USMDP safe-lag requires at least one positive lag when "
            "align_direct_features_to_target=true."
        )
    lags = [int(lag) for lag in raw_lags]
    invalid = [lag for lag in lags if lag <= 0]
    if invalid:
        raise ValueError(f"USMDP safe-lag requires positive lags; got {invalid}.")

    min_lag = min(lags)
    if min_lag < horizon:
        raise ValueError(
            "USMDP safe-lag requires min(lags) >= horizon; "
            f"got min_lag={min_lag}, horizon={horizon}."
        )
