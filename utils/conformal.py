# -*- coding: utf-8 -*-

# ***************************************************
# * File        : conformal.py
# * Description : Conformalized Quantile Regression (CQR) 兼容入口。
# *               权威实现位于 probabilistic.calibration。
# * Reference   : Romano, Patterson, Candès (2019), "Conformalized Quantile Regression"
# * ***************************************************

from probabilistic.calibration import (
    calibrate_quantile_band,
    compute_nonconformity_scores,
)

__all__ = ["calibrate_quantile_band", "compute_nonconformity_scores"]
