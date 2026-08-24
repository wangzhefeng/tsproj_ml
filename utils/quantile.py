# -*- coding: utf-8 -*-
"""分位数预测后处理兼容入口。"""

from probabilistic.postprocessing import repair_quantile_crossing


def monotonize_quantile_columns(df, enabled: bool = False):
    """以 q50 为锚点修复 ``predict_q*`` crossing。

    保留旧函数名供现有调用方使用；权威实现位于
    ``probabilistic.postprocessing.repair_quantile_crossing``。

    Args:
        df: 含 ``predict_qXX`` 列的 DataFrame(如 prediction/cv_plot 结果)。
        enabled: 仅当 True 时排序;False 时原样返回(默认关)。

    Returns:
        与 df 同结构的新 DataFrame；q50 不变且 predict_value 与 q50 同步。
    """
    return repair_quantile_crossing(df, enabled=enabled)
