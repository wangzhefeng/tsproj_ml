# -*- coding: utf-8 -*-
"""分位数预测后处理工具。"""

import numpy as np


def monotonize_quantile_columns(df, enabled: bool = False):
    """逐行排序 ``predict_q*`` 列,保证 q_low <= q_mid <= q_high。

    独立训练的分位数模型会出现 quantile crossing(q90 < q10)。开启本开关时,
    对每一行的分位数值升序后重新贴回各列,消除交叉、保证预测区间有效;
    代价是分位数标签的严格统计含义松动(标签变为"该行排序后的第 k 小值")。

    Args:
        df: 含 ``predict_qXX`` 列的 DataFrame(如 prediction/cv_plot 结果)。
        enabled: 仅当 True 时排序;False 时原样返回(默认关)。

    Returns:
        与 df 同结构的新 DataFrame(predict_q* 列已被逐行升序重排)。
    """
    if not enabled:
        return df
    qcols = sorted(c for c in df.columns if str(c).startswith("predict_q"))
    if len(qcols) < 2:
        return df
    out = df.copy()
    out[qcols] = np.sort(out[qcols].astype(float).values, axis=1)
    return out
