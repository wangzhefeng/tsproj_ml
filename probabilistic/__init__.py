# -*- coding: utf-8 -*-
"""时间序列概率预测的分位数训练、校准与后处理。

主链现役：``objectives``（quantile objective 注入）、``training``
（逐 level canonical artifact 编排）、``calibration``（CQR 内核，经
runtime 的 as-of 校准追踪器接线）。休眠预留：``pipeline`` /
``postprocessing``（DataFrame 版组合入口，仅测试引用，2026-09-01 标注）。
"""

from forecasting_core.probabilistic_spec import validate_quantile_grid

__all__ = ["validate_quantile_grid"]
