"""训练样本权重子包：算法本体（temporal）+ 配置接线（resolve）。

分层：model_training 只依赖 forecasting_core 与 models；resolve 读
catalog 能力位做前置校验，算法本体零项目内依赖。消费入口为编排层
拟合函数（fold_fit），预测端不消费权重。
"""
from model_training.weights.temporal import temporal_sample_weight
from model_training.weights.resolve import (
    resolve_training_sample_weight,
    training_sample_weight_spec,
)

__all__ = [
    "temporal_sample_weight",
    "resolve_training_sample_weight",
    "training_sample_weight_spec",
]
