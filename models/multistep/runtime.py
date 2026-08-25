# -*- coding: utf-8 -*-
"""执行器与 Forecaster 上下文之间的运行期辅助。

执行器通过 duck-typing 消费 Forecaster；本模块提供：
- context_horizon: horizon 属性缺失时回退 df_future 长度（裸构造/测试上下文）
- ensure_resolved_strategy: 惰性解析并缓存 ResolvedStrategy，避免强制走 __init__
"""

from models.multistep.resolve import resolve_strategy
from utils.log_util import logger


def format_execution_summary(family: str, model_calls: int, **details) -> str:
    """生成字段顺序稳定的多步执行审计摘要。"""
    fields = [
        "multistep execution",
        f"family={str(family)}",
        f"model_calls={int(model_calls)}",
    ]
    fields.extend(
        f"{name}={details[name]}"
        for name in sorted(details)
        if details[name] is not None
    )
    return " ".join(fields)


def log_execution_summary(context, family: str, model_calls: int, **details) -> None:
    prefix = str(getattr(context, "log_prefix", "[Forecaster]"))
    logger.info(
        f"{prefix} {format_execution_summary(family, model_calls, **details)}"
    )


def context_horizon(context) -> int:
    """解析预测跨度：优先 horizon 属性，缺失时以未来帧长度为准。"""
    horizon = getattr(context, "horizon", None)
    if horizon:
        return int(horizon)
    future = getattr(context, "df_future", None)
    if future is None:
        raise ValueError("forecast context is missing both horizon and df_future.")
    return int(len(future))


def ensure_resolved_strategy(context):
    """返回上下文的 ResolvedStrategy；未解析时从 args 现场解析并缓存。"""
    strategy = getattr(context, "resolved_strategy", None)
    if strategy is not None:
        return strategy
    target_feature = getattr(context, "target_feature", "y")
    strategy = resolve_strategy(
        context.args, context_horizon(context), target_feature=target_feature
    )
    try:
        context.resolved_strategy = strategy
    except Exception:  # pragma: no cover - 只读上下文（SimpleNamespace 等）不缓存
        pass
    return strategy
