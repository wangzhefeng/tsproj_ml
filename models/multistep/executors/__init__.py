# -*- coding: utf-8 -*-
"""五类多步推进执行器 catalog。"""

from models.multistep.executors.blend import BlendExecutor
from models.multistep.executors.direct import DirectExecutor
from models.multistep.executors.dirrec import DirRecExecutor
from models.multistep.executors.pointwise import PointwiseExecutor
from models.multistep.executors.recursive import RecursiveExecutor
from models.multistep.spec import RolloutFamily


EXECUTOR_CATALOG = {
    RolloutFamily.POINTWISE: PointwiseExecutor,
    RolloutFamily.DIRECT: DirectExecutor,
    RolloutFamily.RECURSIVE: RecursiveExecutor,
    RolloutFamily.DIRREC: DirRecExecutor,
    RolloutFamily.BLEND: BlendExecutor,
}


def get_executor(resolved_strategy):
    return EXECUTOR_CATALOG[resolved_strategy.spec.rollout]()


__all__ = ["EXECUTOR_CATALOG", "get_executor"]
