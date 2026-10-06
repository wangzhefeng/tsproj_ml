"""训练、预测与融合共用的策略坐标和特征注入合同。"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class TargetCoordinate:
    target: str
    horizon_step: int

    def __post_init__(self) -> None:
        if not isinstance(self.target, str):
            raise TypeError("target must be a string")
        if not self.target.strip() or self.target != self.target.strip():
            raise ValueError("target must be nonblank without surrounding whitespace")
        if isinstance(self.horizon_step, bool) or not isinstance(self.horizon_step, int):
            raise TypeError("horizon_step must be an integer")
        if self.horizon_step <= 0:
            raise ValueError("horizon_step must be positive")


FeatureProvider = Callable[
    [int, tuple[TargetCoordinate, ...], tuple[TargetCoordinate, ...],
     Mapping[TargetCoordinate, np.ndarray]],
    np.ndarray,
]
