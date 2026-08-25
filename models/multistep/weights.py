# -*- coding: utf-8 -*-
"""Direct/Recursive 融合权重的严格运行期契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from models.multistep.contracts import require_exact_vector_output


@dataclass(frozen=True)
class BlendWeights:
    """可审计且可随模型保存的凸组合权重。"""

    direct: float
    recursive: float
    strategy: str
    calibration_windows: int = 0

    def __post_init__(self) -> None:
        values = np.asarray([self.direct, self.recursive], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Blend weights must be finite.")
        if (values < 0.0).any():
            raise ValueError("Blend weights must be non-negative.")
        if not np.isclose(float(values.sum()), 1.0, atol=1e-9):
            raise ValueError(
                "Blend weights must sum to 1; "
                f"got direct={self.direct}, recursive={self.recursive}."
            )
        if int(self.calibration_windows) < 0:
            raise ValueError("calibration_windows must be >= 0.")
        object.__setattr__(self, "direct", float(self.direct))
        object.__setattr__(self, "recursive", float(self.recursive))
        object.__setattr__(self, "strategy", str(self.strategy).lower())
        object.__setattr__(self, "calibration_windows", int(self.calibration_windows))

    @classmethod
    def from_args(cls, args: Any) -> "BlendWeights":
        strategy = str(getattr(args, "blend_weight_strategy", "fixed") or "fixed").lower()
        if strategy == "fixed":
            direct = float(getattr(args, "blend_direct_weight", 0.5))
            return cls(direct, 1.0 - direct, strategy="fixed")
        if strategy != "ridge_stacking":
            raise ValueError(f"Unsupported blend_weight_strategy='{strategy}'.")

        resolved = getattr(args, "resolved_blend_weights", None)
        if resolved is None:
            raise ValueError(
                "ridge_stacking requires resolved_blend_weights before model training; "
                "runtime CSV lookup is not allowed."
            )
        if isinstance(resolved, cls):
            return resolved
        if not isinstance(resolved, dict):
            raise TypeError("resolved_blend_weights must be BlendWeights or dict.")
        return cls(
            direct=float(resolved["direct"]),
            recursive=float(resolved["recursive"]),
            strategy="ridge_stacking",
            calibration_windows=int(resolved.get("calibration_windows", 0)),
        )

    def combine(self, direct_output, recursive_output, horizon: int) -> np.ndarray:
        direct = require_exact_vector_output(direct_output, horizon, label="blend direct")
        recursive = require_exact_vector_output(
            recursive_output,
            horizon,
            label="blend recursive",
        )
        return self.direct * direct + self.recursive * recursive

    def metadata(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "direct": self.direct,
            "recursive": self.recursive,
            "calibration_windows": self.calibration_windows,
        }
