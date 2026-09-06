"""真实历史分量诊断计算；不访问 Pipeline 私有状态，不承担 IO。"""
import numpy as np
import pandas as pd
from decomposition.contracts.types import ComponentFrame


def summarize_components(frame: ComponentFrame) -> pd.DataFrame:
    parts = {"y": frame.target, "deterministic": frame.deterministic,
             "residual": frame.residual, "trend": frame.trend, "seasonal": frame.seasonal_total}
    parts.update({f"seasonal_{period}": values for period, values in frame.seasonal.items()})
    rows = []
    for name, values in parts.items():
        arr = np.asarray(values, dtype=float)
        if arr.shape != (len(frame.times),) or not len(arr) or not np.isfinite(arr).all():
            raise ValueError("Component diagnostics require aligned finite nonempty arrays.")
        rows.append({"component": name, "mean": float(np.mean(arr)), "std": float(np.std(arr)),
                     "min": float(np.min(arr)), "max": float(np.max(arr)), "n": len(arr)})
    return pd.DataFrame(rows)
