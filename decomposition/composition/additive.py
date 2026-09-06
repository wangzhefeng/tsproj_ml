"""唯一加性合成实现；拒绝数组广播、截断和非有限分量。"""
import numpy as np
from decomposition.contracts.types import ComponentForecast


class AdditiveComposer:
    def compose(self, residual_pred: np.ndarray, component_forecast: ComponentForecast) -> np.ndarray:
        residual = np.asarray(residual_pred, dtype=float)
        n = len(component_forecast.times)
        if residual.shape != (n,) or not np.isfinite(residual).all():
            raise ValueError("Additive composition requires finite residuals aligned to forecast times.")
        deterministic = np.zeros(n, dtype=float)
        for values in component_forecast.components.values():
            arr = np.asarray(values, dtype=float)
            if arr.shape != (n,) or not np.isfinite(arr).all():
                raise ValueError("Additive components must be finite and aligned to forecast times.")
            deterministic += arr
        return residual + deterministic

    def restore_quantiles(self, quantile_map: dict[float, np.ndarray], component_forecast: ComponentForecast) -> dict[float, np.ndarray]:
        return {level: self.compose(values, component_forecast) for level, values in quantile_map.items()}
