"""单序列目标分解编排；算法拟合与构造分别由组件和工厂负责。"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from decomposition.construction.component_factory import build_components
from decomposition.configuration.spec import DecompositionSpec
from decomposition.contracts.time_axis import to_datetime_index
from decomposition.contracts.types import ComponentForecast, ComponentFrame


class DecompositionPipeline:
    def __init__(self, spec: DecompositionSpec):
        self._state_version = 2
        self.spec = spec.expand_preset()
        self.method = spec.method
        self.enabled = spec.enabled
        self.is_fitted = False
        self.time_col = "time"
        self.target_col = "y"
        self._components = build_components(self.spec)
        self.history_frame: ComponentFrame | None = None
        self.history_component: pd.Series | None = None

    def fit(self, df: pd.DataFrame, time_col: str = "time", target_col: str = "y") -> DecompositionPipeline:
        self.time_col = time_col
        self.target_col = target_col
        self.is_fitted = False
        self.history_frame = None
        self.history_component = None
        self._components = build_components(self.spec)
        if not self.enabled:
            self.is_fitted = True
            return self
        extractor = self._components.extractor
        if extractor is None:
            raise RuntimeError("Enabled decomposition requires an extractor.")
        frame = extractor.fit_transform(df, time_col, target_col)
        for forecaster in (self._components.trend, self._components.seasonal):
            if forecaster is not None:
                forecaster.fit(frame)
        self.history_frame = frame
        self.history_component = pd.Series(frame.deterministic, index=frame.times.asi8, dtype=float)
        self.is_fitted = True
        return self

    def component(self, times: Any) -> np.ndarray:
        if not self.enabled:
            return np.zeros(len(times), dtype=float)
        index = to_datetime_index(times)
        component = self._components.composer.compose(np.zeros(len(index)), self.forecast_components(index))
        # 已见历史用真实分解结果；未来用外推。不能用模板覆盖真实历史。
        history_component = self.history_component
        if history_component is None:
            raise RuntimeError("Decomposition pipeline has no fitted historical components.")
        exact = history_component.reindex(index.asi8)
        known = exact.notna().to_numpy()
        component[known] = exact.to_numpy(dtype=float)[known]
        return component

    def forecast_components(self, future_times: Any) -> ComponentForecast:
        if self.enabled and not self.is_fitted:
            raise RuntimeError("Decomposition pipeline must be fitted before forecast.")
        index = to_datetime_index(future_times)
        parts = {}
        for name, forecaster in (("trend", self._components.trend), ("seasonal", self._components.seasonal)):
            if forecaster is not None:
                parts[name] = forecaster.predict(index)
        return ComponentForecast(index, parts)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.enabled:
            out[self.target_col] = out[self.target_col].to_numpy(dtype=float) - self.component(out[self.time_col])
        return out

    def fit_transform(self, df: pd.DataFrame, time_col: str = "time", target_col: str = "y") -> pd.DataFrame:
        return self.fit(df, time_col, target_col).transform(df)

    def restore(self, values: Any, times: Any) -> np.ndarray:
        if not self.enabled:
            return np.asarray(values)
        index = to_datetime_index(times)
        components = ComponentForecast(index, {"deterministic": self.component(index)})
        return self._components.composer.compose(np.asarray(values, dtype=float), components)

    def __setstate__(self, state: dict[str, Any]) -> None:
        if state.get("_state_version") != 2:
            raise ValueError("incompatible decomposition artifact: explicitly refit with component_fit_v2; no legacy state conversion")
        self.__dict__.update(state)
