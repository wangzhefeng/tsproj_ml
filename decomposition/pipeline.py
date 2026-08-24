# -*- coding: utf-8 -*-
"""DecompositionPipeline：extractor + forecaster + composer 的编排器。

对主链暴露与旧 TargetDecomposer 相同的语义接口：
fit / transform / component / restore，使主链替换为纯入口切换。
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from decomposition.base import ComponentExtractor, to_datetime_index
from decomposition.composers import AdditiveComposer
from decomposition.extractors import MSTLExtractor, NoneExtractor, PolyTrendExtractor, STLExtractor
from decomposition.forecasters import PhaseTemplateForecaster, PolyTrendForecastForecaster
from decomposition.spec import ComponentSpec, DecompositionSpec
from decomposition.types import TARGET_KEY


class DecompositionPipeline:
    """在可见训练窗口内分解目标，并将分量预测还原到原始电平。

    与旧 TargetDecomposer 逐位兼容（除 Phase 0 两处已修复缺陷）。
    可 pickle：所有组件持有 numpy 数组和基本类型。
    """

    def __init__(self, spec: DecompositionSpec):
        # preset 方法自动展开为组件组合；custom 已是展开形态（当前 fail-fast）
        self.spec = spec.expand_preset() if spec.method != "custom" else spec
        self.method = spec.method
        self.enabled = spec.enabled
        self.is_fitted = False
        self.time_col = "time"
        self.target_col = "y"

        # 组件实例（由 _build_components 构造）
        self._extractor: Optional[ComponentExtractor] = None
        self._trend_forecaster = None
        self._seasonal_forecaster = None
        self._composer = AdditiveComposer({})

        # 历史 exact 分量（旧 history_component 语义）
        self.history_component: Optional[pd.Series] = None

    def _build_components(self) -> None:
        s = self.spec
        if s.method == "none":
            self._extractor = NoneExtractor()
            return
        if s.extractor is None:  # pragma: no cover - resolve 阶段已展开
            raise ValueError(f"spec for method={s.method} has no extractor.")
        ext_type = s.extractor.type
        ext_params = dict(s.extractor.params)
        if ext_type == "poly_trend":
            self._extractor = PolyTrendExtractor(ext_params)
        elif ext_type == "stl":
            self._extractor = STLExtractor(ext_params)
        elif ext_type == "mstl":
            self._extractor = MSTLExtractor(ext_params)
        else:
            raise ValueError(f"Unknown extractor type '{ext_type}'.")

        if s.trend_forecaster is not None:
            tf_type = s.trend_forecaster.type
            tf_params = dict(s.trend_forecaster.params)
            if tf_type == "poly_trend_forecast":
                self._trend_forecaster = PolyTrendForecastForecaster(tf_params)
            else:
                raise ValueError(f"Unknown trend forecaster type '{tf_type}'.")
        if s.seasonal_forecaster is not None:
            sf_type = s.seasonal_forecaster.type
            sf_params = dict(s.seasonal_forecaster.params)
            if sf_type == "phase_template":
                self._seasonal_forecaster = PhaseTemplateForecaster(sf_params)
            else:
                raise ValueError(f"Unknown seasonal forecaster type '{sf_type}'.")

    # ------------------------------------------------------------------
    # fit / transform（历史段）
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, time_col: str = "time", target_col: str = "y") -> "DecompositionPipeline":
        self.time_col = time_col
        self.target_col = target_col
        self._build_components()
        if not self.enabled:
            self.is_fitted = True
            return self
        if time_col not in df.columns or target_col not in df.columns:
            raise ValueError(f"Target decomposition requires columns: {time_col}, {target_col}.")

        times = to_datetime_index(df[time_col])
        y = np.asarray(pd.to_numeric(df[target_col], errors="coerce"), dtype=float)

        assert self._extractor is not None
        self._extractor.fit(df, time_col=time_col, target_col=target_col)
        components = self._extractor.components()
        # 历史 exact 确定分量优先取分解器原始输出（STL/MSTL），
        # 与旧实现的 history_component 逐位一致
        exact_deterministic = getattr(self._extractor, "exact_deterministic", None)

        # exact 分量：STL/MSTL 用分解结果重构；linear 用全局多项式趋势
        if exact_deterministic is not None:
            deterministic = np.asarray(exact_deterministic, dtype=float)
        else:
            deterministic = (
                sum(v for k, v in components.items() if k != TARGET_KEY)
                if components
                else np.zeros(len(y), dtype=float)
            )
        self.history_component = pd.Series(deterministic, index=times.asi8, dtype=float)

        # 注入 trend forecaster 状态：直接桥接 extractor 属性
        if isinstance(self._trend_forecaster, PolyTrendForecastForecaster) and hasattr(self._extractor, "trend_coefficients"):
            self._trend_forecaster.attach(
                trend_coefficients=self._extractor.trend_coefficients,
                last_time_ns=self._extractor.last_time_ns,
                last_trend=self._extractor.last_trend,
                last_trend_slope_per_day=self._extractor.last_trend_slope_per_day,
            )
        if isinstance(self._seasonal_forecaster, PhaseTemplateForecaster) and hasattr(self._extractor, "seasonal_templates"):
            self._seasonal_forecaster.attach(
                templates=self._extractor.seasonal_templates,
                n_obs=self._extractor.n_obs,
                last_time_ns=self._extractor.last_time_ns,
                step_ns=self._extractor.step_ns,
            )

        # polyval origin：与 extractor 的 origin_ns 一致
        if isinstance(self._trend_forecaster, PolyTrendForecastForecaster):
            self._trend_forecaster.trend_coefficients_context_ns = getattr(self._extractor, "origin_ns", 0)

        self.is_fitted = True
        return self

    def component(self, times: Any) -> np.ndarray:
        if not self.enabled or not self.is_fitted:
            return np.zeros(len(times), dtype=float)
        index = to_datetime_index(times)
        future_fc = self.forecast_components(index)
        component = np.zeros(len(index), dtype=float)
        for series in future_fc.components.values():
            component += series
        if self.history_component is not None:
            exact = self.history_component.reindex(index.asi8)
            known = exact.notna().to_numpy()
            component[known] = exact.to_numpy(dtype=float)[known]
        return component

    def forecast_components(self, future_times: pd.DatetimeIndex):
        from decomposition.types import ComponentForecast

        parts: dict[str, np.ndarray] = {}
        if self._trend_forecaster is not None:
            t = self._trend_forecaster.predict(future_times)
            if t is not None:
                parts["trend"] = t
        if self._seasonal_forecaster is not None:
            s = self._seasonal_forecaster.predict(future_times)
            if s is not None:
                parts["seasonal"] = s
        return ComponentForecast(times=future_times, components=parts)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled or not self.is_fitted:
            return df.copy()
        out = df.copy()
        out[self.target_col] = (
            out[self.target_col].to_numpy(dtype=float) - self.component(out[self.time_col])
        )
        return out

    def fit_transform(
        self, df: pd.DataFrame, time_col: str = "time", target_col: str = "y"
    ) -> pd.DataFrame:
        return self.fit(df, time_col=time_col, target_col=target_col).transform(df)

    def restore(self, values, times: Any) -> np.ndarray:
        values_arr = np.asarray(values)
        if not self.enabled or not self.is_fitted:
            return values_arr
        return np.asarray(values, dtype=float) + self.component(times)

    @staticmethod
    def from_args(args: Any, log_prefix: str = "[DecompositionPipeline]", verbose: bool = False) -> "DecompositionPipeline":
        """从 cfg 构造 pipeline（主链入口，替代旧 TargetDecomposer(args)）。"""
        from decomposition.spec import resolve_decomposition_spec

        spec = resolve_decomposition_spec(args)
        return DecompositionPipeline(spec)
