"""Canonical forecaster（2026-08-29 架构收敛自旧 models/ModelForecasting.py 迁入，实现逐字保真）。"""

from typing import Any, Callable

import numpy as np
import pandas as pd

from forecasting_core.specs import ForecastConfigSpec
from forecasting_core.probabilistic_spec import resolve_crossing_settings
from model_training.strategies import (
    CanonicalStrategyArtifact,
    TargetCoordinate,
    get_standard_executor,
    target_plan_for_config,
)
from forecasting_core.tensors import (
    MarginalQuantileForecastTensor,
    PointForecastTensor,
)
from model_training.quantile import CanonicalMarginalQuantileArtifact
from forecasting_core.artifacts import MarginalForecastDistribution


class CanonicalForecaster:
    """执行已拟合的 canonical 策略 artifact，保持 ``(N,H,K)`` 形状合同。"""

    def __init__(
        self,
        config: ForecastConfigSpec,
        artifact: CanonicalStrategyArtifact,
    ) -> None:
        if not isinstance(config, ForecastConfigSpec):
            raise TypeError("config must be a ForecastConfigSpec")
        if not isinstance(artifact, CanonicalStrategyArtifact):
            raise TypeError("artifact must be a CanonicalStrategyArtifact")
        strategy = config.strategy
        if strategy is None:
            raise ValueError("CanonicalForecaster requires a strategy config")
        if artifact.target_plan.strategy_name is not strategy.name:
            raise ValueError("artifact strategy does not match canonical config")
        if artifact.H != config.problem.horizon:
            raise ValueError("artifact horizon does not match canonical config")
        if artifact.target_plan.targets != config.problem.targets:
            raise ValueError("artifact targets do not match canonical config")
        if artifact.target_plan != target_plan_for_config(config):
            raise ValueError("artifact target plan does not match canonical config")
        self.config = config
        self.strategy = strategy
        self.artifact = artifact

    def predict(
        self,
        X: np.ndarray,
        *,
        series_ids: tuple[Any, ...],
        forecast_times: pd.DatetimeIndex,
        feature_provider=None,
    ) -> PointForecastTensor:
        design = np.asarray(X, dtype=float)
        if design.ndim != 2:
            raise ValueError("canonical forecast X must be two-dimensional")
        if design.shape[1] != len(self.artifact.feature_schema):
            raise ValueError(
                "canonical forecast feature width does not match artifact schema"
            )
        if design.shape[0] != self.artifact.N:
            raise ValueError(
                f"canonical forecast requires N={self.artifact.N} rows; "
                f"got {design.shape[0]}"
            )
        if len(series_ids) != self.artifact.N:
            raise ValueError(
                f"canonical forecast requires N={self.artifact.N} series IDs"
            )
        times = pd.DatetimeIndex(forecast_times)
        if len(times) != self.artifact.H:
            raise ValueError(
                f"canonical forecast requires H={self.artifact.H} times"
            )
        executor_type = get_standard_executor(self.strategy)
        executor = executor_type(
            self.strategy,
            self.artifact.target_plan,
            self.artifact.predictors,
        )
        return executor.predict(
            design,
            series_ids=series_ids,
            forecast_times=times,
            feature_provider=feature_provider,
        )


def repair_marginal_quantile_crossing(
    tensor: MarginalQuantileForecastTensor,
    *,
    method: str = "median_preserving_isotonic",
) -> MarginalQuantileForecastTensor:
    """按配置方法修复分位数交叉；``none`` 时原样返回。

    - ``median_preserving_isotonic``：排序 + 钳制到 point level 锚点（历史默认行为）；
    - ``rearrangement``：沿分位轴整体排序，point level 不锚定（可能平移）；
    - ``none``：不修复（保留模型原始输出，允许下界 > 上界）。
    """
    if not isinstance(tensor, MarginalQuantileForecastTensor):
        raise TypeError("tensor must be a MarginalQuantileForecastTensor")
    method = str(method).lower()
    if method == "none":
        return tensor
    if method == "rearrangement":
        return MarginalQuantileForecastTensor(
            values=np.sort(tensor.values, axis=-1),
            levels=tensor.levels,
            point_level=tensor.point_level,
            series_ids=tensor.series_ids,
            forecast_times=tensor.forecast_times,
            targets=tensor.targets,
        )
    if method != "median_preserving_isotonic":
        raise ValueError(f"unsupported crossing method: {method!r}")
    values = tensor.values.copy()
    point_index = tensor.levels.index(tensor.point_level)
    anchor = values[..., point_index].copy()
    if point_index:
        lower = np.sort(values[..., :point_index], axis=-1)
        lower = np.minimum(lower, anchor[..., None])
        values[..., :point_index] = lower
    if point_index + 1 < tensor.n_levels:
        upper = np.sort(values[..., point_index + 1 :], axis=-1)
        upper = np.maximum(upper, anchor[..., None])
        values[..., point_index + 1 :] = upper
    values[..., point_index] = anchor
    return MarginalQuantileForecastTensor(
        values=values,
        levels=tensor.levels,
        point_level=tensor.point_level,
        series_ids=tensor.series_ids,
        forecast_times=tensor.forecast_times,
        targets=tensor.targets,
    )


def assemble_marginal_quantile_distribution(
    artifact: CanonicalMarginalQuantileArtifact,
    *,
    crossing_method: str,
    feature_provider: Callable[..., np.ndarray] | None,
    predict_level: Callable[
        [CanonicalStrategyArtifact, Callable[..., np.ndarray] | None],
        PointForecastTensor,
    ],
) -> MarginalForecastDistribution:
    """逐分位点预测张量组装为 ``(N,H,K,Q)`` 边际分布（median path 递归 + 交叉修复）。

    训练期（``CanonicalMarginalQuantileForecaster``）与部署期
    （``predict_strategy_bundle``）共用本组装段；两侧仅「单 level 如何预测」
    （predict_level 回调）与 crossing 配置来源（config vs bundle spec）不同。
    """
    point_artifact = artifact.artifacts_by_level[artifact.point_level]
    has_recursive_dependencies = any(point_artifact.target_plan.dependencies)
    if has_recursive_dependencies and feature_provider is None:
        raise ValueError(
            "recursive quantile median_path requires a fixed-schema feature_provider"
        )
    point_tensor = predict_level(point_artifact, feature_provider)
    median_predictions = {
        TargetCoordinate(target, step): point_tensor.values[
            :, step - 1, target_index
        ]
        for step in range(1, point_tensor.n_steps + 1)
        for target_index, target in enumerate(point_tensor.targets)
    }

    def median_path_provider(call_index, coordinates, dependencies, _predicted):
        assert feature_provider is not None
        return feature_provider(
            call_index,
            coordinates,
            dependencies,
            median_predictions,
        )

    tensors_by_level = {artifact.point_level: point_tensor}
    for level in artifact.levels:
        if level == artifact.point_level:
            continue
        tensors_by_level[level] = predict_level(
            artifact.artifacts_by_level[level],
            (
                median_path_provider
                if has_recursive_dependencies
                else feature_provider
            ),
        )
    point_tensors = [tensors_by_level[level] for level in artifact.levels]
    quantiles = repair_marginal_quantile_crossing(
        MarginalQuantileForecastTensor(
            values=np.stack(
                [tensor.values for tensor in point_tensors],
                axis=-1,
            ),
            levels=artifact.levels,
            point_level=artifact.point_level,
            series_ids=point_tensors[0].series_ids,
            forecast_times=point_tensors[0].forecast_times,
            targets=point_tensors[0].targets,
        ),
        method=crossing_method,
    )
    return MarginalForecastDistribution(
        point=quantiles.point(),
        quantiles=quantiles,
        dependence_model=None,
        metadata={
            "recursive_propagation": "median_path",
            "crossing_method": crossing_method,
        },
    )


class CanonicalMarginalQuantileForecaster:
    """将每个边际分位 artifact 预测为 ``(N,H,K,Q)`` 张量。"""

    def __init__(
        self,
        config: ForecastConfigSpec,
        artifact: CanonicalMarginalQuantileArtifact,
    ) -> None:
        if not isinstance(config, ForecastConfigSpec):
            raise TypeError("config must be a ForecastConfigSpec")
        if not isinstance(artifact, CanonicalMarginalQuantileArtifact):
            raise TypeError(
                "artifact must be a CanonicalMarginalQuantileArtifact"
            )
        self.config = config
        self.artifact = artifact

    def predict(
        self,
        X: np.ndarray,
        *,
        series_ids: tuple[Any, ...],
        forecast_times: pd.DatetimeIndex,
        feature_provider=None,
    ) -> MarginalForecastDistribution:
        def predict_level(level_artifact, provider):
            return CanonicalForecaster(self.config, level_artifact).predict(
                X,
                series_ids=series_ids,
                forecast_times=forecast_times,
                feature_provider=provider,
            )

        # 交叉修复消费 probabilistic.crossing.method（2026-09-01 裂缝修复：
        # 此前无条件修复，配置被静默忽略）。
        crossing_method, _report_raw = resolve_crossing_settings(
            self.config.probabilistic
        )
        return assemble_marginal_quantile_distribution(
            self.artifact,
            crossing_method=crossing_method,
            feature_provider=feature_provider,
            predict_level=predict_level,
        )


__all__ = [
    "CanonicalForecaster",
    "CanonicalMarginalQuantileForecaster",
    "assemble_marginal_quantile_distribution",
    "repair_marginal_quantile_crossing",
]
