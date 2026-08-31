"""Shared canonical strategy target planning and prediction execution."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from forecasting_core.specs.strategy import ForecastStrategySpec, StrategyName
from forecasting_core.tensors import PointForecastTensor


FeatureProvider = Callable[
    [
        int,
        tuple["TargetCoordinate", ...],
        tuple["TargetCoordinate", ...],
        Mapping["TargetCoordinate", np.ndarray],
    ],
    np.ndarray,
]


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


@dataclass(frozen=True, slots=True)
class StrategyTargetPlan:
    strategy_name: StrategyName
    targets: tuple[str, ...]
    horizon: int
    coordinates: tuple[TargetCoordinate, ...]
    call_coordinates: tuple[tuple[TargetCoordinate, ...], ...]
    dependencies: tuple[tuple[TargetCoordinate, ...], ...]
    model_indices: tuple[int, ...]
    model_count: int

    @classmethod
    def from_spec(
        cls,
        spec: ForecastStrategySpec,
        targets: tuple[str, ...],
        horizon: int,
    ) -> "StrategyTargetPlan":
        if not isinstance(spec, ForecastStrategySpec):
            raise TypeError("spec must be ForecastStrategySpec")
        if not isinstance(targets, tuple):
            raise TypeError("targets must be a tuple")
        if not targets:
            raise ValueError("targets must be nonempty")
        if len(set(targets)) != len(targets):
            raise ValueError("targets must be unique")

        resolved = spec.resolve(horizon)
        coordinates = tuple(
            TargetCoordinate(target, horizon_step)
            for horizon_step in range(1, horizon + 1)
            for target in targets
        )
        coordinates_per_call = resolved.steps_per_call * len(targets)
        call_coordinates = tuple(
            coordinates[start : start + coordinates_per_call]
            for start in range(0, len(coordinates), coordinates_per_call)
        )
        dependencies = tuple(
            coordinates[: call_index * coordinates_per_call]
            if resolved.consumes_previous
            else ()
            for call_index in range(resolved.n_calls)
        )
        model_indices = (
            tuple(0 for _ in range(resolved.n_calls))
            if resolved.model_count == 1
            else tuple(range(resolved.n_calls))
        )
        return cls(
            strategy_name=spec.name,
            targets=targets,
            horizon=horizon,
            coordinates=coordinates,
            call_coordinates=call_coordinates,
            dependencies=dependencies,
            model_indices=model_indices,
            model_count=resolved.model_count,
        )


class AdapterPredictor:
    """Expose a fitted multi-target adapter through the executor 2D contract."""

    __slots__ = ("adapter",)

    def __init__(self, adapter: object) -> None:
        if not callable(getattr(adapter, "predict", None)):
            raise TypeError("adapter must provide predict(X)")
        self.adapter = adapter

    def predict(self, X: np.ndarray) -> np.ndarray:
        prediction = np.asarray(getattr(self.adapter, "predict")(X), dtype=float)
        if prediction.ndim != 3:
            raise ValueError(
                "multi-target adapter prediction must have shape (N, H, K); "
                f"got {prediction.shape}"
            )
        return prediction.reshape(prediction.shape[0], -1)


@dataclass(frozen=True, slots=True)
class StrategyModelGroupArtifact:
    model_index: int
    coordinate_groups: tuple[tuple[TargetCoordinate, ...], ...]
    predictor: AdapterPredictor


@dataclass(frozen=True, slots=True)
class CanonicalStrategyArtifact:
    target_plan: StrategyTargetPlan
    model_groups: tuple[StrategyModelGroupArtifact, ...]
    feature_schema: tuple[str, ...]
    estimator_coupling: str
    estimator_capabilities: tuple[tuple[str, bool], ...]
    N: int
    H: int
    K: int
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("CanonicalStrategyArtifact schema_version must be 2")
        if self.N <= 0 or self.H <= 0 or self.K <= 0:
            raise ValueError("CanonicalStrategyArtifact N/H/K must be positive")
        if self.H != self.target_plan.horizon or self.K != len(self.target_plan.targets):
            raise ValueError("CanonicalStrategyArtifact dimensions do not match target_plan")
        if len(self.model_groups) != self.target_plan.model_count:
            raise ValueError("CanonicalStrategyArtifact model_groups do not match model_count")
        expected_indices = tuple(range(self.target_plan.model_count))
        actual_indices = tuple(group.model_index for group in self.model_groups)
        if actual_indices != expected_indices:
            raise ValueError("CanonicalStrategyArtifact model groups must be ordered by model_index")

    def __reduce__(self):
        return (
            type(self),
            (
                self.target_plan,
                self.model_groups,
                self.feature_schema,
                self.estimator_coupling,
                dict(self.estimator_capabilities),
                self.N,
                self.H,
                self.K,
                self.schema_version,
            ),
        )

    @property
    def model_count(self) -> int:
        return len(self.model_groups)

    @property
    def predictors(self) -> tuple[AdapterPredictor, ...]:
        return tuple(group.predictor for group in self.model_groups)


class StandardStrategyExecutor:
    strategy_name: ClassVar[StrategyName]

    def __init__(
        self,
        spec: ForecastStrategySpec,
        target_plan: StrategyTargetPlan,
        estimators: tuple[object, ...],
    ) -> None:
        if not isinstance(spec, ForecastStrategySpec):
            raise TypeError("spec must be ForecastStrategySpec")
        if spec.name is not self.strategy_name:
            raise ValueError(
                f"{type(self).__name__} requires {self.strategy_name.value} strategy"
            )
        if not isinstance(target_plan, StrategyTargetPlan):
            raise TypeError("target_plan must be StrategyTargetPlan")
        if target_plan.strategy_name is not spec.name:
            raise ValueError("target_plan strategy does not match spec")
        if not isinstance(estimators, tuple):
            raise TypeError("estimators must be a tuple")
        if len(estimators) != target_plan.model_count:
            raise ValueError(
                f"expected {target_plan.model_count} estimators, got {len(estimators)}"
            )
        for estimator in estimators:
            if not callable(getattr(estimator, "predict", None)):
                raise TypeError("each estimator must provide predict(X)")

        self.spec = spec
        self.target_plan = target_plan
        self.estimators = estimators

    def predict(
        self,
        X: np.ndarray,
        *,
        series_ids: tuple[Any, ...],
        forecast_times: pd.DatetimeIndex,
        feature_provider: FeatureProvider | None = None,
    ) -> PointForecastTensor:
        design_base = np.asarray(X, dtype=float)
        if design_base.ndim != 2:
            raise ValueError("X must be a two-dimensional array")

        predicted: dict[TargetCoordinate, np.ndarray] = {}
        for call_index, (call_coordinates, dependencies, model_index) in enumerate(zip(
            self.target_plan.call_coordinates,
            self.target_plan.dependencies,
            self.target_plan.model_indices,
        )):
            if feature_provider is not None:
                design = np.asarray(
                    feature_provider(
                        call_index,
                        call_coordinates,
                        dependencies,
                        predicted,
                    ),
                    dtype=float,
                )
                if design.ndim != 2 or design.shape[0] != design_base.shape[0]:
                    raise ValueError(
                        "feature_provider must return a two-dimensional design with "
                        "the same row count as X"
                    )
            elif dependencies:
                design = np.column_stack(
                    (design_base, *(predicted[coordinate] for coordinate in dependencies))
                )
            else:
                design = design_base
            raw_prediction = np.asarray(
                self.estimators[model_index].predict(design),
                dtype=float,
            )
            prediction = self._validated_prediction(
                raw_prediction,
                n_rows=design_base.shape[0],
                output_width=len(call_coordinates),
            )
            for coordinate_index, coordinate in enumerate(call_coordinates):
                predicted[coordinate] = prediction[:, coordinate_index]

        flat_values = np.column_stack(
            tuple(predicted[coordinate] for coordinate in self.target_plan.coordinates)
        )
        values = flat_values.reshape(
            design_base.shape[0],
            self.target_plan.horizon,
            len(self.target_plan.targets),
        )
        return PointForecastTensor(
            values=values,
            series_ids=series_ids,
            forecast_times=forecast_times,
            targets=self.target_plan.targets,
        )

    @staticmethod
    def _validated_prediction(
        prediction: np.ndarray,
        *,
        n_rows: int,
        output_width: int,
    ) -> np.ndarray:
        if output_width == 1 and prediction.shape == (n_rows,):
            return prediction.reshape(n_rows, 1)
        expected_shape = (n_rows, output_width)
        if prediction.shape != expected_shape:
            raise ValueError(
                f"prediction shape {prediction.shape} does not match {expected_shape}"
            )
        return prediction
