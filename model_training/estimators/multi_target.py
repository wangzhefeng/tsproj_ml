"""Canonical adapters between estimators and ``(N, H, K)`` targets."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from forecasting_core.checkpoints import FitCheckpoint
from forecasting_core.tensors import unflatten_time_major
from model_training.estimators.capabilities import EstimatorCapabilities
from model_training.strategies.base import TargetCoordinate


@dataclass(frozen=True, slots=True)
class AdapterMetadata:
    target_coordinates: tuple[TargetCoordinate, ...]

    def __getitem__(self, key: str):
        if key != "target_coordinates":
            raise KeyError(key)
        return self.target_coordinates


class _BaseMultiTargetAdapter:
    def __init__(
        self,
        estimator_factory: Callable[[], object],
        capabilities: EstimatorCapabilities,
        target_coordinates: tuple[TargetCoordinate, ...],
        probabilistic_mode: str = "point",
        checkpoint: FitCheckpoint | None = None,
    ) -> None:
        if not callable(estimator_factory):
            raise TypeError("estimator_factory must be callable")
        if not isinstance(capabilities, EstimatorCapabilities):
            raise TypeError("capabilities must be EstimatorCapabilities")
        if probabilistic_mode not in {"point", "quantile"}:
            raise ValueError(f"unknown probabilistic_mode: {probabilistic_mode!r}")

        targets, horizon = self._validate_coordinates(target_coordinates)
        self._validate_capabilities(capabilities, probabilistic_mode)

        self.checkpoint = checkpoint
        self.estimator_factory = estimator_factory
        self.capabilities = capabilities
        self.target_coordinates = target_coordinates
        self.targets = targets
        self.horizon = horizon
        self.probabilistic_mode = probabilistic_mode
        self.metadata = AdapterMetadata(self.target_coordinates)
        self._is_fit = False

    def __getstate__(self):
        state = dict(self.__dict__)
        state["checkpoint"] = None
        return state

    @staticmethod
    def _validate_coordinates(
        coordinates: tuple[TargetCoordinate, ...],
    ) -> tuple[tuple[str, ...], int]:
        if not isinstance(coordinates, tuple):
            raise TypeError("target_coordinates must be a tuple")
        if not coordinates:
            raise ValueError("target_coordinates must be nonempty")
        if not all(isinstance(coordinate, TargetCoordinate) for coordinate in coordinates):
            raise TypeError("target_coordinates must contain TargetCoordinate values")

        first_step_targets = tuple(
            coordinate.target
            for coordinate in coordinates
            if coordinate.horizon_step == 1
        )
        if not first_step_targets or len(set(first_step_targets)) != len(first_step_targets):
            raise ValueError("target_coordinates must define unique first-step targets")
        horizon = max(coordinate.horizon_step for coordinate in coordinates)
        expected = tuple(
            TargetCoordinate(target, horizon_step)
            for horizon_step in range(1, horizon + 1)
            for target in first_step_targets
        )
        if coordinates != expected:
            raise ValueError("target_coordinates must be complete and time-major")
        return first_step_targets, horizon

    def _validate_capabilities(
        self,
        capabilities: EstimatorCapabilities,
        probabilistic_mode: str,
    ) -> None:
        raise NotImplementedError

    def _validated_fit_arrays(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        design = self._validated_design(X)
        targets = np.asarray(Y, dtype=float)
        expected_shape = (design.shape[0], self.horizon, len(self.targets))
        if targets.shape != expected_shape:
            raise ValueError(f"Y shape {targets.shape} does not match {expected_shape}")

        validated_weight = None
        if sample_weight is not None:
            if not self.capabilities.sample_weight:
                raise ValueError("estimator capabilities do not support sample_weight")
            validated_weight = np.asarray(sample_weight, dtype=float)
            if validated_weight.shape != (design.shape[0],):
                raise ValueError(
                    "sample_weight shape must match the training row axis"
                )
        return design, targets.reshape(design.shape[0], -1), validated_weight

    @staticmethod
    def _validated_design(X: np.ndarray) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        if design.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        return design

    def _fit_estimator(
        self,
        estimator: object,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> None:
        if not callable(getattr(estimator, "fit", None)):
            raise TypeError("estimator must provide fit(X, y)")
        if not callable(getattr(estimator, "predict", None)):
            raise TypeError("estimator must provide predict(X)")
        if sample_weight is None:
            estimator.fit(X, y)
        else:
            estimator.fit(X, y, sample_weight=sample_weight)

    @staticmethod
    def _scalar_prediction(
        estimator: object,
        X: np.ndarray,
    ) -> np.ndarray:
        prediction = np.asarray(estimator.predict(X), dtype=float)
        expected_shape = (X.shape[0],)
        if prediction.shape != expected_shape:
            raise ValueError(
                f"scalar prediction shape {prediction.shape} does not match {expected_shape}"
            )
        return prediction

    def _reshape_prediction(self, flat_prediction: np.ndarray) -> np.ndarray:
        # (N, steps*K) -> (N, steps, K)：time-major 合同唯一实现（tensors.py）
        return unflatten_time_major(
            flat_prediction,
            steps=self.horizon,
            width=len(self.targets),
        )

    def _require_fit(self) -> None:
        if not self._is_fit:
            raise RuntimeError("adapter must be fit before predict")


class IndependentMultiTargetAdapter(_BaseMultiTargetAdapter):
    def _validate_capabilities(
        self,
        capabilities: EstimatorCapabilities,
        probabilistic_mode: str,
    ) -> None:
        supported = (
            capabilities.scalar_target
            if probabilistic_mode == "point"
            else capabilities.scalar_quantile
        )
        if not supported:
            required = "scalar_target" if probabilistic_mode == "point" else "scalar_quantile"
            raise ValueError(f"independent adapter requires {required} capability")

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        *,
        max_workers: int = 1,
    ) -> "IndependentMultiTargetAdapter":
        fit_independent_adapters(
            ((self, X, Y, sample_weight),),
            max_workers=max_workers,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._require_fit()
        design = self._validated_design(X)
        flat_prediction = np.column_stack(
            tuple(self._scalar_prediction(estimator, design) for estimator in self.estimators)
        )
        return self._reshape_prediction(flat_prediction)


IndependentFitJob = tuple[
    IndependentMultiTargetAdapter,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]


def fit_independent_adapters(
    jobs: Sequence[IndependentFitJob],
    *,
    max_workers: int = 1,
) -> tuple[IndependentMultiTargetAdapter, ...]:
    """Fit all independent adapter outputs in one deterministic worker pool."""
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or max_workers <= 0
    ):
        raise ValueError("max_workers must be a positive integer")
    if not jobs:
        return ()

    prepared = []
    tasks = []
    for adapter, X, Y, sample_weight in jobs:
        if not isinstance(adapter, IndependentMultiTargetAdapter):
            raise TypeError("jobs must contain IndependentMultiTargetAdapter values")
        design, flat_targets, validated_weight = adapter._validated_fit_arrays(
            X, Y, sample_weight
        )
        factory = adapter.estimator_factory
        if not callable(factory):
            raise RuntimeError("adapter cannot be fit more than once")
        batch_fitter = getattr(factory, "fit_independent_outputs", None)
        if callable(batch_fitter) and flat_targets.shape[1] > 1:
            prepared.append((adapter, 1))
            tasks.append(
                (
                    "batch",
                    adapter,
                    batch_fitter,
                    design,
                    flat_targets,
                    validated_weight,
                )
            )
            continue
        estimators = tuple(factory() for _ in range(flat_targets.shape[1]))
        prepared.append((adapter, len(estimators)))
        tasks.extend(
            (
                "scalar",
                column_index,
                adapter,
                estimator,
                design,
                flat_targets[:, column_index],
                validated_weight,
            )
            for column_index, estimator in enumerate(estimators)
        )

    def fit_task(task):
        if task[0] == "batch":
            _, adapter, batch_fitter, design, targets, sample_weight = task
            def fit_batch():
                return tuple(batch_fitter(design, targets, sample_weight))
            if adapter.checkpoint is None:
                return fit_batch()
            return adapter.checkpoint.run(identity={"model": "batch_outputs"},
                arrays=(design, targets, sample_weight), fit=fit_batch)
        _, column_index, adapter, estimator, design, target, sample_weight = task
        def fit_scalar():
            adapter._fit_estimator(estimator, design, target, sample_weight)
            return estimator
        if adapter.checkpoint is not None:
            estimator = adapter.checkpoint.run(identity={"model": f"scalar/{column_index}"},
                arrays=(design, target, sample_weight), fit=fit_scalar)
        else:
            estimator = fit_scalar()
        return (estimator,)

    worker_count = min(max_workers, len(tasks))
    if worker_count == 1:
        fitted = tuple(fit_task(task) for task in tasks)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            fitted = tuple(executor.map(fit_task, tasks))

    cursor = 0
    adapters = []
    for adapter, task_count in prepared:
        task_results = fitted[cursor : cursor + task_count]
        adapter.estimators = tuple(
            estimator
            for result in task_results
            for estimator in result
        )
        adapter.estimator_factory = None
        adapter._is_fit = True
        adapters.append(adapter)
        cursor += task_count
    return tuple(adapters)


class RegressorChainMultiTargetAdapter(_BaseMultiTargetAdapter):
    def _validate_capabilities(
        self,
        capabilities: EstimatorCapabilities,
        probabilistic_mode: str,
    ) -> None:
        if probabilistic_mode == "quantile":
            raise ValueError("regressor_chain does not support quantile mode")
        if not capabilities.scalar_target:
            raise ValueError("regressor_chain requires scalar_target capability")

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "RegressorChainMultiTargetAdapter":
        design, flat_targets, validated_weight = self._validated_fit_arrays(
            X, Y, sample_weight
        )
        estimators = []
        for column_index in range(flat_targets.shape[1]):
            estimator = self.estimator_factory()
            chained_design = np.column_stack(
                (design, flat_targets[:, :column_index])
            )
            def fit_chain_unit():
                self._fit_estimator(estimator, chained_design,
                                    flat_targets[:, column_index], validated_weight)
                return estimator
            if self.checkpoint is None:
                estimator = fit_chain_unit()
            else:
                estimator = self.checkpoint.run(
                    identity={"model": f"chain/{column_index}"},
                    arrays=(chained_design, flat_targets[:, column_index], validated_weight),
                    fit=fit_chain_unit,
                )
            estimators.append(estimator)
        self.estimators = tuple(estimators)
        self.estimator_factory = None
        self._is_fit = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._require_fit()
        design = self._validated_design(X)
        predictions = []
        for estimator in self.estimators:
            chained_design = np.column_stack((design, *predictions))
            predictions.append(self._scalar_prediction(estimator, chained_design))
        return self._reshape_prediction(np.column_stack(tuple(predictions)))


class NativeMultiTargetAdapter(_BaseMultiTargetAdapter):
    def _validate_capabilities(
        self,
        capabilities: EstimatorCapabilities,
        probabilistic_mode: str,
    ) -> None:
        supported = (
            capabilities.native_multi_target_point
            if probabilistic_mode == "point"
            else capabilities.native_multi_target_quantile
        )
        if not supported:
            required = (
                "native_multi_target_point"
                if probabilistic_mode == "point"
                else "native_multi_target_quantile"
            )
            raise ValueError(f"native adapter requires {required} capability")

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "NativeMultiTargetAdapter":
        design, flat_targets, validated_weight = self._validated_fit_arrays(
            X, Y, sample_weight
        )
        estimator = self.estimator_factory()
        def fit_native():
            self._fit_estimator(estimator, design, flat_targets, validated_weight)
            return estimator
        if self.checkpoint is None:
            estimator = fit_native()
        else:
            estimator = self.checkpoint.run(identity={"model": "native"},
                arrays=(design, flat_targets, validated_weight), fit=fit_native)
        self.estimator = estimator
        self.estimator_factory = None
        self._is_fit = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._require_fit()
        design = self._validated_design(X)
        flat_prediction = np.asarray(self.estimator.predict(design), dtype=float)
        expected_shape = (design.shape[0], len(self.target_coordinates))
        if flat_prediction.shape != expected_shape:
            raise ValueError(
                f"native prediction shape {flat_prediction.shape} does not match "
                f"{expected_shape}"
            )
        return self._reshape_prediction(flat_prediction)
