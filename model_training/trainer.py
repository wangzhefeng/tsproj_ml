# -*- coding: utf-8 -*-
"""Canonical trainer（2026-08-29 架构收敛自 models/ModelTraining.py 迁入，类实现逐字保真）。"""

# python libraries
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np

from model_training.estimators import (
    EstimatorCapabilities,
    IndependentMultiTargetAdapter,
    NativeMultiTargetAdapter,
    RegressorChainMultiTargetAdapter,
    fit_independent_adapters,
)
from forecasting_core.checkpoints import FitCheckpoint
from forecasting_core.specs import ForecastConfigSpec, TargetAdapter
from forecasting_core.tensors import unflatten_time_major
from model_training.strategies import (
    AdapterPredictor,
    CanonicalStrategyArtifact,
    StrategyModelGroupArtifact,
    TargetCoordinate,
    target_plan_for_config,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class CanonicalTrainer:
    """Train canonical strategy model groups without reading legacy method strings."""

    _ADAPTERS = {
        TargetAdapter.INDEPENDENT: IndependentMultiTargetAdapter,
        TargetAdapter.REGRESSOR_CHAIN: RegressorChainMultiTargetAdapter,
        TargetAdapter.NATIVE: NativeMultiTargetAdapter,
    }

    def __init__(
        self,
        config: ForecastConfigSpec,
        *,
        estimator_factory,
        capabilities: EstimatorCapabilities,
        feature_schema: Sequence[str],
        checkpoint: FitCheckpoint | None = None,
    ) -> None:
        if not isinstance(config, ForecastConfigSpec):
            raise TypeError("config must be a ForecastConfigSpec")
        if config.strategy is None:
            raise ValueError(
                "CanonicalTrainer requires a strategy config; train ensemble members separately"
            )
        if not callable(estimator_factory):
            raise TypeError("estimator_factory must be callable")
        if not isinstance(capabilities, EstimatorCapabilities):
            raise TypeError("capabilities must be EstimatorCapabilities")
        if isinstance(feature_schema, (str, bytes)) or not isinstance(feature_schema, Sequence):
            raise TypeError("feature_schema must be a sequence of names")
        normalized_schema = tuple(str(name) for name in feature_schema)
        if not normalized_schema or any(
            not name or name != name.strip() for name in normalized_schema
        ):
            raise ValueError(
                "feature_schema names must be nonblank without surrounding whitespace"
            )
        if len(normalized_schema) != len(set(normalized_schema)):
            raise ValueError("feature_schema names must be unique")

        mode = str(config.probabilistic.get("mode", "point"))
        config.estimator.validate_capabilities(capabilities, mode)
        self.checkpoint = checkpoint
        self.config = config
        self.estimator_factory = estimator_factory
        self.capabilities = capabilities
        self.feature_schema = normalized_schema
        self.probabilistic_mode = mode
        self.target_plan = target_plan_for_config(config)

    def train(
        self,
        X_by_call: Sequence[np.ndarray],
        Y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        n_series: int = 1,
        max_workers: int = 1,
    ) -> CanonicalStrategyArtifact:
        if isinstance(X_by_call, np.ndarray) or not isinstance(X_by_call, Sequence):
            raise TypeError(
                "X_by_call must be a sequence of two-dimensional designs"
            )
        designs = tuple(np.asarray(design, dtype=float) for design in X_by_call)
        if len(designs) != len(self.target_plan.call_coordinates):
            raise ValueError(
                f"X_by_call must contain {len(self.target_plan.call_coordinates)} designs"
            )
        if any(design.ndim != 2 for design in designs):
            raise ValueError("every X_by_call design must be two-dimensional")
        n_rows = designs[0].shape[0]
        expected_width = len(self.feature_schema)
        if any(
            design.shape != (n_rows, expected_width) for design in designs
        ):
            raise ValueError(
                "all X_by_call designs must share the configured fixed feature schema"
            )
        targets = np.asarray(Y, dtype=float)
        expected_target_shape = (
            n_rows,
            self.config.problem.horizon,
            len(self.config.problem.targets),
        )
        if targets.shape != expected_target_shape:
            raise ValueError(
                f"Y shape {targets.shape} does not match {expected_target_shape}"
            )
        if not np.isfinite(targets).all() or any(
            not np.isfinite(design).all() for design in designs
        ):
            raise ValueError("canonical training arrays must contain only finite values")
        if (
            isinstance(n_series, bool)
            or not isinstance(n_series, int)
            or n_series <= 0
        ):
            raise ValueError("n_series must be a positive integer")

        if (
            isinstance(max_workers, bool)
            or not isinstance(max_workers, int)
            or max_workers <= 0
        ):
            raise ValueError("max_workers must be a positive integer")

        validated_weight = None
        if sample_weight is not None:
            validated_weight = np.asarray(sample_weight, dtype=float)
            if validated_weight.shape != (n_rows,):
                raise ValueError("sample_weight must match the training row axis")

        def prepare_model_group(model_index: int):
            call_indices = tuple(
                index
                for index, candidate in enumerate(self.target_plan.model_indices)
                if candidate == model_index
            )
            coordinate_groups = tuple(
                self.target_plan.call_coordinates[index] for index in call_indices
            )
            steps_per_call = len(coordinate_groups[0]) // len(
                self.config.problem.targets
            )
            if any(
                len(group)
                != steps_per_call * len(self.config.problem.targets)
                for group in coordinate_groups
            ):
                raise ValueError(
                    "shared strategy model calls must have one stable output schema"
                )

            group_design = np.concatenate(
                [designs[index] for index in call_indices],
                axis=0,
            )
            group_targets = np.concatenate(
                [self._target_block(targets, group) for group in coordinate_groups],
                axis=0,
            )
            group_weight = (
                np.tile(validated_weight, len(call_indices))
                if validated_weight is not None
                else None
            )
            local_coordinates = tuple(
                TargetCoordinate(target, step)
                for step in range(1, steps_per_call + 1)
                for target in self.config.problem.targets
            )
            return (
                coordinate_groups,
                group_design,
                group_targets,
                group_weight,
                local_coordinates,
            )

        model_indices = tuple(range(self.target_plan.model_count))
        prepared_groups = tuple(
            prepare_model_group(model_index) for model_index in model_indices
        )
        def group_checkpoint(model_index):
            if self.checkpoint is None:
                return None
            return self.checkpoint.child(
                config=self.config.fingerprint(), feature_schema=self.feature_schema,
                capabilities=self.capabilities.canonical_payload(), group=model_index,
                n_series=n_series,
            )

        adapter_type = self._ADAPTERS[self.config.estimator.target_adapter]
        if adapter_type is IndependentMultiTargetAdapter:
            adapters = tuple(
                IndependentMultiTargetAdapter(
                    self.estimator_factory,
                    self.capabilities,
                    prepared[4],
                    probabilistic_mode=self.probabilistic_mode,
                    checkpoint=group_checkpoint(model_index),
                )
                for model_index, prepared in enumerate(prepared_groups)
            )
            fitted_adapters = fit_independent_adapters(
                tuple(
                    (adapter, prepared[1], prepared[2], prepared[3])
                    for adapter, prepared in zip(adapters, prepared_groups)
                ),
                max_workers=max_workers,
            )
            model_groups = tuple(
                StrategyModelGroupArtifact(
                    model_index=model_index,
                    coordinate_groups=prepared[0],
                    predictor=AdapterPredictor(adapter),
                )
                for model_index, prepared, adapter in zip(
                    model_indices,
                    prepared_groups,
                    fitted_adapters,
                )
            )
        else:
            def fit_model_group(model_index: int) -> StrategyModelGroupArtifact:
                prepared = prepared_groups[model_index]
                adapter = adapter_type(
                    self.estimator_factory,
                    self.capabilities,
                    prepared[4],
                    probabilistic_mode=self.probabilistic_mode,
                    checkpoint=group_checkpoint(model_index),
                ).fit(
                    prepared[1],
                    prepared[2],
                    sample_weight=prepared[3],
                )
                return StrategyModelGroupArtifact(
                    model_index=model_index,
                    coordinate_groups=prepared[0],
                    predictor=AdapterPredictor(adapter),
                )

            worker_count = min(max_workers, len(model_indices))
            if worker_count == 1:
                model_groups = tuple(
                    fit_model_group(index) for index in model_indices
                )
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    model_groups = tuple(
                        executor.map(fit_model_group, model_indices)
                    )

        return CanonicalStrategyArtifact(
            target_plan=self.target_plan,
            model_groups=tuple(model_groups),
            feature_schema=self.feature_schema,
            estimator_coupling=self.config.estimator.target_adapter.value,
            estimator_capabilities=tuple(
                sorted(self.capabilities.canonical_payload().items())
            ),
            N=n_series,
            H=self.config.problem.horizon,
            K=len(self.config.problem.targets),
        )

    def _target_block(
        self,
        targets: np.ndarray,
        coordinates: tuple[TargetCoordinate, ...],
    ) -> np.ndarray:
        target_indices = {
            target: index
            for index, target in enumerate(self.config.problem.targets)
        }
        flat = np.column_stack(
            tuple(
                targets[
                    :,
                    coordinate.horizon_step - 1,
                    target_indices[coordinate.target],
                ]
                for coordinate in coordinates
            )
        )
        steps = len(coordinates) // len(self.config.problem.targets)
        # 列收集即 time-major 顺序，收口到 (N, steps, K) 走张量合同唯一实现
        return unflatten_time_major(flat, steps=steps, width=len(self.config.problem.targets))


__all__ = [
    "CanonicalTrainer",
]
