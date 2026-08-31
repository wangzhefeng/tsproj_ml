# -*- coding: utf-8 -*-
"""Canonical trainer（2026-08-29 架构收敛自 models/ModelTraining.py 迁入，类实现逐字保真）。"""

# python libraries
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from model_training.estimators import (
    EstimatorCapabilities,
    IndependentMultiTargetAdapter,
    NativeMultiTargetAdapter,
    RegressorChainMultiTargetAdapter,
)
from forecasting_core.specs import ForecastConfigSpec, TargetAdapter
from model_training.strategies import (
    AdapterPredictor,
    CanonicalStrategyArtifact,
    StrategyModelGroupArtifact,
    StrategyTargetPlan,
    TargetCoordinate,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class DirectMultiOutputRegressor:
    """
    为 Direct 多步预测定制的多输出训练器。

    - 每个 horizon 单独训练一个回归器
    - 支持为每个输出传入独立 eval_set / early stopping
    - 支持按输出维度并行训练
    """

    def __init__(self, estimator_factory, n_jobs: int = 1, log_prefix: str = "[DirectMultiOutputRegressor]"):
        self.estimator_factory = estimator_factory
        self.n_jobs = max(1, int(n_jobs or 1))
        self.log_prefix = log_prefix
        self.estimators_: List[Any] = []

    @staticmethod
    def _to_1d(values: Any) -> np.ndarray:
        return np.asarray(values).reshape(-1)

    def _fit_single_output(self, output_idx: int, X_train, y_train, fit_kwargs: Optional[Dict[str, Any]] = None):
        estimator = self.estimator_factory()
        estimator.fit(X_train, self._to_1d(y_train), **(fit_kwargs or {}))
        return output_idx, estimator

    def fit(self, X_train, Y_train, fit_kwargs_list: Optional[List[Dict[str, Any]]] = None):
        y_frame = Y_train if isinstance(Y_train, pd.DataFrame) else pd.DataFrame(Y_train)
        n_outputs = y_frame.shape[1]
        fit_kwargs_list = fit_kwargs_list or [{} for _ in range(n_outputs)]
        if len(fit_kwargs_list) != n_outputs:
            raise ValueError(
                f"{self.log_prefix} fit_kwargs_list length ({len(fit_kwargs_list)}) "
                f"does not match n_outputs ({n_outputs})."
            )

        estimators = [None] * n_outputs
        if self.n_jobs > 1 and n_outputs > 1:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [
                    executor.submit(
                        self._fit_single_output,
                        output_idx,
                        X_train,
                        y_frame.iloc[:, output_idx],
                        fit_kwargs_list[output_idx],
                    )
                    for output_idx in range(n_outputs)
                ]
                for future in as_completed(futures):
                    output_idx, estimator = future.result()
                    estimators[output_idx] = estimator
        else:
            for output_idx in range(n_outputs):
                _, estimator = self._fit_single_output(
                    output_idx,
                    X_train,
                    y_frame.iloc[:, output_idx],
                    fit_kwargs_list[output_idx],
                )
                estimators[output_idx] = estimator

        self.estimators_ = estimators
        # 训练完成后不再依赖工厂，清空以避免模型保存时因闭包/lambda 无法 pickle。
        self.estimator_factory = None
        return self

    def predict(self, X) -> np.ndarray:
        if not self.estimators_:
            raise ValueError(f"{self.log_prefix} multi-output estimators are not fitted yet.")
        preds = [self._to_1d(estimator.predict(X)) for estimator in self.estimators_]
        return np.column_stack(preds)


class HorizonAlignedDirectRegressor(DirectMultiOutputRegressor):
    """每个输出仅消费其对应目标horizon外生列的Direct训练器。

    输入宽表仍使用 ``feature_h1..feature_hH`` 表达目标时刻外生轨迹，
    但第h个估计器只接收 ``feature_h{h}``，并在交给基模型前重命名回
    canonical ``feature``。原点lag/advanced以及未展开的origin-frozen列共享。
    """

    _HORIZON_PATTERN = re.compile(r"^(.+)_h(\d+)$")

    def __init__(self, estimator_factory, n_jobs: int = 1, log_prefix: str = "[HorizonAlignedDirectRegressor]"):
        super().__init__(estimator_factory, n_jobs=n_jobs, log_prefix=log_prefix)
        self.shared_features_: List[str] = []
        self.horizon_bases_: List[str] = []

    def _resolve_feature_layout(self, columns: List[str], n_outputs: int) -> None:
        expanded = {}
        expanded_columns = set()
        for column in columns:
            match = self._HORIZON_PATTERN.match(str(column))
            if match is None:
                continue
            base = match.group(1)
            horizon = int(match.group(2))
            expanded.setdefault(base, {})[horizon] = column
            expanded_columns.add(column)
        if not expanded:
            raise ValueError(f"{self.log_prefix} no horizon-aware exogenous columns found.")
        for base, by_horizon in expanded.items():
            missing = [h for h in range(1, n_outputs + 1) if h not in by_horizon]
            if missing:
                raise ValueError(
                    f"{self.log_prefix} feature '{base}' missing horizon {missing[0]} "
                    f"for {n_outputs} outputs."
                )
        expanded_bases = set(expanded)
        self.shared_features_ = [
            column for column in columns
            if column not in expanded_columns and column not in expanded_bases
        ]
        self.horizon_bases_ = list(expanded)

    def _frame_for_output(self, X: pd.DataFrame, output_idx: int) -> pd.DataFrame:
        horizon = output_idx + 1
        result = X.reindex(columns=self.shared_features_).copy()
        for base in self.horizon_bases_:
            source = f"{base}_h{horizon}"
            if source not in X.columns:
                raise ValueError(f"{self.log_prefix} feature '{base}' missing horizon {horizon}.")
            result[base] = X[source].to_numpy()
        return result

    def _fit_single_output(self, output_idx: int, X_train, y_train, fit_kwargs: Optional[Dict[str, Any]] = None):
        X_output = self._frame_for_output(X_train, output_idx)
        kwargs = dict(fit_kwargs or {})
        if kwargs.get("eval_set"):
            kwargs["eval_set"] = [
                (self._frame_for_output(X_eval, output_idx), y_eval)
                for X_eval, y_eval in kwargs["eval_set"]
            ]
        return super()._fit_single_output(output_idx, X_output, y_train, kwargs)

    def fit(self, X_train, Y_train, fit_kwargs_list: Optional[List[Dict[str, Any]]] = None):
        y_frame = Y_train if isinstance(Y_train, pd.DataFrame) else pd.DataFrame(Y_train)
        self._resolve_feature_layout(list(X_train.columns), y_frame.shape[1])
        return super().fit(X_train, y_frame, fit_kwargs_list=fit_kwargs_list)

    def predict(self, X) -> np.ndarray:
        if not self.estimators_:
            raise ValueError(f"{self.log_prefix} multi-output estimators are not fitted yet.")
        preds = [
            self._to_1d(estimator.predict(self._frame_for_output(X, output_idx)))
            for output_idx, estimator in enumerate(self.estimators_)
        ]
        return np.column_stack(preds)


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
        self.config = config
        self.estimator_factory = estimator_factory
        self.capabilities = capabilities
        self.feature_schema = normalized_schema
        self.probabilistic_mode = mode
        self.target_plan = StrategyTargetPlan.from_spec(
            config.strategy,
            config.problem.targets,
            config.problem.horizon,
        )

    def train(
        self,
        X_by_call: Sequence[np.ndarray],
        Y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        n_series: int = 1,
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

        validated_weight = None
        if sample_weight is not None:
            validated_weight = np.asarray(sample_weight, dtype=float)
            if validated_weight.shape != (n_rows,):
                raise ValueError("sample_weight must match the training row axis")

        model_groups = []
        for model_index in range(self.target_plan.model_count):
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
            adapter_type = self._ADAPTERS[
                self.config.estimator.target_adapter
            ]
            adapter = adapter_type(
                self.estimator_factory,
                self.capabilities,
                local_coordinates,
                probabilistic_mode=self.probabilistic_mode,
            ).fit(
                group_design,
                group_targets,
                sample_weight=group_weight,
            )
            model_groups.append(
                StrategyModelGroupArtifact(
                    model_index=model_index,
                    coordinate_groups=coordinate_groups,
                    predictor=AdapterPredictor(adapter),
                )
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
        return flat.reshape(
            len(targets),
            steps,
            len(self.config.problem.targets),
        )


__all__ = [
    "CanonicalTrainer",
    "DirectMultiOutputRegressor",
    "HorizonAlignedDirectRegressor",
]
