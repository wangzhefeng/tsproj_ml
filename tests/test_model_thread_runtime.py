# -*- coding: utf-8 -*-
"""Estimator 运行线程参数解析测试。"""
from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from forecasting_core.specs import (
    ColumnSpec,
    DataSourceSpec,
    DataSpec,
    EstimatorSpec,
    FeatureSpec,
    ForecastConfigSpec,
    ForecastProblemSpec,
    ForecastStrategySpec,
)
from model_forecasting.fit_service import (
    _runtime_estimator_params,
    _runtime_fit_worker_plan,
    _runtime_model_workers,
    _runtime_scalar_fit_count,
)
from model_training.estimators import make_model_factory, resolve_model_capabilities
from model_training.trainer import CanonicalTrainer


def _config(
    *,
    strategy: str,
    model_type: str = "lightgbm",
    performance: dict[str, int] | None = None,
    horizon: int = 2,
    output_chunk_length: int | None = None,
    mode: str = "point",
    direct_layout: str | None = None,
) -> ForecastConfigSpec:
    validation = {
        "forecast_origin": "2026-01-03T00:00:00",
        "history_steps": 24,
        "train_window_steps": 12,
        "fold_count": 1,
        "stride_steps": 2,
    }
    if performance is not None:
        validation["performance"] = performance
    return ForecastConfigSpec(
        problem=ForecastProblemSpec(
            time_col="time",
            freq="1h",
            horizon=horizon,
            targets=("load",),
            training_scope="local",
            series_id_cols=(),
        ),
        data=DataSpec(
            (
                DataSourceSpec(
                    name="target_history",
                    source_type="file",
                    columns=(ColumnSpec("load", "target"),),
                    history_path="unused.csv",
                    time_col="time",
                    availability="source_time",
                ),
            )
        ),
        features=FeatureSpec(
            target_lags={"load": (2,)},
            observed_past_lags={},
            datetime_features=(),
            transformations=(
                {}
                if direct_layout is None
                else {
                    "direct": {
                        "layout": direct_layout,
                        "align_to_target": True,
                        "horizon_feature": {
                            "name": "forecast_horizon_idx",
                            "cyclical": False,
                        },
                    }
                }
            ),
        ),
        strategy=ForecastStrategySpec(
            strategy,
            output_chunk_length=output_chunk_length,
        ),
        estimator=EstimatorSpec(
            model_type=model_type,
            target_adapter="independent",
            params={},
        ),
        probabilistic=(
            {"mode": "point"}
            if mode == "point"
            else {
                "mode": "quantile",
                "quantiles": [0.1, 0.5, 0.9],
                "point_quantile": 0.5,
            }
        ),
        validation=validation,
        output={"scenario_subpath": "thread-runtime"},
    )


class _ConcurrentEstimator:
    lock = threading.Lock()
    active = 0
    max_active = 0

    @classmethod
    def reset(cls) -> None:
        with cls.lock:
            cls.active = 0
            cls.max_active = 0

    def fit(self, X, y, sample_weight=None):
        with type(self).lock:
            type(self).active += 1
            type(self).max_active = max(type(self).max_active, type(self).active)
        try:
            time.sleep(0.02)
            self.value = float(np.mean(y))
            return self
        finally:
            with type(self).lock:
                type(self).active -= 1

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


class _RecordingEstimator:
    created = []

    def __init__(self):
        self.fit_X = None
        self.fit_y = None
        self.fit_sample_weight = None
        type(self).created.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.created = []

    def fit(self, X, y, sample_weight=None):
        self.fit_X = np.asarray(X, dtype=float).copy()
        self.fit_y = np.asarray(y, dtype=float).copy()
        self.fit_sample_weight = (
            None
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).copy()
        )
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


class RuntimeEstimatorParamsTest(unittest.TestCase):
    def test_single_model_horizon_uses_one_shared_fit_plan(self) -> None:
        config = _config(
            strategy="direct",
            horizon=4,
            direct_layout="single_model_horizon",
        )
        trainer = CanonicalTrainer(
            config,
            estimator_factory=_ConcurrentEstimator,
            capabilities=resolve_model_capabilities("ridge"),
            feature_schema=("x", "forecast_horizon_idx"),
        )

        self.assertEqual(trainer.target_plan.model_count, 1)
        self.assertEqual(trainer.target_plan.model_indices, (0, 0, 0, 0))
        self.assertEqual(_runtime_scalar_fit_count(config), 1)
        self.assertEqual(_runtime_model_workers(config), 1)
        self.assertGreaterEqual(_runtime_estimator_params(config)["n_jobs"], 1)

    def test_single_model_horizon_pools_calls_in_time_major_order(self) -> None:
        _RecordingEstimator.reset()
        config = _config(
            strategy="direct",
            model_type="ridge",
            horizon=4,
            direct_layout="single_model_horizon",
        )
        designs = tuple(
            np.column_stack(
                (
                    np.arange(3.0),
                    np.full(3, float(step)),
                )
            )
            for step in range(1, 5)
        )
        targets = np.stack(
            tuple(100.0 * step + np.arange(3.0) for step in range(1, 5)),
            axis=1,
        )[:, :, None]
        sample_weight = np.array([1.0, 2.0, 3.0])

        artifact = CanonicalTrainer(
            config,
            estimator_factory=_RecordingEstimator,
            capabilities=resolve_model_capabilities("ridge"),
            feature_schema=("x", "forecast_horizon_idx"),
        ).train(
            designs,
            targets,
            sample_weight=sample_weight,
            max_workers=4,
        )

        self.assertEqual(artifact.model_count, 1)
        self.assertEqual(len(_RecordingEstimator.created), 1)
        estimator = _RecordingEstimator.created[0]
        np.testing.assert_array_equal(estimator.fit_X, np.concatenate(designs, axis=0))
        np.testing.assert_array_equal(
            estimator.fit_y,
            np.concatenate(tuple(targets[:, step, 0] for step in range(4))),
        )
        np.testing.assert_array_equal(
            estimator.fit_sample_weight,
            np.tile(sample_weight, 4),
        )

    def test_multi_model_lightgbm_defaults_to_one_thread(self) -> None:
        params = _runtime_estimator_params(_config(strategy="direct"))
        self.assertEqual(params["n_jobs"], 1)

    def test_single_scalar_lightgbm_receives_resolved_model_threads(self) -> None:
        params = _runtime_estimator_params(_config(strategy="recursive"))
        self.assertGreaterEqual(params["n_jobs"], 1)

    def test_mimo_lightgbm_uses_single_thread_per_output(self) -> None:
        params = _runtime_estimator_params(_config(strategy="mimo"))
        self.assertEqual(params["n_jobs"], 1)

    def test_multi_output_threaded_models_default_to_one_model_thread(self) -> None:
        cases = (
            ("xgboost", "n_jobs"),
            ("catboost", "thread_count"),
            ("randomforest", "n_jobs"),
        )
        for model_type, parameter in cases:
            with self.subTest(model_type=model_type):
                params = _runtime_estimator_params(
                    _config(strategy="direct", model_type=model_type)
                )
                self.assertEqual(params[parameter], 1)

    def test_explicit_model_thread_count_overrides_runtime_default(self) -> None:
        params = _runtime_estimator_params(
            _config(
                strategy="direct",
                performance={"model_thread_count": 3},
            )
        )
        self.assertEqual(params["n_jobs"], 3)

    def test_multi_model_lightgbm_defaults_to_at_most_four_workers(self) -> None:
        self.assertEqual(_runtime_model_workers(_config(strategy="direct")), 2)

    def test_supported_models_receive_bounded_output_workers(self) -> None:
        expected = {
            "xgboost": 2,
            "catboost": 2,
            "randomforest": 2,
            "ridge": 2,
            "lasso": 2,
            "enet": 2,
        }
        for model_type, workers in expected.items():
            with self.subTest(model_type=model_type):
                self.assertEqual(
                    _runtime_model_workers(
                        _config(strategy="direct", model_type=model_type)
                    ),
                    workers,
                )

    def test_single_group_multi_output_strategy_uses_output_workers(self) -> None:
        self.assertEqual(_runtime_model_workers(_config(strategy="mimo")), 2)

    def test_mimo_trainer_applies_workers_inside_single_model_group(self) -> None:
        _ConcurrentEstimator.reset()
        config = _config(strategy="mimo", model_type="ridge")
        X = np.arange(12.0).reshape(-1, 1)
        Y = np.stack((2.0 * X[:, 0], -3.0 * X[:, 0]), axis=1)[:, :, None]

        CanonicalTrainer(
            config,
            estimator_factory=_ConcurrentEstimator,
            capabilities=resolve_model_capabilities("ridge"),
            feature_schema=("x",),
        ).train((X,), Y, max_workers=2)

        self.assertGreaterEqual(_ConcurrentEstimator.max_active, 2)

    def test_dirmo_flattens_workers_across_groups_and_outputs(self) -> None:
        _ConcurrentEstimator.reset()
        config = _config(
            strategy="dirmo",
            model_type="ridge",
            horizon=4,
            output_chunk_length=2,
        )
        first = np.arange(12.0).reshape(-1, 1)
        second = first + 100.0
        Y = np.stack(
            tuple((step + 1.0) * first[:, 0] for step in range(4)),
            axis=1,
        )[:, :, None]

        CanonicalTrainer(
            config,
            estimator_factory=_ConcurrentEstimator,
            capabilities=resolve_model_capabilities("ridge"),
            feature_schema=("x",),
        ).train((first, second), Y, max_workers=4)

        self.assertGreaterEqual(_ConcurrentEstimator.max_active, 4)

    def test_quantile_direct_prefers_output_workers_without_nesting(self) -> None:
        config = _config(strategy="direct", mode="quantile")
        self.assertEqual(_runtime_fit_worker_plan(config), (1, 2))

    def test_quantile_recursive_prefers_level_workers(self) -> None:
        config = _config(strategy="recursive", mode="quantile")
        self.assertEqual(_runtime_fit_worker_plan(config), (3, 1))

    def test_explicit_quantile_and_output_workers_conflict_raises(self) -> None:
        config = _config(
            strategy="direct",
            mode="quantile",
            performance={
                "quantile_parallel_workers": 2,
                "multi_output_n_jobs": 2,
            },
        )
        with self.assertRaisesRegex(ValueError, "outer parallel axes"):
            _runtime_fit_worker_plan(config)

    def test_single_scalar_fit_strategy_uses_one_worker(self) -> None:
        self.assertEqual(_runtime_model_workers(_config(strategy="recursive")), 1)

    def test_explicit_multi_output_workers_override_runtime_default(self) -> None:
        self.assertEqual(
            _runtime_model_workers(
                _config(
                    strategy="direct",
                    performance={"multi_output_n_jobs": 2},
                )
            ),
            2,
        )

    def test_parallel_model_groups_match_serial_order_and_predictions(self) -> None:
        config = _config(strategy="direct", model_type="ridge")
        capabilities = resolve_model_capabilities("ridge")
        factory = make_model_factory(
            "ridge",
            {"alpha": 1e-6},
            feature_names=("x",),
        )
        X_by_call = (
            np.arange(12.0).reshape(-1, 1),
            np.arange(12.0, 24.0).reshape(-1, 1),
        )
        Y = np.stack(
            (
                2.0 * X_by_call[0][:, 0] + 1.0,
                -3.0 * X_by_call[1][:, 0] + 5.0,
            ),
            axis=1,
        )[:, :, None]

        serial = CanonicalTrainer(
            config,
            estimator_factory=factory,
            capabilities=capabilities,
            feature_schema=("x",),
        ).train(X_by_call, Y, max_workers=1)
        parallel = CanonicalTrainer(
            config,
            estimator_factory=factory,
            capabilities=capabilities,
            feature_schema=("x",),
        ).train(X_by_call, Y, max_workers=2)

        self.assertEqual(
            tuple(group.model_index for group in parallel.model_groups),
            tuple(group.model_index for group in serial.model_groups),
        )
        for call_index, (serial_group, parallel_group) in enumerate(
            zip(serial.model_groups, parallel.model_groups)
        ):
            np.testing.assert_allclose(
                parallel_group.predictor.predict(X_by_call[call_index]),
                serial_group.predictor.predict(X_by_call[call_index]),
            )

    def test_unsupported_explicit_thread_control_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "model_thread_count.*ridge"):
            _runtime_estimator_params(
                _config(
                    strategy="direct",
                    model_type="ridge",
                    performance={"model_thread_count": 2},
                )
            )


if __name__ == "__main__":
    unittest.main()
