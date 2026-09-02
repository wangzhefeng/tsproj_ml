# -*- coding: utf-8 -*-
"""quantile 训练优化测试（2026-09-01，Task 4）。

钉住两条契约：
1. level 线程并行的产物与串行逐值一致（并行只是调度，不改变数值）；
2. xgb 原生多分位共享池：每个子模型位置只训练一次，逐 level 视图切列，
   经 canonical trainer/forecaster 全链形状与语义正确。
"""

import io
import pickle
import unittest

import numpy as np
import pandas as pd

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
from model_forecasting.forecaster import CanonicalMarginalQuantileForecaster
from model_training.estimators import (
    EstimatorCapabilities,
    SharedMultiQuantilePool,
    supports_native_multi_quantile,
)
from probabilistic.training import CanonicalMarginalQuantileTrainer


class MeanQuantileRegressor:
    def __init__(self, level):
        self.level = float(level)

    def fit(self, X, y, sample_weight=None):
        self.value = float(np.mean(y) + (self.level - 0.5) * 2.0)
        return self

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


CAPABILITIES = EstimatorCapabilities(
    scalar_target=True,
    scalar_quantile=True,
    native_multi_target_point=False,
    native_multi_target_quantile=False,
    sample_weight=False,
    categorical=False,
    nan_support=False,
)


def _config(model_type: str = "mean_quantile") -> ForecastConfigSpec:
    targets = ("load",)
    return ForecastConfigSpec(
        problem=ForecastProblemSpec(
            time_col="time",
            freq="1h",
            horizon=2,
            targets=targets,
            information_mode="forecast",
            training_scope="local",
            series_id_cols=(),
        ),
        data=DataSpec(
            (
                DataSourceSpec(
                    name="targets",
                    source_type="file",
                    columns=tuple(ColumnSpec(target, "target") for target in targets),
                    history_path="unused.csv",
                    time_col="time",
                    availability="source_time",
                ),
            )
        ),
        features=FeatureSpec(
            target_lags={},
            observed_past_lags={},
            datetime_features=(),
            transformations={},
        ),
        strategy=ForecastStrategySpec("direct"),
        estimator=EstimatorSpec(
            model_type=model_type,
            target_adapter="independent",
        ),
        probabilistic={
            "mode": "quantile",
            "quantiles": [0.1, 0.5, 0.9],
            "point_quantile": 0.5,
        },
        validation={},
        output={},
    )


def _toy_arrays():
    X_by_call = (
        np.arange(16.0).reshape(-1, 1),
        np.arange(16.0).reshape(-1, 1) * 2.0,
    )
    rng = np.random.default_rng(42)
    Y = rng.normal(50.0, 5.0, size=(16, 2, 1))
    return X_by_call, Y


class LevelParallelismTest(unittest.TestCase):
    def test_parallel_matches_sequential_exactly(self):
        config = _config()
        X_by_call, Y = _toy_arrays()

        def factory_for_level(level):
            return lambda: MeanQuantileRegressor(level)

        sequential = CanonicalMarginalQuantileTrainer(
            config,
            estimator_factory_for_level=factory_for_level,
            capabilities=CAPABILITIES,
            feature_schema=("x",),
        ).train(X_by_call, Y, n_series=1, max_workers=1)
        parallel = CanonicalMarginalQuantileTrainer(
            config,
            estimator_factory_for_level=factory_for_level,
            capabilities=CAPABILITIES,
            feature_schema=("x",),
        ).train(X_by_call, Y, n_series=1, max_workers=3)

        self.assertEqual(sequential.levels, parallel.levels)
        X_predict = np.array([[1.0]])
        times = pd.date_range("2026-09-01", periods=2, freq="1h")
        provider = lambda call_index, *_: (X_predict, X_predict)[call_index]
        seq_dist = CanonicalMarginalQuantileForecaster(
            config, sequential
        ).predict(
            X_predict,
            series_ids=("__local__",),
            forecast_times=times,
            feature_provider=provider,
        )
        par_dist = CanonicalMarginalQuantileForecaster(
            config, parallel
        ).predict(
            X_predict,
            series_ids=("__local__",),
            forecast_times=times,
            feature_provider=provider,
        )
        np.testing.assert_allclose(
            seq_dist.quantiles.values, par_dist.quantiles.values
        )

    def test_invalid_max_workers_raises(self):
        trainer = CanonicalMarginalQuantileTrainer(
            _config(),
            estimator_factory_for_level=lambda level: lambda: MeanQuantileRegressor(level),
            capabilities=CAPABILITIES,
            feature_schema=("x",),
        )
        X_by_call, Y = _toy_arrays()
        with self.assertRaises(ValueError):
            trainer.train(X_by_call, Y, n_series=1, max_workers=0)


class XgbNativeMultiQuantileTest(unittest.TestCase):
    def setUp(self):
        if not supports_native_multi_quantile("xgb"):
            self.skipTest("xgboost >= 2.0 not available")

    def test_capability_gate(self):
        self.assertTrue(supports_native_multi_quantile("xgboost"))
        self.assertFalse(supports_native_multi_quantile("lgb"))
        self.assertFalse(supports_native_multi_quantile("ridge"))

    def test_pool_trains_once_per_position_and_slices(self):
        levels = (0.1, 0.5, 0.9)
        pool = SharedMultiQuantilePool(
            "xgb",
            {"n_estimators": 20, "max_depth": 3},
            levels,
            ("x",),
        )
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 1))
        y = (X[:, 0] * 3.0 + rng.normal(0, 1.0, 200)).reshape(-1)

        # level 0 训练位置 0；level 1/2 同位置 fit 必须幂等
        est_0 = pool.factory_for_level(0)()
        est_0.fit(X, y)
        self.assertEqual(len(pool._fitted), 1)
        est_1 = pool.factory_for_level(1)()
        est_1.fit(X, y)
        self.assertEqual(len(pool._fitted), 1)

        pred_0 = est_0.predict(X)
        pred_2 = pool.factory_for_level(2)().predict(X)
        self.assertEqual(pred_0.shape, (200,))
        # 同一 booster 的不同列：大体单调（允许少量交叉）
        crossings = np.mean(pred_0 > pred_2)
        self.assertLess(crossings, 0.05)
        # 中位数拟合合理
        median_pred = est_1.predict(X)
        self.assertLess(np.mean(np.abs(median_pred - y)), 1.5)

    def test_full_chain_through_canonical_trainer(self):
        config = _config(model_type="xgb")
        X_by_call, Y = _toy_arrays()
        levels = (0.1, 0.5, 0.9)
        pool = SharedMultiQuantilePool(
            "xgb",
            {"n_estimators": 20, "max_depth": 3},
            levels,
            ("x",),
        )
        trainer = CanonicalMarginalQuantileTrainer(
            config,
            estimator_factory_for_level=lambda level: pool.factory_for_level(
                levels.index(level)
            ),
            capabilities=CAPABILITIES,
            feature_schema=("x",),
        )
        artifact = trainer.train(X_by_call, Y, n_series=1, max_workers=1)
        # 两个 horizon 各一个子模型位置，只训练一次
        self.assertEqual(len(pool._fitted), 2)

        forecaster = CanonicalMarginalQuantileForecaster(config, artifact)
        X_predict = np.array([[3.0]])
        distribution = forecaster.predict(
            X_predict,
            series_ids=("__local__",),
            forecast_times=pd.date_range("2026-09-01", periods=2, freq="1h"),
            feature_provider=lambda call_index, *_: (X_predict, X_predict)[call_index],
        )
        self.assertEqual(distribution.shape, (1, 2, 1, 3))
        self.assertTrue(np.isfinite(distribution.quantiles.values).all())

    def test_artifact_pickle_roundtrip_shares_booster(self):
        config = _config(model_type="xgb")
        X_by_call, Y = _toy_arrays()
        levels = (0.1, 0.5, 0.9)
        pool = SharedMultiQuantilePool(
            "xgb",
            {"n_estimators": 10, "max_depth": 2},
            levels,
            ("x",),
        )
        artifact = CanonicalMarginalQuantileTrainer(
            config,
            estimator_factory_for_level=lambda level: pool.factory_for_level(
                levels.index(level)
            ),
            capabilities=CAPABILITIES,
            feature_schema=("x",),
        ).train(X_by_call, Y, n_series=1, max_workers=1)
        blob = pickle.dumps(artifact)
        restored = pickle.loads(blob)
        forecaster = CanonicalMarginalQuantileForecaster(config, restored)
        X_predict = np.array([[3.0]])
        distribution = forecaster.predict(
            X_predict,
            series_ids=("__local__",),
            forecast_times=pd.date_range("2026-09-01", periods=2, freq="1h"),
            feature_provider=lambda call_index, *_: (X_predict, X_predict)[call_index],
        )
        self.assertTrue(np.isfinite(distribution.quantiles.values).all())

    def test_predict_before_fit_raises(self):
        pool = SharedMultiQuantilePool("xgb", {}, (0.5,), ("x",))
        view = pool.factory_for_level(0)()
        with self.assertRaises(ValueError):
            view.predict(np.zeros((2, 1)))

    def test_unsupported_model_type_raises(self):
        with self.assertRaises(ValueError):
            SharedMultiQuantilePool("lgb", {}, (0.5,), ("x",))


if __name__ == "__main__":
    unittest.main()
