# -*- coding: utf-8 -*-

from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from models.ModelFactory import ModelFactory
from models.ModelTraining import DirectMultiOutputRegressor, Trainer


class QuantileMultiOutputLinearTest(unittest.TestCase):
    def test_qr_default_and_tuning_grid_exclude_unregularized_solution(self):
        self.assertEqual(ModelFactory.get_default_model_params("qr")["alpha"], 1e-3)
        trainer = Trainer(self._calendar_month_args(), log_prefix="[test]")
        alpha_grid = trainer._get_tuning_param_grid("qr")["alpha"]
        self.assertNotIn(0.0, alpha_grid)
        self.assertIn(1e-3, alpha_grid)

    @staticmethod
    def _calendar_month_args() -> Any:
        return SimpleNamespace(
            model_type="qr",
            model_params={},
            model_thread_count=1,
            multi_output_n_jobs=1,
            pred_method="usmd",
            horizon=31,
            target_feature="y",
            lags=[1, 7, 28],
            block_size=0,
            direct_strategy="multioutput",
            align_direct_features_to_target=False,
            use_horizon_exogenous_for_direct=False,
            endogenous_backfill_strategy="persistence",
            blend_weight_strategy="fixed",
            blend_weights=[0.5, 0.5],
            enable_feature_cache=False,
            enable_global_training=False,
            enable_lags_features=True,
        )

    def test_calendar_month_fold_uses_fold_horizon_for_target_contract(self):
        X_train = pd.DataFrame({"x": [1.0, 2.0]})
        Y_train = pd.DataFrame(
            {
                f"y_shift_{step}": [float(step), float(step + 1)]
                for step in range(1, 31)
            }
        )

        # 旧行为使用全局 horizon=31，30 天月份的目标宽度会被拒绝。
        global_horizon_trainer = Trainer(self._calendar_month_args(), log_prefix="[test]")
        with self.assertRaisesRegex(ValueError, "training targets do not match"):
            global_horizon_trainer.train(X_train, Y_train, None, None, [])

        # calendar_month 滑窗显式传入 fold horizon=30，TargetPlan 与特征工程一致。
        fold_horizon_trainer = Trainer(self._calendar_month_args(), log_prefix="[test]")
        fold_horizon_trainer.set_train_horizon(30)
        sentinel = object()
        expected = (sentinel, None, None, None)
        with (
            patch.object(fold_horizon_trainer, "_should_use_fourmethods_baseline_training", return_value=True),
            patch.object(fold_horizon_trainer, "_train_fourmethods_baseline", return_value=expected),
        ):
            result = fold_horizon_trainer.train(X_train, Y_train, None, None, [])
        self.assertEqual(result, expected)

    def test_qr_multioutput_quantile_handles_lag_nan_through_model_wrapper(self):
        args: Any = SimpleNamespace(
                model_type="qr",
                model_params={},
                model_thread_count=1,
                multi_output_n_jobs=1,
            )
        trainer = Trainer(
            args,
            log_prefix="[test]",
        )
        X_train = pd.DataFrame(
            {
                "y_lag_1": [np.nan, 1.0, 2.0, 3.0, 4.0],
                "dt_day": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        Y_train = pd.DataFrame(
            {
                "y_shift_1": [2.0, 3.0, 4.0, 5.0, 6.0],
                "y_shift_2": [3.0, 4.0, 5.0, 6.0, 7.0],
            }
        )

        quantile, model = trainer._train_quantile_single_model(
            quantile=0.5,
            model_type="qr",
            X_train_df_processed=X_train,
            Y_train_df_processed=Y_train,
            lgbm_categorical=[],
        )
        prediction = model.predict(
            pd.DataFrame({"y_lag_1": [5.0], "dt_day": [6.0]})
        )

        self.assertEqual(quantile, 0.5)
        self.assertIsInstance(model, DirectMultiOutputRegressor)
        self.assertEqual(prediction.shape, (1, 2))
        self.assertTrue(np.isfinite(prediction).all())

    def test_multioutput_quantile_forwards_same_sample_weight_to_every_output(self):
        recorded_weights = []

        class RecordingWrapper:
            def fit(self, X, y, sample_weight=None, **kwargs):
                recorded_weights.append(
                    None if sample_weight is None else np.asarray(sample_weight).copy()
                )
                return self

            def predict(self, X):
                return np.zeros(len(X), dtype=float)

        args: Any = SimpleNamespace(
            model_type="qr",
            model_params={},
            model_thread_count=1,
            multi_output_n_jobs=1,
            multi_output_strategy="multioutput",
        )
        trainer = Trainer(args, log_prefix="[test]")
        trainer.sample_weight = np.array([0.5, 1.0, 1.5])
        trainer._create_model_instance = lambda *args, **kwargs: RecordingWrapper()
        X_train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        Y_train = pd.DataFrame(
            {
                "y_shift_1": [2.0, 3.0, 4.0],
                "y_shift_2": [3.0, 4.0, 5.0],
            }
        )

        trainer._train_quantile_single_model(
            quantile=0.5,
            model_type="qr",
            X_train_df_processed=X_train,
            Y_train_df_processed=Y_train,
            lgbm_categorical=[],
        )

        self.assertEqual(len(recorded_weights), 2)
        for weight in recorded_weights:
            np.testing.assert_array_equal(weight, trainer.sample_weight)


if __name__ == "__main__":
    unittest.main()
