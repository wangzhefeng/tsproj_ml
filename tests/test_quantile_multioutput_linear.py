# -*- coding: utf-8 -*-

from types import SimpleNamespace
from typing import Any
import unittest

import numpy as np
import pandas as pd

from models.ModelTraining import DirectMultiOutputRegressor, Trainer


class QuantileMultiOutputLinearTest(unittest.TestCase):
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
