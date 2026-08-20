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


if __name__ == "__main__":
    unittest.main()
