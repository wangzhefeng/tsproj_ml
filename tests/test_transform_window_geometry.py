"""纯数据窗口检查，不拟合 scaler、分解或模型。"""
import unittest
import numpy as np
import pandas as pd
from forecasting_core.tensors import PointForecastTensor
from model_forecasting.transform_windows import select_transform_history


class TransformWindowGeometryTest(unittest.TestCase):
    def test_unique_label_times_and_explicit_context(self):
        times = pd.date_range("2026-01-01", periods=20, freq="1h")
        history = PointForecastTensor(values=np.arange(20, dtype=float).reshape(1, 20, 1), series_ids=("A",), forecast_times=times, targets=("load",))
        origins = (times[10], times[11], times[11])
        context, labels, audit = select_transform_history(history, origins, horizon=3, freq="1h")
        self.assertEqual(list(labels), list(times[11:15]))
        self.assertEqual(list(context.forecast_times), list(labels))
        self.assertEqual(audit["scaler_unique_label_count"], 4)
        longer, same_labels, _ = select_transform_history(history, origins, horizon=3, freq="1h", decomposition_history_steps=8)
        self.assertEqual(list(longer.forecast_times), list(times[7:15]))
        self.assertEqual(list(same_labels), list(labels))

    def test_explicit_context_cannot_shorten_training_labels(self):
        times = pd.date_range("2026-01-01", periods=20, freq="1h")
        history = PointForecastTensor(values=np.ones((1, 20, 1)), series_ids=("A",), forecast_times=times, targets=("load",))
        with self.assertRaises(ValueError):
            select_transform_history(history, (times[10], times[11]), horizon=3, freq="1h", decomposition_history_steps=2)


if __name__ == "__main__":
    unittest.main()
