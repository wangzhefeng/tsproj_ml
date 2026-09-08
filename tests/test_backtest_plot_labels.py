"""通用逐窗图/总图的Trues与Preds不得对调。"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from forecasting_core.tensors import PointForecastTensor
from model_evaluation.point import evaluate_point_forecasts
from model_testing.tensor_frames import backtest_tensors_to_long
from model_testing.reporting import write_backtest_results


class BacktestPlotLabelsTest(unittest.TestCase):
    def test_truth_and_prediction_labels_match_values(self):
        times = pd.date_range("2025-03-01", periods=3, freq="1D")
        actual = PointForecastTensor(np.array([1., 2., 3.]).reshape(1, 3, 1), ("__local__",), times, ("value",))
        prediction = PointForecastTensor(np.array([4., 5., 6.]).reshape(1, 3, 1), ("__local__",), times, ("value",))
        captured = {}

        def capture(figure, path, **kwargs):
            captured[Path(path).name] = {line.get_label(): line.get_ydata() for line in figure.axes[0].lines}

        with tempfile.TemporaryDirectory() as directory, patch.object(Figure, "savefig", new=capture):
            write_backtest_results(directory, backtest_tensors_to_long(actual, prediction, window=1),
                                   evaluate_point_forecasts(actual, prediction), aggregate_weighting={"value": 1.0})
        self.assertEqual(set(captured), {"window_01.png", "test_prediction.png"})
        for lines in captured.values():
            np.testing.assert_array_equal(lines["Trues"], [1., 2., 3.])
            np.testing.assert_array_equal(lines["Preds"], [4., 5., 6.])


if __name__ == "__main__":
    unittest.main()
