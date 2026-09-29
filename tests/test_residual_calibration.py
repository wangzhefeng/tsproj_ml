"""点预测残差区间：组隔离、as-of、有限样本秩和返回合同。"""
import unittest

import numpy as np
import pandas as pd

from forecasting_core.point_intervals import PointIntervalForecast, ResidualCalibrationSpec
from forecasting_core.probabilistic_spec import probabilistic_spec_from_mapping
from forecasting_core.tensors import PointForecastTensor
from probabilistic.residual import ResidualCalibrationTracker, apply_residual_state
from model_evaluation.point_intervals import evaluate_point_intervals


class ResidualCalibrationTest(unittest.TestCase):
    def point(self, origin, values=None):
        return PointForecastTensor(
            values=np.asarray(values if values is not None else [[[10.0], [10.0]], [[100.0], [100.0]]]),
            series_ids=("A", "B"), targets=("load",),
            forecast_times=pd.date_range(pd.Timestamp(origin) + pd.Timedelta(hours=1), periods=2, freq="h"),
        )

    def test_point_spec_is_distinct_from_quantile_calibration(self):
        spec = probabilistic_spec_from_mapping({"mode": "point", "calibration": {
            "method": "absolute_residual", "target_coverage": 0.5,
            "calibration_windows": 4, "min_windows": 2, "min_scores": 2,
        }})
        self.assertIsInstance(spec.calibration, ResidualCalibrationSpec)
        self.assertEqual(spec.quantiles, ())
        self.assertIsNone(spec.calibration_interval)
        with self.assertRaises(ValueError):
            probabilistic_spec_from_mapping({"mode": "point", "calibration": {
                "method": "cqr", "interval": "fake", "target_coverage": 0.5,
            }})

    def test_grouped_radii_and_saved_state_match_without_cross_group_pooling(self):
        spec = ResidualCalibrationSpec(target_coverage=0.5, calibration_windows=4, min_windows=2, min_scores=2)
        tracker = ResidualCalibrationTracker(spec, freq_offset=pd.offsets.Hour())
        for window, hour in enumerate((0, 3), start=1):
            origin = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=hour)
            prediction = self.point(origin)
            errors = np.asarray([[[1.0], [4.0]], [[10.0], [40.0]]]) * window
            actual = PointForecastTensor(values=prediction.values + errors, series_ids=prediction.series_ids,
                                         forecast_times=prediction.forecast_times, targets=prediction.targets)
            tracker.collect(actual, prediction, forecast_origin=origin, window=window)
        origin = pd.Timestamp("2026-01-01T06:00:00")
        prediction = self.point(origin)
        state = tracker.state(prediction, forecast_origin=origin)
        restored = apply_residual_state(prediction, state)
        self.assertIsInstance(restored, PointIntervalForecast)
        # n=2, coverage=.5 -> 第2顺序统计量；每序列、目标、horizon 独立。
        np.testing.assert_array_equal(restored.radius, [[[2.0], [8.0]], [[20.0], [80.0]]])
        np.testing.assert_array_equal(restored.lower, prediction.values - restored.radius)
        self.assertTrue(restored.available.all())
        self.assertEqual(restored.point, prediction)

    def test_current_and_delayed_labels_cannot_calibrate_and_rank_is_not_clipped(self):
        spec = ResidualCalibrationSpec(target_coverage=0.9, calibration_windows=4, min_windows=1, min_scores=1,
                                       label_availability_delay_steps=2)
        tracker = ResidualCalibrationTracker(spec, freq_offset=pd.offsets.Hour())
        origin = pd.Timestamp("2026-01-01")
        prediction = self.point(origin)
        tracker.collect(prediction, prediction, forecast_origin=origin, window=1)
        for hour in (0, 2):
            now = origin + pd.Timedelta(hours=hour)
            result = tracker.apply(self.point(now), forecast_origin=now)
            self.assertFalse(result.available.any())
        later = origin + pd.Timedelta(hours=6)
        result = tracker.apply(self.point(later), forecast_origin=later)
        self.assertFalse(result.available.any())
        self.assertTrue(all(status == "insufficient_rank" for status in result.statuses))
        self.assertTrue(np.isnan(result.lower).all())

    def test_partial_horizon_availability_and_target_isolation(self):
        spec = ResidualCalibrationSpec(target_coverage=0.5, calibration_windows=2, min_windows=1,
                                       min_scores=1, label_availability_delay_steps=2)
        tracker = ResidualCalibrationTracker(spec, freq_offset=pd.offsets.Hour())
        origin = pd.Timestamp("2026-01-01")
        prediction = PointForecastTensor(values=np.ones((1, 2, 2)), series_ids=("A",),
            forecast_times=pd.date_range(origin + pd.Timedelta(hours=1), periods=2, freq="h"), targets=("x", "y"))
        actual = PointForecastTensor(values=prediction.values + np.asarray([[[2.0, 20.0], [4.0, 40.0]]]),
            series_ids=prediction.series_ids, forecast_times=prediction.forecast_times, targets=prediction.targets)
        tracker.collect(actual, prediction, forecast_origin=origin, window=1)
        now = origin + pd.Timedelta(hours=3)
        future = PointForecastTensor(values=prediction.values, series_ids=prediction.series_ids,
            forecast_times=pd.date_range(now + pd.Timedelta(hours=1), periods=2, freq="h"), targets=prediction.targets)
        result = tracker.apply(future, forecast_origin=now)
        np.testing.assert_array_equal(result.available, [[[True, True], [False, False]]])
        np.testing.assert_array_equal(result.radius[:, 0, :], [[2.0, 20.0]])
        state = tracker.state(future, forecast_origin=now)
        with self.assertRaisesRegex(ValueError, "at/before"):
            apply_residual_state(prediction, state)
        scores = evaluate_point_intervals(future, result, window=2)
        self.assertTrue((scores.loc[scores["scope"] == "target", "n_points"] == 1).all())
        self.assertTrue((scores.loc[(scores["target"] == "x") & (scores["metric"] == "interval_width") & (scores["scope"] == "target"), "value"] == 4.0).all())

    def test_bad_radius_or_group_axes_are_rejected(self):
        origin = pd.Timestamp("2026-01-01")
        point = self.point("2026-01-01")
        tracker = ResidualCalibrationTracker(ResidualCalibrationSpec(target_coverage=0.5), freq_offset=pd.offsets.Hour())
        state = tracker.state(point, forecast_origin=origin)
        # 首个预测时刻虽然在校准时刻之后，但隐含 forecast_origin 仍在之前。
        early = PointForecastTensor(values=point.values, series_ids=point.series_ids, targets=point.targets,
            forecast_times=point.forecast_times - pd.Timedelta(minutes=30))
        with self.assertRaisesRegex(ValueError, "origin|frequency"):
            apply_residual_state(early, state)
        point = self.point("2026-01-01")
        with self.assertRaises(ValueError):
            PointIntervalForecast(point, (-1.0,) * 4, 0.8, ("applied",) * 4)
        with self.assertRaises(ValueError):
            PointIntervalForecast(point, (None,) * 4, 0.8, ("applied",) * 4)


if __name__ == "__main__":
    unittest.main()
