# -*- coding: utf-8 -*-
"""CQR as-of 校准追踪器测试（2026-09-01 激活）。

钉住语义：
1. apply-before-collect：第 k 折只消费严格更早折的记录；
2. 双门槛（min_windows/min_scores）不足时诚实降级，不产出 pi 列；
3. pooled 分组：同 target_time 的多 (series, target) 记录都计入；
4. final 修正量消费全部合格历史折。
"""

import unittest

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from forecasting_core.probabilistic_spec import probabilistic_spec_from_mapping
from probabilistic.calibration import ConformalCalibrationTracker


def _spec(min_windows=2, min_scores=2, delay=0):
    return probabilistic_spec_from_mapping(
        {
            "mode": "quantile",
            "quantiles": [0.1, 0.5, 0.9],
            "point_quantile": 0.5,
            "intervals": [
                {"name": "q10_q90", "lower_quantile": 0.1, "upper_quantile": 0.9}
            ],
            "calibration": {
                "method": "cqr",
                "interval": "q10_q90",
                "target_coverage": 0.8,
                "calibration_windows": 5,
                "min_windows": min_windows,
                "min_scores": min_scores,
                "label_availability_delay_steps": delay,
            },
        }
    )


def _fold_frame(day: int, *, bias: float = 0.0, targets=("load",)) -> pd.DataFrame:
    """构造一折 long 帧：区间整体平移 bias（bias>10 时 actual 落到 q10 之下，
    score = bias - 10 > 0，用于模拟模型区间偏低的欠覆盖情形）。"""
    times = pd.date_range(f"2026-01-{day:02d}", periods=2, freq="1h")
    rows = []
    for target in targets:
        for time in times:
            actual = 100.0
            rows.append(
                {
                    "series_id": "s0",
                    "time": time,
                    "target": target,
                    "actual_value": actual,
                    "predict_value": actual,
                    "predict_q10": actual - 10.0 + bias,
                    "predict_q50": actual,
                    "predict_q90": actual + 10.0 + bias,
                    "window": day,
                    "plot_valid": True,
                }
            )
    return pd.DataFrame(rows)


class TrackerGatingTest(unittest.TestCase):
    def test_inverted_interval_rejected_before_collecting_any_records(self):
        tracker = ConformalCalibrationTracker(_spec(), freq_offset=to_offset("1h"))
        frame = _fold_frame(1)
        frame.loc[1, "predict_q10"] = 120.0
        with self.assertRaisesRegex(ValueError, "q_low <= q_high"):
            tracker.collect_from_frame(
                frame, forecast_origin=pd.Timestamp("2026-01-01"), window=1
            )
        _, audit = tracker.apply_to_frame(
            _fold_frame(2), forecast_origin=pd.Timestamp("2026-01-02")
        )
        self.assertEqual(audit["selected_scores"], 0)

    def test_apply_before_collect_and_double_gate(self):
        tracker = ConformalCalibrationTracker(
            _spec(min_windows=2, min_scores=2), freq_offset=to_offset("1h")
        )
        # 第 1 折：池为空 → insufficient_windows，无 pi 列
        frame1, audit1 = tracker.apply_to_frame(
            _fold_frame(1), forecast_origin=pd.Timestamp("2026-01-01")
        )
        self.assertEqual(audit1["status"], "insufficient_windows")
        self.assertNotIn("predict_pi80_lower", frame1.columns)
        tracker.collect_from_frame(
            frame1, forecast_origin=pd.Timestamp("2026-01-01"), window=1
        )
        # 第 2 折：只有 1 个历史窗 → 仍不足
        frame2, audit2 = tracker.apply_to_frame(
            _fold_frame(2), forecast_origin=pd.Timestamp("2026-01-02")
        )
        self.assertEqual(audit2["status"], "insufficient_windows")
        tracker.collect_from_frame(
            frame2, forecast_origin=pd.Timestamp("2026-01-02"), window=2
        )
        # 第 3 折：2 个历史窗 × 每窗 2 score → applied
        frame3, audit3 = tracker.apply_to_frame(
            _fold_frame(3), forecast_origin=pd.Timestamp("2026-01-03")
        )
        self.assertEqual(audit3["status"], "applied")
        self.assertIn("predict_pi80_lower", frame3.columns)
        self.assertIn("predict_pi80_upper", frame3.columns)
        # 校准只加宽不缩窄（allow_interval_shrink=False）：bias=0 时 actual
        # 居中，历史 score 全为 -10 → correction 钳到 0
        self.assertEqual(audit3["correction"], 0.0)
        np.testing.assert_allclose(
            frame3["predict_pi80_lower"], frame3["predict_q10"]
        )

    def test_positive_correction_when_model_overconfident(self):
        # bias=12：actual 落在 q10 之下 2 个单位（score=2），区间向下偏
        tracker = ConformalCalibrationTracker(
            _spec(min_windows=1, min_scores=2), freq_offset=to_offset("1h")
        )
        frame1 = _fold_frame(1, bias=12.0)
        tracker.collect_from_frame(
            frame1, forecast_origin=pd.Timestamp("2026-01-01"), window=1
        )
        frame2, audit2 = tracker.apply_to_frame(
            _fold_frame(2, bias=12.0),
            forecast_origin=pd.Timestamp("2026-01-02"),
        )
        self.assertEqual(audit2["status"], "applied")
        # score = max(q10-y, y-q90) = max(2, -22) = 2
        self.assertAlmostEqual(audit2["correction"], 2.0)
        np.testing.assert_allclose(
            frame2["predict_pi80_lower"], frame2["predict_q10"] - 2.0
        )

    def test_pooled_multi_target_all_counted(self):
        tracker = ConformalCalibrationTracker(
            _spec(min_windows=1, min_scores=4), freq_offset=to_offset("1h")
        )
        frame = _fold_frame(1, targets=("load", "power"))
        count = tracker.collect_from_frame(
            frame, forecast_origin=pd.Timestamp("2026-01-01"), window=1
        )
        self.assertEqual(count, 4)  # 2 targets × 2 times，不按 target_time 塌缩

    def test_nan_actual_rows_skipped(self):
        frame = _fold_frame(1)
        frame.loc[0, "actual_value"] = np.nan
        tracker = ConformalCalibrationTracker(
            _spec(), freq_offset=to_offset("1h")
        )
        count = tracker.collect_from_frame(
            frame, forecast_origin=pd.Timestamp("2026-01-01"), window=1
        )
        self.assertEqual(count, 1)

    def test_final_correction_uses_all_eligible_folds(self):
        tracker = ConformalCalibrationTracker(
            _spec(min_windows=2, min_scores=2), freq_offset=to_offset("1h")
        )
        for day in (1, 2):
            tracker.collect_from_frame(
                _fold_frame(day, bias=12.0),
                forecast_origin=pd.Timestamp(f"2026-01-{day:02d}"),
                window=day,
            )
        result, audit = tracker.final_correction(pd.Timestamp("2026-01-10"))
        self.assertEqual(result.status, "applied")
        self.assertEqual(audit["selected_windows"], 2)
        self.assertAlmostEqual(audit["correction"], 2.0)
        lower, upper = tracker.correction_bounds(
            np.zeros((1, 2, 1, 3)), (0.1, 0.5, 0.9), audit["correction"]
        )
        np.testing.assert_allclose(lower, np.full((1, 2, 1), -2.0))
        np.testing.assert_allclose(upper, np.full((1, 2, 1), 2.0))

    def test_requires_calibration_spec(self):
        point_spec = probabilistic_spec_from_mapping({"mode": "point"})
        with self.assertRaises(ValueError):
            ConformalCalibrationTracker(point_spec, freq_offset=to_offset("1h"))


class DeploymentPiColumnWiringTest(unittest.TestCase):
    """部署期 pi 列接线（2026-09-02）：bundle metadata 的 prediction_intervals
    必须经 distribution_to_long 落成 prediction.csv 列，不再是死写。"""

    def test_distribution_to_long_emits_pi_columns_from_metadata(self):
        from forecasting_core.artifacts import MarginalForecastDistribution
        from forecasting_core.tensors import (
            MarginalQuantileForecastTensor,
            PointForecastTensor,
        )
        from model_forecasting.results import distribution_to_long

        times = pd.date_range("2026-01-01", periods=2, freq="1h")
        point = PointForecastTensor(
            values=np.full((1, 2, 1), 100.0),
            series_ids=("s1",),
            forecast_times=times,
            targets=("load",),
        )
        quantiles = MarginalQuantileForecastTensor(
            values=np.stack(
                [
                    np.full((1, 2, 1), 90.0),
                    np.full((1, 2, 1), 100.0),
                    np.full((1, 2, 1), 110.0),
                ],
                axis=-1,
            ),
            levels=(0.1, 0.5, 0.9),
            point_level=0.5,
            series_ids=("s1",),
            forecast_times=times,
            targets=("load",),
        )
        distribution = MarginalForecastDistribution(
            point=point,
            quantiles=quantiles,
            metadata={
                "prediction_intervals": {
                    "q10_q90": {
                        "method": "cqr",
                        "target_coverage": 0.8,
                        "correction": 2.0,
                        "lower": np.full((1, 2, 1), 88.0),
                        "upper": np.full((1, 2, 1), 112.0),
                        "columns": ["predict_pi80_lower", "predict_pi80_upper"],
                    }
                }
            },
        )
        frame = distribution_to_long(distribution)
        self.assertIn("predict_pi80_lower", frame.columns)
        self.assertIn("predict_pi80_upper", frame.columns)
        self.assertEqual(frame["predict_pi80_lower"].tolist(), [88.0, 88.0])
        self.assertEqual(frame["predict_pi80_upper"].tolist(), [112.0, 112.0])

    def test_distribution_to_long_rejects_misaligned_pi_arrays(self):
        from forecasting_core.artifacts import MarginalForecastDistribution
        from forecasting_core.tensors import (
            MarginalQuantileForecastTensor,
            PointForecastTensor,
        )
        from model_forecasting.results import distribution_to_long

        times = pd.date_range("2026-01-01", periods=2, freq="1h")
        point = PointForecastTensor(
            values=np.full((1, 2, 1), 100.0),
            series_ids=("s1",),
            forecast_times=times,
            targets=("load",),
        )
        quantiles = MarginalQuantileForecastTensor(
            values=np.full((1, 2, 1, 3), 100.0),
            levels=(0.1, 0.5, 0.9),
            point_level=0.5,
            series_ids=("s1",),
            forecast_times=times,
            targets=("load",),
        )
        distribution = MarginalForecastDistribution(
            point=point,
            quantiles=quantiles,
            metadata={
                "prediction_intervals": {
                    "q10_q90": {
                        "lower": np.zeros(3),  # 长度不匹配：frame 只有 2 行
                        "upper": np.zeros(3),
                        "columns": ["predict_pi80_lower", "predict_pi80_upper"],
                    }
                }
            },
        )
        with self.assertRaises(ValueError):
            distribution_to_long(distribution)


if __name__ == "__main__":
    unittest.main()
