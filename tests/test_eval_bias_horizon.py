# -*- coding: utf-8 -*-
"""per-horizon 诊断与 bias 指标（2026-09-02）的定向测试。

覆盖：
- 点评估：Bias/Naive Bias 手算值、符号语义（正=高估）、加权 aggregate、
  per-horizon 行（scope=horizon/aggregate_horizon）、horizon 列 1-based、
  掩码切片到同一 horizon、掩码后 horizon 行 n_points 求和 = target 行 n_points；
- 概率评估：bias metric 行手算值、per-horizon 行、aggregate_horizon 池化
  与 aggregate 整体的一致性（未掩码时 pinball 逐 h 均值 = 整体均值）。
"""

import unittest

import numpy as np
import pandas as pd

from forecasting_core.tensors import (
    MarginalQuantileForecastTensor,
    PointForecastTensor,
)
from forecasting_core.artifacts import MarginalForecastDistribution
from model_evaluation.marginal import evaluate_marginal_distribution
from model_evaluation.point import evaluate_point_forecasts


TIMES = pd.date_range("2026-09-01", periods=4, freq="1h")


def _point_tensor(values_2d, targets=("load",)):
    values = np.asarray(values_2d, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    return PointForecastTensor(
        values=values[None, :, :],
        series_ids=("s0",),
        forecast_times=TIMES[: values.shape[0]],
        targets=targets,
    )


class PointBiasAndHorizonTest(unittest.TestCase):
    def test_bias_sign_and_value_match_manual_computation(self):
        actual = _point_tensor([10.0, 10.0, 10.0, 10.0])
        prediction = _point_tensor([12.0, 11.0, 9.0, 8.0])  # 误差 +2,+1,-1,-2

        report = evaluate_point_forecasts(actual, prediction)

        row = report[report["scope"] == "target"].iloc[0]
        self.assertAlmostEqual(row["Bias"], 0.0)  # (+2+1-1-2)/4
        self.assertAlmostEqual(row["MAE"], 1.5)
        # bias 与 MAE 解耦：系统高估时 bias > 0
        high = evaluate_point_forecasts(actual, _point_tensor([15.0] * 4))
        self.assertAlmostEqual(
            high[high["scope"] == "target"].iloc[0]["Bias"], 5.0
        )

    def test_aggregate_bias_is_target_weighted(self):
        # 双 target：load 系统高估 +2、power 系统低估 −2（各 8 点）
        values = np.arange(16.0).reshape(2, 4, 2)
        actual = PointForecastTensor(
            values=values,
            series_ids=("A", "B"),
            forecast_times=TIMES,
            targets=("load", "power"),
        )
        prediction = PointForecastTensor(
            values=values + np.array([2.0, -2.0])[None, None, :],
            series_ids=actual.series_ids,
            forecast_times=TIMES,
            targets=actual.targets,
        )

        equal = evaluate_point_forecasts(actual, prediction)
        self.assertAlmostEqual(
            equal[equal["scope"] == "aggregate"].iloc[0]["Bias"], 0.0
        )
        self.assertAlmostEqual(
            equal[
                (equal["scope"] == "target") & (equal["target"] == "load")
            ].iloc[0]["Bias"],
            2.0,
        )

        weighted = evaluate_point_forecasts(
            actual, prediction, aggregate_weighting={"load": 1.0, "power": 0.0}
        )
        self.assertAlmostEqual(
            weighted[weighted["scope"] == "aggregate"].iloc[0]["Bias"], 2.0
        )

    def test_per_horizon_rows_match_manual_per_step_values(self):
        actual = _point_tensor([10.0, 20.0, 30.0, 40.0])
        prediction = _point_tensor([11.0, 24.0, 27.0, 52.0])  # 误差 -1,-4,+3,-12

        report = evaluate_point_forecasts(actual, prediction, window=7)

        horizons = report[report["scope"] == "horizon"]
        self.assertEqual(horizons["horizon"].tolist(), [1, 2, 3, 4])
        # Bias = pred − actual：[+1, +4, −3, +12]
        np.testing.assert_allclose(
            horizons["Bias"].tolist(), [1.0, 4.0, -3.0, 12.0]
        )
        np.testing.assert_allclose(
            horizons["MAE"].tolist(), [1.0, 4.0, 3.0, 12.0]
        )
        self.assertTrue((horizons["window"] == 7).all())
        agg_h = report[report["scope"] == "aggregate_horizon"]
        self.assertEqual(agg_h["target"].unique().tolist(), ["__aggregate__"])
        np.testing.assert_allclose(agg_h["MAE"].tolist(), [1.0, 4.0, 3.0, 12.0])

    def test_per_horizon_mask_slices_and_point_counts_reconcile(self):
        actual_values = [10.0, 2.0, 20.0, 5.0]
        actual = _point_tensor(actual_values)
        prediction = _point_tensor([11.0, 4.0, 19.0, 6.0])
        eval_mask = {"mode": "absolute", "min_value": 4.0}

        report = evaluate_point_forecasts(actual, prediction, eval_mask=eval_mask)

        target_row = report[report["scope"] == "target"].iloc[0]
        horizons = report[report["scope"] == "horizon"]
        # 掩码保留 idx 0,2,3（值 10,20,5）→ n_points 逐 h = [1,0,1,1]
        self.assertEqual(horizons["n_points"].tolist(), [1, 0, 1, 1])
        self.assertEqual(int(target_row["n_points"]), 3)
        self.assertEqual(int(horizons["n_points"].sum()), int(target_row["n_points"]))
        h3 = horizons[horizons["horizon"] == 3].iloc[0]
        self.assertAlmostEqual(h3["MAE"], 1.0)
        self.assertTrue(
            horizons[horizons["n_points"] == 0]["MAE"].isna().all()
        )

    def test_naive_columns_carry_bias_and_horizon_rows(self):
        actual = _point_tensor([10.0, 20.0, 30.0, 40.0])
        prediction = _point_tensor([11.0, 22.0, 33.0, 44.0])
        naive = _point_tensor([9.0, 25.0, 27.0, 50.0])

        report = evaluate_point_forecasts(
            actual, prediction, seasonal_naive=naive
        )

        horizons = report[report["scope"] == "horizon"]
        # Naive Bias = naive − actual：[9−10, 25−20, 27−30, 50−40]
        np.testing.assert_allclose(
            horizons["Naive Bias"].tolist(), [-1.0, 5.0, -3.0, 10.0]
        )
        agg = report[report["scope"] == "aggregate"].iloc[0]
        self.assertAlmostEqual(agg["Naive Bias"], 2.75)

    def test_aggregate_horizon_equals_flattened_mean_without_mask(self):
        # 双 target 无掩码：aggregate_horizon MAE 必须等于该 h 全部 (N,K) 点的
        # 平铺均值（proper score 池化语义）
        values = np.arange(16.0).reshape(2, 4, 2)
        actual = PointForecastTensor(
            values=values, series_ids=("A", "B"), forecast_times=TIMES,
            targets=("load", "power"),
        )
        prediction = PointForecastTensor(
            values=values + np.array([[1.0, -1.0], [-2.0, 2.0], [3.0, 0.0], [0.5, 4.0]])[None],
            series_ids=actual.series_ids,
            forecast_times=TIMES,
            targets=actual.targets,
        )

        report = evaluate_point_forecasts(actual, prediction)

        for h in range(1, 5):
            agg_h = report[
                (report["scope"] == "aggregate_horizon")
                & (report["horizon"] == h)
            ].iloc[0]
            errors = np.abs(
                prediction.values[:, h - 1, :] - actual.values[:, h - 1, :]
            )
            self.assertAlmostEqual(agg_h["MAE"], float(np.mean(errors)))


class MarginalBiasAndHorizonTest(unittest.TestCase):
    def _distribution(self, y_true, offset=2.0):
        y = np.asarray(y_true, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        actual = PointForecastTensor(
            values=y[None, :, :],
            series_ids=("s0",),
            forecast_times=TIMES[: y.shape[0]],
            targets=("load",),
        )
        # q50 = y + offset → point 偏高 offset → bias = +offset；区间保持有序
        q_values = np.repeat(y[None, :, :, None], 3, axis=-1) + np.array(
            [0.0, offset, 2.0 * offset]
        )
        quantiles = MarginalQuantileForecastTensor(
            values=q_values,
            levels=(0.1, 0.5, 0.9),
            point_level=0.5,
            series_ids=actual.series_ids,
            forecast_times=actual.forecast_times,
            targets=actual.targets,
        )
        distribution = MarginalForecastDistribution(
            point=quantiles.point(), quantiles=quantiles, dependence_model=None
        )
        return actual, distribution

    def test_bias_row_is_point_minus_actual(self):
        actual, distribution = self._distribution([10.0, 20.0, 30.0, 40.0])

        report = evaluate_marginal_distribution(actual, distribution)

        target_rows = report[
            (report["scope"] == "target") & (report["metric"] == "bias")
        ]
        self.assertAlmostEqual(target_rows["value"].iloc[0], 2.0)  # point = y+2

    def test_per_horizon_bias_rows_and_pooling_consistency(self):
        y = np.array([10.0, 20.0, 30.0, 40.0])
        actual, distribution = self._distribution(y)

        report = evaluate_marginal_distribution(actual, distribution, window=5)

        h_bias = report[
            (report["scope"] == "horizon") & (report["metric"] == "bias")
        ]
        self.assertEqual(h_bias["horizon"].tolist(), [1, 2, 3, 4])
        np.testing.assert_allclose(h_bias["value"].tolist(), [2.0] * 4)
        self.assertTrue((h_bias["window"] == 5).all())
        # 未掩码：逐 h 池化 bias 均值 = aggregate 整体 bias
        agg_bias = report[
            (report["scope"] == "aggregate") & (report["metric"] == "bias")
        ]["value"].iloc[0]
        agg_h_bias = report[
            (report["scope"] == "aggregate_horizon") & (report["metric"] == "bias")
        ]
        np.testing.assert_allclose(agg_h_bias["value"].tolist(), [agg_bias] * 4)
        # 未掩码：逐 h pinball@q50 均值 = aggregate 整体 pinball@q50
        agg_pin = report[
            (report["scope"] == "aggregate")
            & (report["metric"] == "pinball")
            & (report["quantile"] == 0.5)
        ]["value"].iloc[0]
        agg_h_pin = report[
            (report["scope"] == "aggregate_horizon")
            & (report["metric"] == "pinball")
            & (report["quantile"] == 0.5)
        ]
        np.testing.assert_allclose(agg_h_pin["value"].tolist(), [agg_pin] * 4)

    def test_horizon_rows_absent_when_no_valid_points_pool(self):
        # 全 False 掩码：target/horizon 行 NaN、aggregate/aggregate_horizon 不产出
        actual, distribution = self._distribution([10.0, 20.0, 30.0, 40.0])

        report = evaluate_marginal_distribution(
            actual, distribution, valid_masks={"load": np.zeros(4, dtype=bool)}
        )

        scored = report[report["scope"].isin(["target", "horizon"])]
        self.assertTrue((scored["n_points"] == 0).all())
        self.assertTrue(scored["value"].isna().all())
        self.assertTrue(
            report[report["scope"].isin(["aggregate", "aggregate_horizon"])].empty
        )


if __name__ == "__main__":
    unittest.main()
