# -*- coding: utf-8 -*-
"""概率评估生产接线的定向测试（2026-08-30 项 1，方案 A）。

覆盖：
- evaluate_marginal_distribution 的 valid_masks（与点评估同一 eval_mask 口径）；
- central 区间行（interval_coverage/width/winkler/coverage_gap/calibration_error）
  与手工对账；
- window 列写入；无对称 quantile 对时不产出区间行；
- 全掩码 target → NaN 行且 n_points=0；
- aggregate 行跨 target 按有效点池化（proper score 语义）；
- 未传 valid_masks 时 mae/pinball 与历史行为逐值一致（零变化证据）。
"""

import unittest

import numpy as np
import pandas as pd

from forecasting_core.tensors import MarginalQuantileForecastTensor, PointForecastTensor
from model_evaluation.marginal import evaluate_marginal_distribution
from forecasting_core.artifacts import MarginalForecastDistribution


TIMES = pd.date_range("2026-01-01", periods=4, freq="1h")


def _distribution(
    y_true: np.ndarray,
    *,
    offset_low: float = -5.0,
    offset_high: float = 5.0,
    levels: tuple = (0.1, 0.5, 0.9),
    targets: tuple = ("load",),
) -> tuple[PointForecastTensor, MarginalForecastDistribution]:
    """构造 N=1 的合成分布：q50 = y_true，q10/q90 按 offset 平移。"""
    y = np.asarray(y_true, dtype=float)
    if y.ndim == 1:
        y = y[:, None]  # (H,) -> (H, K=1)
    horizon, n_targets = y.shape
    actual = PointForecastTensor(
        values=y[None, :, :],
        series_ids=("s",),
        forecast_times=TIMES[:horizon],
        targets=targets,
    )
    level_offsets = {0.1: offset_low, 0.5: 0.0, 0.9: offset_high}
    q_values = np.stack(
        [
            y[None, :, :] + level_offsets.get(level, 0.0)
            for level in levels
        ],
        axis=-1,
    )  # (1, H, K, Q)
    quantiles = MarginalQuantileForecastTensor(
        values=q_values,
        levels=tuple(float(level) for level in levels),
        point_level=0.5,
        series_ids=("s",),
        forecast_times=TIMES[:horizon],
        targets=targets,
    )
    distribution = MarginalForecastDistribution(
        point=quantiles.point(),
        quantiles=quantiles,
        dependence_model=None,
    )
    return actual, distribution


class EvaluateMarginalDistributionWiringTest(unittest.TestCase):
    def test_unmasked_metrics_match_legacy_manual_computation(self):
        y = np.array([10.0, 20.0, 30.0, 40.0])
        actual, distribution = _distribution(y)

        report = evaluate_marginal_distribution(actual, distribution, window=3)

        self.assertTrue((report["window"] == 3).all())
        target_rows = report[report["scope"] == "target"]
        mae = target_rows[target_rows["metric"] == "mae"]["value"].iloc[0]
        self.assertEqual(mae, 0.0)
        # q10 = y-5：error = +5，pinball@0.1 = 0.1*5 = 0.5
        pin10 = target_rows[
            (target_rows["metric"] == "pinball") & (target_rows["quantile"] == 0.1)
        ]["value"].iloc[0]
        self.assertAlmostEqual(pin10, 0.5)
        # q90 = y+5：error = -5，pinball@0.9 = (0.9-1)*(-5) = 0.5
        pin90 = target_rows[
            (target_rows["metric"] == "pinball") & (target_rows["quantile"] == 0.9)
        ]["value"].iloc[0]
        self.assertAlmostEqual(pin90, 0.5)
        # central80 = [y-5, y+5]：coverage=1.0、width=10、无惩罚 winkler=10、gap=+0.2
        coverage = target_rows[target_rows["metric"] == "interval_coverage"].iloc[0]
        self.assertEqual(coverage["interval_name"], "central80")
        self.assertEqual(coverage["value"], 1.0)
        self.assertEqual(coverage["target_coverage"], 0.8)
        width = target_rows[target_rows["metric"] == "interval_width"]["value"].iloc[0]
        self.assertEqual(width, 10.0)
        winkler = target_rows[target_rows["metric"] == "interval_winkler"]["value"].iloc[0]
        self.assertEqual(winkler, 10.0)
        gap = target_rows[target_rows["metric"] == "coverage_gap"]["value"].iloc[0]
        self.assertAlmostEqual(gap, 0.2)

    def test_valid_masks_apply_business_caliber_to_probabilistic_metrics(self):
        y = np.array([10.0, 20.0, 30.0, 40.0])
        actual, distribution = _distribution(y)
        # 掩掉前两点：与点评估同一 eval_mask 口径作用于 pinball/区间
        mask = np.array([False, False, True, True])

        report = evaluate_marginal_distribution(
            actual, distribution, valid_masks={"load": mask}
        )

        target_rows = report[report["scope"] == "target"]
        self.assertTrue((target_rows["n_points"] == 2).all())
        # 掩码后 q50 仍等于 actual → mae=0；pinball@0.1 恒为 0.5（每点 loss 相同）
        self.assertEqual(target_rows[target_rows["metric"] == "mae"]["value"].iloc[0], 0.0)
        pin10 = target_rows[
            (target_rows["metric"] == "pinball") & (target_rows["quantile"] == 0.1)
        ]["value"].iloc[0]
        self.assertAlmostEqual(pin10, 0.5)

    def test_fully_masked_target_yields_nan_rows_with_zero_points(self):
        y = np.array([10.0, 20.0, 30.0, 40.0])
        actual, distribution = _distribution(y)
        mask = np.zeros(4, dtype=bool)

        report = evaluate_marginal_distribution(
            actual, distribution, valid_masks={"load": mask}
        )

        target_rows = report[report["scope"] == "target"]
        self.assertTrue((target_rows["n_points"] == 0).all())
        self.assertTrue(target_rows["value"].isna().all())
        # 全掩码时不产出 aggregate 行（无有效点可池化）
        self.assertTrue((report["scope"] == "target").all())

    def test_mask_shape_mismatch_raises(self):
        y = np.array([10.0, 20.0, 30.0, 40.0])
        actual, distribution = _distribution(y)
        with self.assertRaisesRegex(ValueError, "valid mask"):
            evaluate_marginal_distribution(
                actual, distribution, valid_masks={"load": np.ones(3, dtype=bool)}
            )

    def test_no_symmetric_pair_no_interval_rows(self):
        y = np.array([10.0, 20.0, 30.0, 40.0])
        actual, distribution = _distribution(y, levels=(0.5,))

        report = evaluate_marginal_distribution(actual, distribution)

        self.assertEqual(set(report["metric"].unique()), {"mae", "pinball"})

    def test_aggregate_pools_valid_points_across_targets(self):
        # K=2：target A 两点误差 0（全有效），target B 一点误差 10（掩码后 1 点）
        y = np.array([[10.0, 100.0], [20.0, 200.0]])  # (H=2, K=2)
        actual, distribution = _distribution(
            y, levels=(0.5,), targets=("a", "b")
        )
        # 让 point 偏离 target B：q50 对 target B 偏移 +10
        q_values = (actual.values + np.array([0.0, 10.0])[None, None, :])[:, :, :, None]
        quantiles = MarginalQuantileForecastTensor(
            values=q_values,
            levels=(0.5,),
            point_level=0.5,
            series_ids=("s",),
            forecast_times=TIMES[:2],
            targets=("a", "b"),
        )
        distribution = MarginalForecastDistribution(
            point=quantiles.point(), quantiles=quantiles, dependence_model=None
        )
        masks = {"a": np.array([True, True]), "b": np.array([True, False])}

        report = evaluate_marginal_distribution(
            actual, distribution, valid_masks=masks
        )

        aggregate_mae = report[
            (report["scope"] == "aggregate") & (report["metric"] == "mae")
        ].iloc[0]
        # 池化：A 两点误差 0 + B 一点误差 10 → (0+0+10)/3
        self.assertAlmostEqual(aggregate_mae["value"], 10.0 / 3.0)
        self.assertEqual(aggregate_mae["n_points"], 3)


if __name__ == "__main__":
    unittest.main()
