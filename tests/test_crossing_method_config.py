# -*- coding: utf-8 -*-
"""crossing 配置消费测试（2026-09-01 裂缝修复）。

钉住三条契约：
1. ``repair_marginal_quantile_crossing`` 按 method 分派（none/rearrangement/
   median_preserving_isotonic），默认方法与历史硬编码行为一致；
2. ``resolve_crossing_settings`` 是运行时唯一解析入口，缺省回落默认方法；
3. YAML 层合同拒绝 legacy ``crossing_method``/``conformal`` 键与非法 method。
"""

import unittest

import numpy as np
import pandas as pd

from forecasting_core.probabilistic_spec import resolve_crossing_settings
from forecasting_core.specs.probabilistic import ProbabilisticConfigSpec
from forecasting_core.tensors import MarginalQuantileForecastTensor
from model_forecasting.forecaster import repair_marginal_quantile_crossing


def _crossed_tensor() -> MarginalQuantileForecastTensor:
    # q10 > q50 < q90 的交叉样本：下界高于锚点、上界低于锚点
    values = np.array(
        [
            [
                [[12.0, 10.0, 8.0]],  # 全部倒挂
                [[3.0, 5.0, 4.0]],  # q90 < q50
            ]
        ]
    )
    return MarginalQuantileForecastTensor(
        values=values,
        levels=(0.1, 0.5, 0.9),
        point_level=0.5,
        series_ids=("s0",),
        forecast_times=pd.DatetimeIndex(["2026-01-01", "2026-01-02"]),
        targets=("load",),
    )


class RepairDispatchTest(unittest.TestCase):
    def test_default_matches_legacy_hardcoded_behavior(self):
        tensor = _crossed_tensor()
        repaired = repair_marginal_quantile_crossing(tensor)
        # 历史行为：下半轴排序并钳到锚点以下，上半轴排序并钳到锚点以上，锚点不动
        np.testing.assert_allclose(
            repaired.values[0, 0, 0, :], [10.0, 10.0, 10.0]
        )
        np.testing.assert_allclose(
            repaired.values[0, 1, 0, :], [3.0, 5.0, 5.0]
        )

    def test_none_returns_tensor_unchanged(self):
        tensor = _crossed_tensor()
        repaired = repair_marginal_quantile_crossing(tensor, method="none")
        self.assertIs(repaired, tensor)

    def test_rearrangement_sorts_without_anchor(self):
        tensor = _crossed_tensor()
        repaired = repair_marginal_quantile_crossing(tensor, method="rearrangement")
        np.testing.assert_allclose(
            repaired.values[0, 0, 0, :], [8.0, 10.0, 12.0]
        )
        np.testing.assert_allclose(
            repaired.values[0, 1, 0, :], [3.0, 4.0, 5.0]
        )

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            repair_marginal_quantile_crossing(
                _crossed_tensor(), method="isotonic"
            )


class ResolveCrossingSettingsTest(unittest.TestCase):
    def test_default_without_crossing_block(self):
        method, report_raw = resolve_crossing_settings({"mode": "quantile"})
        self.assertEqual(method, "median_preserving_isotonic")
        self.assertTrue(report_raw)

    def test_explicit_block(self):
        method, report_raw = resolve_crossing_settings(
            {"crossing": {"method": "none", "report_raw": False}}
        )
        self.assertEqual(method, "none")
        self.assertFalse(report_raw)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            resolve_crossing_settings({"crossing": {"method": "isotonic"}})

    def test_unknown_key_raises(self):
        with self.assertRaises(ValueError):
            resolve_crossing_settings(
                {"crossing": {"method": "none", "bogus": 1}}
            )


class YamlContractTest(unittest.TestCase):
    def test_legacy_crossing_method_key_rejected(self):
        with self.assertRaises(ValueError):
            ProbabilisticConfigSpec.from_mapping(
                {"mode": "quantile", "quantiles": [0.1, 0.5, 0.9],
                 "crossing_method": "isotonic"},
                source="test",
            )

    def test_legacy_conformal_key_rejected(self):
        with self.assertRaises(ValueError):
            ProbabilisticConfigSpec.from_mapping(
                {"mode": "quantile", "quantiles": [0.1, 0.5, 0.9],
                 "conformal": {"method": "none"}},
                source="test",
            )

    def test_invalid_crossing_method_rejected(self):
        with self.assertRaises(ValueError):
            ProbabilisticConfigSpec.from_mapping(
                {"mode": "quantile", "quantiles": [0.1, 0.5, 0.9],
                 "crossing": {"method": "isotonic"}},
                source="test",
            )

    def test_canonical_block_accepted(self):
        spec = ProbabilisticConfigSpec.from_mapping(
            {"mode": "quantile", "quantiles": [0.1, 0.5, 0.9],
             "crossing": {"method": "rearrangement"}},
            source="test",
        )
        self.assertEqual(spec["crossing"]["method"], "rearrangement")


if __name__ == "__main__":
    unittest.main()
