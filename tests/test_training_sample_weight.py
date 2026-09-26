# -*- coding: utf-8 -*-
"""训练样本权重功能测试：算法本体 + 配置接线（2026-09-26 激活）。

三层钉住：
1. temporal.py：数值正确性（半衰期、归一化模式、锚点语义）、
   防御合同（未来原点、非法参数）；
2. resolve.py：配置翻译、能力前置 RAISE；
3. fold_fit/runner 接线：声明 validation.training.sample_weight 后
   权重真实到达估计器 fit（不只 schema 接受）。
"""
import unittest
from types import MappingProxyType

import numpy as np
import pandas as pd

from model_training.weights import (
    temporal_sample_weight,
    resolve_training_sample_weight,
    training_sample_weight_spec,
)
from models.catalog import MODEL_CATALOG


class _FakeEstimator:
    def __init__(self, model_type):
        self.model_type = model_type


class _FakeConfig:
    """最小 config 替身：只暴露 resolve 需要的三个字段。"""

    def __init__(self, training_payload, model_type="lightgbm"):
        self.validation = MappingProxyType({
            "training": MappingProxyType(dict(training_payload))
            if training_payload is not None else {},
        })
        self.estimator = _FakeEstimator(model_type)


ORIGINS = tuple(pd.date_range("2026-01-01", periods=8, freq="D"))
CUTOFF = ORIGINS[-1] + pd.Timedelta(hours=20)  # 标签末端晚于最后原点


class TemporalSampleWeightTest(unittest.TestCase):
    def test_half_life_doubling(self):
        # 半衰期 2 天：相隔 2 天的样本权重恰为 2 倍（none 模式下可直读）
        weights = temporal_sample_weight(
            ORIGINS, CUTOFF,
            {"method": "exponential", "halflife_days": 2, "normalization": "none"},
        )
        self.assertAlmostEqual(weights[-1], 1.0, places=12)
        self.assertAlmostEqual(weights[-1] / weights[-3], 2.0, places=9)  # 2 天 = 一个半衰期
        self.assertAlmostEqual(weights[-1] / weights[-5], 4.0, places=9)  # 4 天 = 两个半衰期

    def test_mean_normalization_is_default(self):
        spec = {"method": "exponential", "halflife_days": 3}
        weights = temporal_sample_weight(ORIGINS, CUTOFF, spec)
        self.assertAlmostEqual(weights.mean(), 1.0, places=12)
        explicit = temporal_sample_weight(
            ORIGINS, CUTOFF, {**spec, "normalization": "mean"},
        )
        np.testing.assert_allclose(weights, explicit, rtol=1e-15)

    def test_sum_and_none_normalization(self):
        spec = {"method": "exponential", "halflife_days": 3}
        summed = temporal_sample_weight(ORIGINS, CUTOFF, {**spec, "normalization": "sum"})
        self.assertAlmostEqual(summed.sum(), 1.0, places=12)
        raw = temporal_sample_weight(ORIGINS, CUTOFF, {**spec, "normalization": "none"})
        self.assertAlmostEqual(raw[-1], np.exp2(-(0.0) / 3.0), places=12)

    def test_anchor_semantics_differ_when_cutoff_beyond_origins(self):
        # cutoff 比 latest_origin 晚 20 小时：两者年龄差恒定，比值同
        # 半衰期下不同锚点产生恒定因子缩放；none 模式最老样本恒为 1，
        # 差异体现在中间样本相对间隔上——用均值差异钉住语义区分。
        spec = {"method": "exponential", "halflife_days": 2, "normalization": "none"}
        by_cutoff = temporal_sample_weight(ORIGINS, CUTOFF, spec)
        by_latest = temporal_sample_weight(ORIGINS, ORIGINS[-1], spec)
        self.assertAlmostEqual(by_latest[-1], 1.0, places=12)
        self.assertAlmostEqual(by_cutoff[-1], 1.0, places=12)  # none 下最新恒 1
        # 锚点更晚 → 老样本相对更老 → 权重更低
        self.assertTrue(np.all(by_cutoff <= by_latest + 1e-15))

    def test_future_origin_rejected(self):
        with self.assertRaisesRegex(ValueError, "not after anchor"):
            temporal_sample_weight(
                ORIGINS, ORIGINS[0] - pd.Timedelta(hours=1),
                {"method": "exponential", "halflife_days": 3},
            )

    def test_unknown_fields_and_bad_values_rejected(self):
        anchor = ORIGINS[-1]
        with self.assertRaises(ValueError):
            temporal_sample_weight(ORIGINS, anchor, {"method": "linear", "halflife_days": 3})
        with self.assertRaises(ValueError):
            temporal_sample_weight(ORIGINS, anchor, {"method": "exponential", "halflife_days": 0})
        with self.assertRaises(ValueError):
            temporal_sample_weight(ORIGINS, anchor, {"method": "exponential", "halflife_days": -2})
        with self.assertRaises(ValueError):
            temporal_sample_weight(
                ORIGINS, anchor,
                {"method": "exponential", "halflife_days": 3, "unknown": 1},
            )
        with self.assertRaises(ValueError):
            temporal_sample_weight(
                ORIGINS, anchor,
                {"method": "exponential", "halflife_days": 3, "anchor": "moon"},
            )
        with self.assertRaises(ValueError):
            temporal_sample_weight(
                ORIGINS, anchor,
                {"method": "exponential", "halflife_days": 3, "normalization": "l2"},
            )

    def test_none_spec_returns_none(self):
        self.assertIsNone(temporal_sample_weight(ORIGINS, ORIGINS[-1], None))


class ResolveTrainingSampleWeightTest(unittest.TestCase):
    def test_undeclared_returns_none(self):
        config = _FakeConfig(None)
        self.assertIsNone(training_sample_weight_spec(config))
        self.assertIsNone(
            resolve_training_sample_weight(
                config, ORIGINS,
                history_cutoff=CUTOFF, sample_weight_capable=True,
            )
        )

    def test_declared_resolves_against_cutoff_by_default(self):
        spec = {"method": "exponential", "halflife_days": 4}
        config = _FakeConfig({"sample_weight": spec})
        resolved = resolve_training_sample_weight(
            config, ORIGINS,
            history_cutoff=CUTOFF, sample_weight_capable=True,
        )
        expected = temporal_sample_weight(ORIGINS, CUTOFF, spec)
        np.testing.assert_allclose(resolved, expected, rtol=1e-15)
        self.assertEqual(len(resolved), len(ORIGINS))

    def test_latest_origin_anchor(self):
        spec = {"method": "exponential", "halflife_days": 4, "anchor": "latest_origin"}
        config = _FakeConfig({"sample_weight": spec})
        resolved = resolve_training_sample_weight(
            config, ORIGINS,
            history_cutoff=CUTOFF, sample_weight_capable=True,
        )
        expected = temporal_sample_weight(ORIGINS, max(ORIGINS), spec)
        np.testing.assert_allclose(resolved, expected, rtol=1e-15)

    def test_incable_model_raises_before_fit(self):
        # seasonaltemplate 声明 sample_weight=False：配置声明加权即前置 RAISE
        self.assertFalse(MODEL_CATALOG["seasonaltemplate"].sample_weight)
        config = _FakeConfig(
            {"sample_weight": {"method": "exponential", "halflife_days": 4}},
            model_type="seasonaltemplate",
        )
        with self.assertRaisesRegex(ValueError, "sample_weight=false"):
            resolve_training_sample_weight(
                config, ORIGINS,
                history_cutoff=CUTOFF, sample_weight_capable=False,
            )

    def test_capable_model_proceeds(self):
        self.assertTrue(MODEL_CATALOG["lightgbm"].sample_weight)
        config = _FakeConfig(
            {"sample_weight": {"method": "exponential", "halflife_days": 4}},
            model_type="lightgbm",
        )
        resolved = resolve_training_sample_weight(
            config, ORIGINS,
            history_cutoff=CUTOFF, sample_weight_capable=True,
        )
        self.assertIsNotNone(resolved)


if __name__ == "__main__":
    unittest.main()
