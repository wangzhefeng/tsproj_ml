# -*- coding: utf-8 -*-
"""DecompositionSpec 配置契约测试（设计文档 §12.1 第 10-15 条）。"""
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

from config.config_loader import load_yaml_config
from decomposition import RESERVED_METHODS, DecompositionSpec, resolve_decomposition_spec


def _legacy_cfg(method="none", **overrides):
    """构造旧写法的 cfg 对象（字段默认值与 PreprocessingConfig 一致）。"""
    defaults = dict(
        decomposition_method=method,
        decomposition_periods=[],
        decomposition_robust=True,
        decomposition_trend_degree=1,
        decomposition_trend_forecast="polynomial",
        decomposition_damping=0.98,
        decomposition_trend_lookback=28,
        decomposition_seasonal_cycles=4,
        decomposition={},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class ResolveSpecLegacyTest(unittest.TestCase):
    def test_no_config_resolves_to_none(self):
        spec = resolve_decomposition_spec(_legacy_cfg())
        self.assertEqual(spec.method, "none")
        self.assertFalse(spec.enabled)

    def test_legacy_linear_resolves_to_preset(self):
        spec = resolve_decomposition_spec(_legacy_cfg("linear", decomposition_trend_degree=2))
        self.assertEqual(spec.method, "linear")
        self.assertEqual(spec.preset.trend_degree, 2)

    def test_legacy_stl_resolves_periods(self):
        spec = resolve_decomposition_spec(_legacy_cfg("stl", decomposition_periods=[96]))
        self.assertEqual(spec.method, "stl")
        self.assertEqual(spec.preset.periods, (96,))


class ResolveSpecNewTest(unittest.TestCase):
    def test_new_stl_preset_expands_to_components(self):
        cfg = SimpleNamespace(
            decomposition={"method": "stl", "periods": [96], "robust": False},
            decomposition_method="none",
        )
        spec = resolve_decomposition_spec(cfg).expand_preset()
        self.assertEqual(spec.extractor.type, "stl")
        self.assertEqual(spec.extractor.params["periods"], (96,))
        self.assertFalse(spec.extractor.params["robust"])
        self.assertEqual(spec.trend_forecaster.type, "poly_trend_forecast")
        self.assertEqual(spec.seasonal_forecaster.type, "phase_template")
        self.assertEqual(spec.composer.type, "additive")

    def test_new_mstl_preset_expands(self):
        cfg = SimpleNamespace(decomposition={"method": "mstl", "periods": [96, 672]}, decomposition_method="none")
        spec = resolve_decomposition_spec(cfg).expand_preset()
        self.assertEqual(spec.extractor.type, "mstl")
        self.assertEqual(spec.seasonal_forecaster.params["periods"], (96, 672))

    def test_new_none_stays_none(self):
        cfg = SimpleNamespace(decomposition={"method": "none"}, decomposition_method="none")
        spec = resolve_decomposition_spec(cfg)
        self.assertEqual(spec.method, "none")
        self.assertFalse(spec.enabled)
        expanded = spec.expand_preset()
        self.assertIsNone(expanded.extractor)


class SpecEquivalenceTest(unittest.TestCase):
    def test_legacy_and_new_same_semantics_produce_equal_specs(self):
        legacy = resolve_decomposition_spec(_legacy_cfg(
            "stl", decomposition_periods=[96], decomposition_robust=True,
            decomposition_trend_degree=1, decomposition_trend_forecast="polynomial",
            decomposition_damping=0.98, decomposition_trend_lookback=28,
            decomposition_seasonal_cycles=4,
        ))
        new = resolve_decomposition_spec(SimpleNamespace(
            decomposition={"method": "stl", "periods": [96], "robust": True,
                           "trend_degree": 1, "trend_forecast": "polynomial",
                           "damping": 0.98, "trend_lookback": 28, "seasonal_cycles": 4},
            decomposition_method="none",
        ))
        self.assertEqual(legacy, new)

    def test_conflicting_legacy_and_new_raises(self):
        cfg = SimpleNamespace(
            decomposition={"method": "stl", "periods": [96]},
            decomposition_method="linear",
            decomposition_periods=[],
            decomposition_robust=True,
            decomposition_trend_degree=1,
            decomposition_trend_forecast="polynomial",
            decomposition_damping=0.98,
            decomposition_trend_lookback=28,
            decomposition_seasonal_cycles=4,
        )
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            resolve_decomposition_spec(cfg)


class SpecFailFastTest(unittest.TestCase):
    def test_custom_not_implemented(self):
        cfg = SimpleNamespace(
            decomposition={"method": "custom", "extractor": "mstl", "composer": "additive"},
            decomposition_method="none",
        )
        with self.assertRaisesRegex(ValueError, "not implemented"):
            resolve_decomposition_spec(cfg)

    def test_custom_with_legacy_fields_rejected(self):
        cfg = SimpleNamespace(
            decomposition={"method": "custom"},
            decomposition_method="stl",
            decomposition_periods=[96],
        )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            resolve_decomposition_spec(cfg)

    def test_reserved_methods_rejected(self):
        for method in ["multiplicative", "log_additive", "modal_vmd", "nbeats", "oof_component_feature"]:
            with self.subTest(method=method):
                cfg = SimpleNamespace(decomposition={"method": method}, decomposition_method="none")
                with self.assertRaisesRegex(ValueError, "reserved"):
                    resolve_decomposition_spec(cfg)

    def test_unknown_new_key_rejected(self):
        cfg = SimpleNamespace(decomposition={"method": "stl", "periods": [96], "foo": 1}, decomposition_method="none")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            resolve_decomposition_spec(cfg)

    def test_stl_requires_exactly_one_period(self):
        cfg = SimpleNamespace(decomposition={"method": "stl", "periods": [96, 672]}, decomposition_method="none")
        with self.assertRaisesRegex(ValueError, "exactly 1"):
            resolve_decomposition_spec(cfg)

    def test_mstl_requires_two_periods(self):
        cfg = SimpleNamespace(decomposition={"method": "mstl", "periods": [96]}, decomposition_method="none")
        with self.assertRaisesRegex(ValueError, ">= 2"):
            resolve_decomposition_spec(cfg)

    def test_linear_rejects_seasonal_fields(self):
        cfg = SimpleNamespace(decomposition={"method": "linear", "periods": [96]}, decomposition_method="none")
        with self.assertRaisesRegex(ValueError, "seasonal fields"):
            resolve_decomposition_spec(cfg)

    def test_non_additive_composition_rejected(self):
        cfg = SimpleNamespace(decomposition={"method": "stl", "periods": [96], "composition": "multiplicative"}, decomposition_method="none")
        with self.assertRaisesRegex(ValueError, "additive"):
            resolve_decomposition_spec(cfg)

    def test_params_mapping_is_frozen_copy(self):
        raw = {"method": "custom", "extractor": "mstl", "extractor_params": {"periods": [96]}}
        cfg = SimpleNamespace(decomposition=raw, decomposition_method="none")
        with self.assertRaises(ValueError):
            resolve_decomposition_spec(cfg)
        # 原始 dict 未被修改（fail-fast 前无副作用）
        self.assertEqual(raw["extractor_params"], {"periods": [96]})


if __name__ == "__main__":
    unittest.main()
if __name__ == "__main__":
    unittest.main()
