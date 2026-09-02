# -*- coding: utf-8 -*-
"""DecompositionSpec 配置契约测试（设计文档 §12.1 第 10-15 条）。

legacy ``preprocessing.decomposition_*`` 散列字段写法已随 legacy 配置体系删除，
本文件只覆盖 ``decomposition`` mapping 唯一写法。
"""
import unittest
from types import SimpleNamespace

from decomposition import RESERVED_METHODS, resolve_decomposition_spec


def _cfg(decomposition):
    return SimpleNamespace(decomposition=decomposition)


class ResolveSpecPresetTest(unittest.TestCase):
    def test_no_config_resolves_to_none(self):
        spec = resolve_decomposition_spec(SimpleNamespace())
        self.assertEqual(spec.method, "none")
        self.assertFalse(spec.enabled)

    def test_empty_mapping_resolves_to_none(self):
        spec = resolve_decomposition_spec(_cfg({}))
        self.assertEqual(spec.method, "none")
        self.assertFalse(spec.enabled)

    def test_linear_resolves_to_preset(self):
        spec = resolve_decomposition_spec(_cfg({"method": "linear", "trend_degree": 2}))
        self.assertEqual(spec.method, "linear")
        self.assertEqual(spec.preset.trend_degree, 2)

    def test_stl_resolves_periods(self):
        spec = resolve_decomposition_spec(_cfg({"method": "stl", "periods": [96]}))
        self.assertEqual(spec.method, "stl")
        self.assertEqual(spec.preset.periods, (96,))


class PresetExpansionTest(unittest.TestCase):
    def test_stl_preset_expands_to_components(self):
        spec = resolve_decomposition_spec(
            _cfg({"method": "stl", "periods": [96], "robust": False})
        ).expand_preset()
        self.assertEqual(spec.extractor.type, "stl")
        self.assertEqual(spec.extractor.params["periods"], (96,))
        self.assertFalse(spec.extractor.params["robust"])
        self.assertEqual(spec.trend_forecaster.type, "poly_trend_forecast")
        self.assertEqual(spec.seasonal_forecaster.type, "phase_template")
        self.assertEqual(spec.composer.type, "additive")

    def test_mstl_preset_expands(self):
        spec = resolve_decomposition_spec(
            _cfg({"method": "mstl", "periods": [96, 672]})
        ).expand_preset()
        self.assertEqual(spec.extractor.type, "mstl")
        self.assertEqual(spec.seasonal_forecaster.params["periods"], (96, 672))

    def test_none_stays_none(self):
        spec = resolve_decomposition_spec(_cfg({"method": "none"}))
        self.assertEqual(spec.method, "none")
        self.assertFalse(spec.enabled)
        expanded = spec.expand_preset()
        self.assertIsNone(expanded.extractor)


class SpecFailFastTest(unittest.TestCase):
    def test_custom_not_implemented(self):
        cfg = _cfg({"method": "custom", "extractor": "mstl", "composer": "additive"})
        with self.assertRaisesRegex(ValueError, "not implemented"):
            resolve_decomposition_spec(cfg)

    def test_reserved_methods_rejected(self):
        for method in ["multiplicative", "log_additive", "modal_vmd", "nbeats", "oof_component_feature"]:
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "reserved"):
                    resolve_decomposition_spec(_cfg({"method": method}))

    def test_unknown_key_rejected(self):
        cfg = _cfg({"method": "stl", "periods": [96], "foo": 1})
        with self.assertRaisesRegex(ValueError, "Unknown"):
            resolve_decomposition_spec(cfg)

    def test_stl_requires_exactly_one_period(self):
        cfg = _cfg({"method": "stl", "periods": [96, 672]})
        with self.assertRaisesRegex(ValueError, "exactly 1"):
            resolve_decomposition_spec(cfg)

    def test_mstl_requires_two_periods(self):
        cfg = _cfg({"method": "mstl", "periods": [96]})
        with self.assertRaisesRegex(ValueError, ">= 2"):
            resolve_decomposition_spec(cfg)

    def test_linear_rejects_seasonal_fields(self):
        cfg = _cfg({"method": "linear", "periods": [96]})
        with self.assertRaisesRegex(ValueError, "seasonal fields"):
            resolve_decomposition_spec(cfg)

    def test_non_additive_composition_rejected(self):
        cfg = _cfg({"method": "stl", "periods": [96], "composition": "multiplicative"})
        with self.assertRaisesRegex(ValueError, "additive"):
            resolve_decomposition_spec(cfg)

    def test_params_mapping_is_frozen_copy(self):
        raw = {"method": "custom", "extractor": "mstl", "extractor_params": {"periods": [96]}}
        with self.assertRaises(ValueError):
            resolve_decomposition_spec(_cfg(raw))
        # 原始 dict 未被修改（fail-fast 前无副作用）
        self.assertEqual(raw["extractor_params"], {"periods": [96]})


if __name__ == "__main__":
    unittest.main()
