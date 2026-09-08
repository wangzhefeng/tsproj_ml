"""单模型 pointwise 的 horizon 特征显式开关，覆盖 single/batch。"""
import unittest

import pandas as pd
from data_loading import SourceRegistry
from feature_engineering import FeatureCompiler
import test_feature_visibility_compiler as fixtures


class PointwiseHorizonFeatureTest(unittest.TestCase):
    def test_explicit_disable_keeps_shared_layout_without_horizon_columns(self):
        fixture = fixtures.FeatureVisibilityCompilerTest()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture.write_fixture(global_scope=False)
        request = fixture.request(global_scope=False)
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                config = fixture.build_config(global_scope=False, transformations={
                    "direct": {
                        "layout": "single_model_horizon", "align_to_target": False,
                        "horizon_feature": {"enabled": enabled, "name": "h", "cyclical": True},
                    },
                })
                compiler = FeatureCompiler(config)
                info = SourceRegistry(config.data, fixture.base_dir).materialize(request)
                single = compiler.compile(info, request)
                batch = compiler.compile_batch([info], [request])[0]
                pd.testing.assert_frame_equal(single.frame, batch.frame)
                for name in ("h", "h_sin", "h_cos"):
                    self.assertEqual(name in single.schema.feature_names, enabled)

    def test_non_boolean_enabled_is_rejected(self):
        fixture = fixtures.FeatureVisibilityCompilerTest()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture.write_fixture(global_scope=False)
        config = fixture.build_config(global_scope=False, transformations={
            "direct": {"layout": "single_model_horizon", "horizon_feature": {"enabled": "false"}},
        })
        compiler = FeatureCompiler(config)
        request = fixture.request(global_scope=False)
        info = SourceRegistry(config.data, fixture.base_dir).materialize(request)
        for compile_call in (lambda: compiler.compile(info, request), lambda: compiler.compile_batch([info], [request])):
            with self.assertRaisesRegex(TypeError, "enabled must be a boolean"):
                compile_call()


if __name__ == "__main__":
    unittest.main()
