"""目标历史读取不依赖最终预测天气；source 投影仍严格 as-of。"""
from dataclasses import replace
import unittest

import pandas as pd
from data_loading import SourceRegistry
from forecasting_core.specs import DataSpec
from model_pipeline.supervised_design import SupervisedDesignBuilder
import test_feature_visibility_compiler as fixtures


class TargetHistoryProjectionTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.FeatureVisibilityCompilerTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.fixture.write_fixture(global_scope=False)
        config = self.fixture.build_config(global_scope=False)
        sources = tuple(
            replace(s, future_path="unavailable-weather.csv") if s.name == "weather" else s
            for s in config.data.sources
        )
        self.config = replace(config, data=DataSpec(sources))
        self.registry = SourceRegistry(self.config.data, self.fixture.base_dir)

    def test_target_history_needs_no_unrequested_weather(self):
        builder = SupervisedDesignBuilder(self.config, self.registry)
        origin = pd.Timestamp("2026-01-01 03:00:00")
        times = builder.target_history_times(origin)
        history = builder.target_history(origin)
        self.assertEqual(times[-1], origin)
        self.assertTrue((times <= origin).all())
        self.assertEqual(list(history.forecast_times), list(times))
        with self.assertRaises(FileNotFoundError):
            self.registry.materialize(builder.request(origin))

    def test_source_projection_rejects_unknown_duplicate_and_empty_names(self):
        request = self.fixture.request(global_scope=False)
        for names in (("not_configured",), (), ("target_history", "target_history")):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.registry.materialize(request, source_names=names)


if __name__ == "__main__":
    unittest.main()
