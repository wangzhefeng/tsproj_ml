"""Origin 粒度批统计：值与证据、独立信息集、非法输入及训练接线。"""
from dataclasses import replace
from typing import cast
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_loading import SourceRegistry
from feature_engineering import FeatureCompiler
from pipeline.supervised_design import SupervisedDesignBuilder
import test_compiler_batch_design as panel_fixtures


def advanced_statistics(columns):
    return {
        "percent_change": {"columns": columns, "periods": [1, 2]},
        "time_since": {"columns": columns, "events": ["peak", "trough"]},
        "ewm": {"columns": columns, "halflives": [1.5, 3], "stats": ["mean", "std"]},
    }


class CompilerOriginStatisticsTest(unittest.TestCase):
    def setUp(self):
        self.fixture = panel_fixtures.CompilerBatchDesignTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def config(self, global_scope):
        config = self.fixture._global_config("direct", None) if global_scope else self.fixture._config("direct")
        return replace(config, features=replace(config.features, transformations={
            "advanced": advanced_statistics(["load", "power"] if global_scope else ["load"]),
        }))

    def test_multiple_origins_sparse_horizons_and_training_match(self):
        for global_scope in (False, True):
            with self.subTest(global_scope=global_scope):
                config = self.config(global_scope)
                builder = SupervisedDesignBuilder(config, SourceRegistry(config.data, self.fixture.root))
                # 逆序原点，避免无意依赖调用顺序；后续原点不得污染前面的前缀。
                origins = tuple(cast(pd.Timestamp, self.fixture.times[i]) for i in (25, 12, 19))
                requests = tuple(builder.request(origin) for origin in origins)
                infos = tuple(builder.registry.materialize(request) for request in requests)
                compiler = builder.compiler
                steps = (1, config.problem.horizon)
                expected = tuple(compiler.compile(info, request, horizon_steps=steps)
                                 for info, request in zip(infos, requests))
                with patch.object(compiler, "compile", side_effect=AssertionError("unexpected single fallback")):
                    actual = compiler.compile_batch(infos, requests, horizon_steps=steps)
                    validated = compiler.compile_batch(infos, requests, horizon_steps=steps, proof_mode="validate_only")
                for old, new, validation_only in zip(expected, actual, validated):
                    pd.testing.assert_frame_equal(old.frame, new.frame, check_exact=True)
                    pd.testing.assert_frame_equal(new.frame, validation_only.frame, check_exact=True)
                    self.assertEqual(old.schema, new.schema)
                    self.assertEqual(old.visibility_proof, new.visibility_proof)
                    self.assertEqual(old.source_lineage, new.source_lineage)
                    self.assertEqual(validation_only.visibility_proof, ())
                # 最强消费边界：监督训练接线不止检查 eligibility。
                single_builder = SupervisedDesignBuilder(config, SourceRegistry(config.data, self.fixture.root))
                rows = tuple(single_builder.training_row(origin) for origin in origins)
                batch_rows = builder.training_rows(origins)
                for (designs, labels), (expected_designs, expected_labels) in zip(batch_rows, rows):
                    for design, expected_design in zip(designs, expected_designs):
                        np.testing.assert_array_equal(design, expected_design)
                    np.testing.assert_array_equal(labels, expected_labels)

    def test_same_origin_different_history_prefixes_are_not_shared(self):
        config = self.config(False)
        builder = SupervisedDesignBuilder(config, SourceRegistry(config.data, self.fixture.root))
        request = builder.request(cast(pd.Timestamp, self.fixture.times[25]))
        original = builder.registry.materialize(request)
        frame = pd.read_csv(self.fixture.data_path)
        frame.iloc[10:].to_csv(self.fixture.data_path, index=False)
        shortened = SourceRegistry(config.data, self.fixture.root).materialize(request)
        compiler = FeatureCompiler(config)
        expected = [compiler.compile(info, request) for info in (original, shortened)]
        actual = compiler.compile_batch([original, shortened], [request, request])
        for first, second in zip(expected, actual):
            pd.testing.assert_frame_equal(first.frame, second.frame, check_exact=True)
            self.assertEqual(first.visibility_proof, second.visibility_proof)
        name = "load_ewm_mean_3.0"
        self.assertNotEqual(actual[0].frame[name].iloc[0], actual[1].frame[name].iloc[0])

    def test_percent_change_zero_denominator_is_not_hidden_by_batch(self):
        config = self.config(False)
        frame = pd.read_csv(self.fixture.data_path)
        frame.loc[19, "load"] = 0.0
        frame.to_csv(self.fixture.data_path, index=False)
        builder = SupervisedDesignBuilder(config, SourceRegistry(config.data, self.fixture.root))
        request = builder.request(cast(pd.Timestamp, self.fixture.times[20]))
        info = builder.registry.materialize(request)
        for batch in (False, True):
            with self.subTest(batch=batch), self.assertRaisesRegex(ValueError, "percent_change denominator is zero"):
                if batch:
                    builder.compiler.compile_batch([info], [request])
                else:
                    builder.compiler.compile(info, request)

    def test_ewm_insufficient_history_raises_in_both_paths(self):
        config = self.config(False)
        config = replace(config, features=replace(config.features, target_lags={}, transformations={
            "advanced": {"ewm": {"columns": ["load"], "halflives": [2], "stats": ["std"]}},
        }))
        builder = SupervisedDesignBuilder(config, SourceRegistry(config.data, self.fixture.root))
        request = builder.request(cast(pd.Timestamp, self.fixture.times[0]))
        info = builder.registry.materialize(request)
        for batch in (False, True):
            with self.subTest(batch=batch), self.assertRaisesRegex(ValueError, "produced NaN"):
                if batch:
                    builder.compiler.compile_batch([info], [request])
                else:
                    builder.compiler.compile(info, request)


if __name__ == "__main__":
    unittest.main()
