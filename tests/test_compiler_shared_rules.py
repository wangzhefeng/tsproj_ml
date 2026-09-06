"""R7 shared transformation rules: frozen pre-refactor frames and fallback."""
import json
from dataclasses import asdict
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_loading import SourceRegistry
from feature_engineering import FeatureCompiler
import test_feature_visibility_compiler as fixtures

GOLDEN = Path(__file__).parent / "fixtures" / "compiler_shared_rules.json"


def snapshot(compiled):
    return {
        "csv": compiled.frame.to_csv(index=False),
        "schema": list(compiled.schema.feature_names),
        "categorical": list(compiled.schema.categorical_names),
        "proofs": json.loads(json.dumps(
            [asdict(proof) for proof in compiled.visibility_proof], default=str,
        )),
        "lineage": json.loads(json.dumps(
            [asdict(source) for source in compiled.source_lineage], default=str,
        )),
    }


class CompilerSharedRulesTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.FeatureVisibilityCompilerTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def compile_case(self, global_scope, fallback):
        self.fixture.write_fixture(global_scope=global_scope)
        advanced = {
            "rolling": {"columns": ["load"], "windows": [2], "stats": ["mean", "std"]},
            "expanding": {"columns": ["load"], "stats": ["mean"]},
            "difference": {"columns": ["load"], "periods": [1]},
            "cyclical": {"columns": ["hour"], "period": 24},
            "interaction": {
                "column_pairs": [["load__lag_2", "humidity__lag_2"]],
                "operations": ["add", "subtract", "multiply", "divide"],
            },
            "polynomial": {"columns": ["load__lag_2"], "degree": 3},
        }
        if fallback:
            advanced.update({
                "percent_change": {"columns": ["load"], "periods": [1]},
                "time_since": {"columns": ["load"], "events": ["peak"]},
                "ewm": {"columns": ["load"], "halflives": [2], "stats": ["mean", "std"]},
            })
        config = self.fixture.build_config(
            global_scope=global_scope,
            transformations={
                "direct": {"layout": "single_model_horizon", "align_to_target": False,
                           "horizon_feature": {"name": "h", "cyclical": True}},
                "advanced": advanced,
                "interactions": {"product": ["load__lag_2", "humidity__lag_2", "dt_hour"]},
            },
        )
        compiler = FeatureCompiler(config)
        request = self.fixture.request(global_scope=global_scope)
        info = SourceRegistry(config.data, self.fixture.base_dir).materialize(request)
        single = compiler.compile(info, request)
        with patch.object(compiler, "compile", wraps=compiler.compile) as single_path:
            if fallback:
                with self.assertWarnsRegex(RuntimeWarning, "falling back"):
                    batch = compiler.compile_batch([info], [request])[0]
                self.assertEqual(single_path.call_count, 1)
            else:
                batch = compiler.compile_batch([info], [request])[0]
                single_path.assert_not_called()
        # 改前 global single 的历史派生缓存跨 identity 复用；分别冻结两条
        # 路径，不在行为不变重构中修正该数值问题。fallback 本来就走 single。
        if not global_scope or fallback:
            pd.testing.assert_frame_equal(single.frame, batch.frame, check_exact=True)
        self.assertEqual(single.visibility_proof, batch.visibility_proof)
        self.assertEqual(single.source_lineage, batch.source_lineage)
        return {"single": snapshot(single), "batch": snapshot(batch)}

    def test_frozen_frames_schema_proofs_and_fallback(self):
        expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
        for global_scope in (False, True):
            for fallback in (False, True):
                key = f"global={global_scope},fallback={fallback}"
                with self.subTest(case=key):
                    actual = self.compile_case(global_scope, fallback)
                    # Temporary fixture roots are lineage provenance, not values.
                    for result in actual.values():
                        for source in result["lineage"]:
                            for field, value in source.items():
                                if isinstance(value, str):
                                    source[field] = value.replace(str(self.fixture.base_dir), "<fixture>")
                    self.assertEqual(actual, expected[key])

    def test_shared_rules_preserve_scalar_and_array_errors(self):
        compiler = FeatureCompiler(self.fixture.build_config())
        cases = (
            ("_compile_direct_transformations", [], TypeError, "transformations.direct must be a mapping"),
            ("_compile_direct_transformations", {"layout": "unknown"}, ValueError, "unsupported direct layout: 'unknown'"),
            ("_compile_direct_transformations", {"layout": "single_model_horizon", "horizon_feature": []}, TypeError, "transformations.direct.horizon_feature must be a mapping"),
            ("_compile_cyclical", {"period": True}, ValueError, "cyclical.period must be positive"),
            ("_compile_cyclical", {"period": 24, "columns": ["absent"]}, ValueError, "cyclical references unknown feature: 'absent'"),
            ("_compile_interaction_spec", {"column_pairs": "x"}, TypeError, "interaction.column_pairs must be a sequence"),
            ("_compile_polynomial", {"degree": True}, ValueError, "polynomial.degree must be an integer >= 2"),
            ("_compile_named_interactions", {"product": ["x"]}, ValueError, "interaction 'product' requires at least two features"),
            ("_compile_named_interactions", {"product": ["x", "y"]}, ValueError, "interaction 'product' requires finite numeric features"),
        )
        for name, spec, error, message in cases:
            for vectorized in (False, True):
                with self.subTest(rule=name, spec=spec, vectorized=vectorized):
                    row = {"x": [1.0], "y": [np.inf]} if vectorized else {"x": 1.0, "y": np.inf}
                    with self.assertRaises(error) as raised:
                        getattr(compiler, name)(row, spec, vectorized=vectorized)
                    self.assertEqual(str(raised.exception), message)

    def test_shared_rules_keep_numeric_backends_and_feature_names(self):
        compiler = FeatureCompiler(self.fixture.build_config())
        for vectorized in (False, True):
            with self.subTest(vectorized=vectorized):
                value = np.asarray([1.1, -2.3]) if vectorized else 1.1
                row = {"x": value, "y": value}
                compiler._compile_polynomial(row, {"columns": ["x"], "degree": 7}, vectorized=vectorized)
                for degree in range(2, 8):
                    np.testing.assert_array_equal(row[f"x_pow_{degree}"], value ** degree)
                compiler._compile_interaction_spec(row, {
                    "column_pairs": [["x", "y"]], "operations": ["subtract"],
                }, vectorized=vectorized)
                self.assertIn("x_substract_y", row)  # 历史拼写是输出合同，不修正。

    def test_empty_batch_preserves_direct_validation(self):
        config = self.fixture.build_config(transformations={"direct": {"layout": "unknown"}})
        compiler = FeatureCompiler(config)
        with self.assertRaisesRegex(ValueError, "unsupported direct layout"):
            compiler.compile_batch([], [])


if __name__ == "__main__":
    unittest.main()
